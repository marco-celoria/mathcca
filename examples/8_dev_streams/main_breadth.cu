/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include <mathcca/host_matrix.h>
#include <mathcca/device_matrix.h>
#include <mathcca/norm.h>
#include <mathcca/detail/norm_impl.h>
#include <iostream>
#include <mathcca/fill_rand.h>
#include <mathcca/fill_const.h>
#include <mathcca/detail/copy_impl.h>
#include <mathcca/detail/reduce_sum_impl.h>
#include <mathcca/reduce_sum.h>
#include <mathcca/copy.h>
#include <mathcca/host_iterator.h>
#include <vector>

#ifdef _OPENMP
 #include <omp.h>
#endif

template<std::floating_point T>
__global__ void addTo_kernel_ntimes(T* __restrict accululator, const T* __restrict to_be_op, const std::size_t size, const int ntimes) {
  const auto idx{static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x)};
  if(idx < size) {
    for (auto n=0; n < ntimes; ++n) {
      accululator[idx]+= to_be_op[idx];
    }
  }
}


int main(int argc, char **argv)  {

  constexpr int NSTREAM= 4;
  constexpr int NTIMES= 3000;
  constexpr int NITER= 100;

  constexpr unsigned int threads{1024};
  constexpr std::size_t l{1'000};
  constexpr std::size_t m{1'000'000};
  constexpr std::size_t nElem{l * m};
  constexpr std::size_t iElem{nElem/NSTREAM};

#ifdef _USE_DOUBLE_PRECISION
  std::cout << "USE DOUBLE PRECISION\n";
  using value_type= double;
#else
  std::cout << "USE SINGLE PRECISION\n";
  using value_type= float;
#endif
 
#ifdef _OPENMP
  int num_threads = 0;
  #pragma omp parallel reduction(+:num_threads)
  num_threads += 1;
  std::cout << "Running with " << num_threads << " OMP threads\n";
#endif

  value_type avg_speedup{0};

  mathcca::host_matrix<value_type>   hA{l, m};
  mathcca::host_matrix<value_type>   hB{l, m};
  mathcca::host_matrix<value_type>   hA_S{l, m};
  mathcca::host_matrix<value_type>   hA_O{l, m};
  
  mathcca::fill_rand(hA.begin(), hA.end());
  mathcca::fill_rand(hB.begin(), hB.end());

  mathcca::device_matrix<value_type> dA{l, m};
  mathcca::device_matrix<value_type> dB{l, m};

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  std::vector<value_type> sumsA_O(NSTREAM);
  std::vector<value_type> sumsB_O(NSTREAM);
  std::vector<cudaStream_t> streams(NSTREAM);
  
  for (int i = 0; i < NSTREAM; ++i) {
    cudaStreamCreate(&streams[i]);
  }
 
  auto check{hA[0]};
  for (auto n= 0; n < NTIMES; ++n) {
    check+= hB[0];
  }

  for (auto iter= 0; iter < NITER; ++iter) {
    auto sumA_S= static_cast<value_type>(0);
    auto sumB_S= static_cast<value_type>(0);
    auto sumA_O= static_cast<value_type>(0);
    auto sumB_O= static_cast<value_type>(0);

    cudaEventRecord(start, 0);
    mathcca::copy(hA.cbegin(), hA.cend(), dA.begin());
    mathcca::copy(hB.cbegin(), hB.cend(), dB.begin());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float memcpy_h2d_time;
    cudaEventElapsedTime(&memcpy_h2d_time, start, stop);

    cudaEventRecord(start, 0);
    sumA_S= mathcca::reduce_sum<decltype(dA.begin()), value_type, threads>(dA.begin(), dA.end(), static_cast<value_type>(0));
    sumB_S= mathcca::reduce_sum<decltype(dB.begin()), value_type, threads>(dB.begin(), dB.end(), static_cast<value_type>(0));
    constexpr auto nblocks{static_cast<unsigned int>((nElem + static_cast<std::size_t>(threads) - 1)/(static_cast<std::size_t>(threads)))};
    addTo_kernel_ntimes<<<nblocks, threads>>>(dA.data(), dB.data(), nElem, NTIMES);
    getLastCudaError("addTo_kernel_ntimes() execution failed.\n");
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);

    cudaEventRecord(start, 0);
    mathcca::copy(dA.cbegin(), dA.cend(), hA_S.begin());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float memcpy_d2h_time;
    cudaEventElapsedTime(&memcpy_d2h_time, start, stop);

    float itotal = kernel_time + memcpy_h2d_time + memcpy_d2h_time;
    std::cout << "\nMeasured timings (throughput):\n";
    std::cout << "\tMemcpy host to device:\t " << memcpy_h2d_time << " ms\n";
    std::cout << "\tMemcpy device to host:\t " << memcpy_d2h_time << " ms\n";
    std::cout << "\tKernel:\t " << kernel_time << " ms\n";
    std::cout << "\tTotal:\t "  << itotal      << " ms\n";

    cudaEventRecord(start, 0);

    for (int i = 0; i < NSTREAM; ++i) {
      int ioffset = i * iElem;
      mathcca::copy(hA.cbegin() + static_cast<std::ptrdiff_t>(ioffset), hA.cbegin() + static_cast<std::ptrdiff_t>(ioffset + iElem), dA.begin() + static_cast<std::ptrdiff_t>(ioffset), streams[i]);
      
      mathcca::copy(hB.cbegin() + static_cast<std::ptrdiff_t>(ioffset), hB.cbegin() + static_cast<std::ptrdiff_t>(ioffset + iElem), dB.begin() + static_cast<std::ptrdiff_t>(ioffset), streams[i]);
    }
    // launch a kernel in each stream
    for (int i = 0; i < NSTREAM; ++i) {
      int ioffset = i * iElem;
      sumsA_O[i]= mathcca::detail::reduce_sum(mathcca::Cuda(), 
                    dA.cbegin().get() + static_cast<std::ptrdiff_t>(ioffset),
                    dA.cbegin().get() + static_cast<std::ptrdiff_t>(ioffset + iElem), 
                    static_cast<value_type>(0), 
                    streams[i]);
      
      sumsB_O[i]= mathcca::detail::reduce_sum(mathcca::Cuda(), 
                    dB.cbegin().get() + static_cast<std::ptrdiff_t>(ioffset),
                    dB.cbegin().get() + static_cast<std::ptrdiff_t>(ioffset + iElem), 
                    static_cast<value_type>(0), 
                    streams[i]);
      
      constexpr auto iblocks{static_cast<unsigned int>((iElem + static_cast<std::size_t>(threads) - 1)/(static_cast<std::size_t>(threads)))};
      addTo_kernel_ntimes<<<iblocks, threads, 0, streams[i]>>>(dA.begin().get() + static_cast<std::ptrdiff_t>(ioffset), dB.begin().get() + static_cast<std::ptrdiff_t>(ioffset), iElem, NTIMES); 
      getLastCudaError("addTo_kernel_ntimes() execution failed.\n");
    
    }

    // enqueue asynchronous transfers from the device
    for (int i = 0; i < NSTREAM; ++i) {
      sumA_O+= sumsA_O[i];
      sumB_O+= sumsB_O[i];
      int ioffset = i * iElem;
      mathcca::copy(dA.cbegin() + static_cast<std::ptrdiff_t>(ioffset), dA.cbegin() + static_cast<std::ptrdiff_t>(ioffset + iElem), hA_O.begin() + static_cast<std::ptrdiff_t>(ioffset), streams[i]);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float execution_time;
    cudaEventElapsedTime(&execution_time, start, stop);

    std::cout << "\nActual results from overlapped data transfers:\n";
    std::cout << "\toverlap with " << NSTREAM << " streams : " << execution_time << " ms\n";
    auto speedup= ((itotal - execution_time) * 100.0f) / itotal;
    avg_speedup+= speedup;
    std::cout << "\tspeedup: " << speedup << " %\n";
    std::cout << "Check:\n";
    std::cout << "Sum A non-overlapped transfer: " << sumA_S << " - Sum A overlapped transfer: " << sumA_O << " - Difference: " << (sumA_S - sumA_O) << "\n";
    std::cout << "Sum B non-overlapped transfer: " << sumB_S << " - Sum B overlapped transfer: " << sumB_O << " - Difference: " << (sumB_S - sumB_O) << "\n";
    std::cout << "Resulting A matrices for non-overlapped and overlapped transfer correspond? " << std::boolalpha << (hA_S == hA_O) << std::noboolalpha << "\n";
    std::cout << "Host result: " << check << " - Non-overlapped result: " << hA_S[0] << " - Overlapped result: " << hA_O[0] << "\n";
  }

  std::cout << "After " << NITER << " iterations, the average speedup is: " << avg_speedup/NITER << "%\n";
  // destroy events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // destroy streams
  for (int i = 0; i < NSTREAM; ++i) {
    cudaStreamDestroy(streams[i]);
  }
  
}


