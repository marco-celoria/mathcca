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
  constexpr int NTIMES= 2000;
  constexpr unsigned int threads{1024};
  constexpr std::size_t l{1'000};
  constexpr std::size_t m{1'000'000};
  constexpr std::size_t nElem{l * m};
  constexpr std::size_t iElem{nElem/NSTREAM};
#ifdef _USE_DOUBLE_PRECISION
  using value_type= double;
#else
  using value_type= float;
#endif
  constexpr std::size_t nBytes{sizeof(value_type) * nElem};

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
  
  std::vector<cudaStream_t> streams(NSTREAM);
  std::vector<value_type> sumsA_O(NSTREAM);
  std::vector<value_type> sumsB_O(NSTREAM);
  for (int i = 0; i < NSTREAM; ++i) {
    cudaStreamCreate(&streams[i]);
  }
  
  auto check=hA[0];
  for (auto n=0; n < NTIMES; ++n) {
    check+= hB[0];
  }

  for (auto iter=0; iter < 25; ++iter) {
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
    std::cout << " Memcpy host to device\t: " << memcpy_h2d_time << " ms ( " << (nBytes * 1e-6) / memcpy_h2d_time << " GB/s)\n";
    std::cout << " Memcpy device to host\t: " << memcpy_d2h_time << " ms ( " << (nBytes * 1e-6) / memcpy_d2h_time << " GB/s)\n";
    std::cout << " Kernel\t\t\t: " << kernel_time << " ms ( " << (nBytes * 2e-6) / kernel_time << " GB/s)\n";
    std::cout << " Total\t\t\t: " <<  itotal << "ms ( " << (nBytes * 2e-6) / itotal << " GB/s)\n";
    std::cout << "sumA_S= " << sumA_S << "\n";
    std::cout << "sumB_S= " << sumB_S << "\n";
 
    cudaEventRecord(start, 0);

    // initiate all asynchronous transfers to the device
    for (int i = 0; i < NSTREAM; ++i) {
      int ioffset = i * iElem;
      mathcca::detail::copy(mathcca::CudaHtoDcpy(),
                           hA.cbegin().get() + static_cast<std::ptrdiff_t>(ioffset),
                           hA.cbegin().get() + static_cast<std::ptrdiff_t>(ioffset + iElem), 
                           dA.begin().get()  + static_cast<std::ptrdiff_t>(ioffset), 
                           streams[i]);
      
      mathcca::detail::copy(mathcca::CudaHtoDcpy(),
                            hB.cbegin().get() + static_cast<std::ptrdiff_t>(ioffset), 
                            hB.cbegin().get() + static_cast<std::ptrdiff_t>(ioffset + iElem), 
                            dB.begin().get()  + static_cast<std::ptrdiff_t>(ioffset), 
                            streams[i]);
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
      addTo_kernel_ntimes<<<iblocks, threads, 0, streams[i]>>>(dA.begin().get()  + static_cast<std::ptrdiff_t>(ioffset), 
       		                                               dB.begin().get()  + static_cast<std::ptrdiff_t>(ioffset), iElem, NTIMES);  
    }

    // enqueue asynchronous transfers from the device
    for (int i = 0; i < NSTREAM; ++i) {
      sumA_O+= sumsA_O[i];
      sumB_O+= sumsB_O[i];
      int ioffset = i * iElem;
      mathcca::detail::copy(mathcca::CudaDtoHcpy(),
                    dA.cbegin().get() + static_cast<std::ptrdiff_t>(ioffset),
                    dA.cbegin().get() + static_cast<std::ptrdiff_t>(ioffset + iElem),
                    hA_O.begin().get()  + static_cast<std::ptrdiff_t>(ioffset), streams[i]);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float execution_time;
    cudaEventElapsedTime(&execution_time, start, stop);
    std::cout << "\nActual results from overlapped data transfers:\n";
    std::cout << " overlap with " << NSTREAM << " streams : " << execution_time << " ms ( " << (nBytes * 2e-6) / execution_time << " GB/s)\n";
    std::cout << " speedup: " << ((itotal - execution_time) * 100.0f) / itotal << " % \n";
    std::cout << sumA_S << " " << sumA_O << "\n";
    std::cout << sumB_S << " " << sumB_O << "\n";
    std::cout << "hA_S == hA_O? " << std::boolalpha << (hA_S == hA_O) << std::noboolalpha << "\n";
    std::cout << check << " " << hA_S[0] << " " << hA_O[0] << "\n";
  }
  
  // destroy events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // destroy streams
  for (int i = 0; i < NSTREAM; ++i) {
    cudaStreamDestroy(streams[i]);
  }
  
}


