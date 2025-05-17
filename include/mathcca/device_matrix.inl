#include <mathcca/device_helper.h>
#include <mathcca/detail/nextPow2.h>
#include <mathcca/detail/shared_memory_proxy.h>
#include <mathcca/detail/reduce_sum_impl.h>
#include <cstddef>
#include <cstdio>
#include "cooperative_groups.h"
#include "cooperative_groups/reduce.h"
namespace cg = cooperative_groups;

namespace mathcca {

    template<std::floating_point T>
    void swap(device_matrix<T>& a, device_matrix<T>& b) {
      auto tmp {std::move(a)};
      a= std::move(b);
      b= std::move(tmp);
    }
    
    template<std::floating_point T>
    constexpr bool check_equal_size(const device_matrix<T>& lhs, const device_matrix<T>& rhs) {
      if (lhs.num_rows() == rhs.num_rows() && lhs.num_cols() == rhs.num_cols())
        return true;
      return false;
    }

    template <std::floating_point T>
    __global__ void cg_count_if_diffs_kernel(const T* __restrict lhs, const T* __restrict rhs, T* __restrict odata, 
		    const std::size_t size, const T tol) {
      // Shared memory for intermediate steps
      auto sdata = detail::shared_memory_proxy<T>();
      // Handle to thread block group
      cg::thread_block cta = cg::this_thread_block();
      // Handle to tile in thread block
      cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);
      const auto ctaSize = static_cast<std::size_t>(cta.size());
      const auto numCtas = static_cast<std::size_t>(gridDim.x);
      const auto threadRank = static_cast<std::size_t>(cta.thread_rank());
      const auto threadIndex = static_cast<std::size_t>((blockIdx.x * ctaSize) + threadRank);
      auto threadVal{static_cast<T>(0)};
      {
        auto i{threadIndex};
        auto indexStride{numCtas * ctaSize};
        while (i < size) {
          threadVal += (fabs(lhs[i] - rhs[i]) > tol) ? static_cast<T>(1) : static_cast<T>(0);
          i += indexStride;
        }
        sdata[threadRank] = threadVal;
      }
      // Wait for all tiles to finish and reduce within CTA
      {
        auto ctaSteps = static_cast<std::size_t>(tile.meta_group_size());
        auto ctaIndex = static_cast<std::size_t>(ctaSize >> 1);
        while (ctaIndex >= 32) {
          cta.sync();
          if (threadRank < ctaIndex) {
            threadVal += sdata[threadRank + ctaIndex];
            sdata[threadRank] = threadVal;
          }
          ctaSteps >>= 1;
          ctaIndex >>= 1;
        }
      }
      // Shuffle redux instead of smem redux
      {
        cta.sync();
        if (tile.meta_group_rank() == 0) {
          threadVal = cg::reduce(tile, threadVal, cg::plus<T>()); //cg_reduce_n(threadVal, tile);
        }
      }
      if (threadRank == 0)
        odata[blockIdx.x] = threadVal;
    }
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    T count_if_diffs(const T* lhs, const T* rhs, const std::size_t size, const T tol) {
      static_assert(THREAD_BLOCK_DIM <= 1024);
      constexpr unsigned int maxThreads{THREAD_BLOCK_DIM};
      unsigned int threads{size < (static_cast<std::size_t>(maxThreads) * 2) ? detail::nextPow2((size + 1) / 2) : maxThreads};
      unsigned int blocks{static_cast<unsigned int>((size + static_cast<std::size_t>(threads * 2 - 1)) / static_cast<std::size_t>(threads * 2))};
      auto gpu_result{static_cast<T>(0)};
      T* d_odata = nullptr;
      T* d_intermediateSums = nullptr;
      checkCudaErrors(cudaMalloc((void **)&d_odata, blocks * sizeof(T)));
      checkCudaErrors(cudaMalloc((void **)&d_intermediateSums, sizeof(T) * blocks));
      dim3 dimBlock(threads, 1, 1);
      dim3 dimGrid(blocks, 1, 1);
      unsigned int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
      cg_count_if_diffs_kernel<T><<<dimGrid, dimBlock, smemSize>>>(lhs, rhs, d_odata,  size, tol);
      // sum partial block sums on GPU
      unsigned int s{blocks};
      while (s > 1) {
        threads = (s < maxThreads * 2) ? detail::nextPow2((s + 1) / 2) : maxThreads;
        blocks  = (s + (threads * 2 - 1)) / (threads * 2);
        //std::cout << "s = " << s << "; threads = " << threads << "; blocks = " << blocks << "\n";
        checkCudaErrors(cudaMemcpy(d_intermediateSums, d_odata, s * sizeof(T), cudaMemcpyDeviceToDevice));
	mathcca::detail::cg_reduce_kernel<T><<<blocks, threads, smemSize>>>(d_intermediateSums, d_odata, s, static_cast<T>(0));
        s = (s + (threads * 2 - 1)) / (threads * 2);
      }
      checkCudaErrors(cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaFree(d_odata));
      checkCudaErrors(cudaFree(d_intermediateSums));
      return gpu_result;
    }

    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    constexpr bool operator==(const device_matrix<T>& lhs, const device_matrix<T>& rhs) {
      if (!check_equal_size(lhs, rhs))
        return false;
      
      using size_type= typename device_matrix<T>::size_type;
      using value_type= T;
      const size_type size{lhs.size()};
      auto diffs = count_if_diffs<T, THREAD_BLOCK_DIM>(lhs.data(), rhs.data(), size, device_matrix<T>::tol());
      return diffs? false : true;
    }
    
    // Standard kernel  
    template<std::floating_point T>
    __global__ void addTo_kernel(T* __restrict accululator, const T* __restrict to_be_op, const std::size_t size) {
      const auto idx{static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x)};
      if(idx < size) {
        accululator[idx]+= to_be_op[idx];
      }
    }
    
    // Increasing ILP kernel 
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    __global__ void subTo_kernel(T* __restrict accululator, const T* __restrict to_be_op, const std::size_t size) {
      const auto off{static_cast<std::size_t>(2 * THREAD_BLOCK_DIM * blockIdx.x + threadIdx.x)};
      #pragma unroll 2
      for (auto i = 0; i < 2; i++) {
        const std::size_t idx{off + i * THREAD_BLOCK_DIM};
        if(idx < size) {
          accululator[idx]-= to_be_op[idx];
        }
      }
    }
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    __global__ void mulScalarTo_kernel(T* __restrict accululator, const T to_be_op, const std::size_t size) {
      const auto off{static_cast<std::size_t>(2 * THREAD_BLOCK_DIM * blockIdx.x + threadIdx.x)};
      #pragma unroll 2
      for (auto i = 0; i < 2; i++) {
        const std::size_t idx{off + i * THREAD_BLOCK_DIM};
        if(idx < size) {
          accululator[idx]*= to_be_op;
        }
      }
    }

    // Increasing ILP kernel using the class inside the kernel
    template<std::floating_point T, typename A, typename E, unsigned int THREAD_BLOCK_DIM>
    __global__ void mulTo_kernel(device_matrix<T,A,E>& accululator, const device_matrix<T,A,E>& to_be_op) { 
      const auto off{static_cast<std::size_t>(2 * THREAD_BLOCK_DIM * blockIdx.x + threadIdx.x)};
      #pragma unroll 2
      for (auto i = 0; i < 2; i++) {
        const std::size_t idx{off + i * THREAD_BLOCK_DIM};
        if(idx < accululator.size()) {
          accululator[idx]*= to_be_op[idx];
        }
      }
    }
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    device_matrix<T> operator+(device_matrix<T>&& res, const device_matrix<T>& rhs) {
      std::cout <<"operator+ rvalue\n";
      static_assert(THREAD_BLOCK_DIM <= 1024);
      return std::forward<device_matrix<T>>((res).template operator+= <THREAD_BLOCK_DIM>(rhs));
    }
     
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    device_matrix<T> operator+(const device_matrix<T>& lhs, const device_matrix<T>& rhs) {
      std::cout <<"operator+ lvalue\n";
      static_assert(THREAD_BLOCK_DIM <= 1024);
      auto res{lhs};
      return std::forward<device_matrix<T>>((res). template operator+= <THREAD_BLOCK_DIM>(rhs));
    }
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    device_matrix<T> operator-(device_matrix<T>&& res, const device_matrix<T>& rhs) {
      std::cout <<"operator- rvalue\n";
      static_assert(THREAD_BLOCK_DIM <= 1024);
      return std::forward<device_matrix<T>>((res). template operator-= <THREAD_BLOCK_DIM>(rhs));
    }
     
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    device_matrix<T> operator-(const device_matrix<T>& lhs, const device_matrix<T>& rhs) {
      std::cout <<"operator- lvalue\n";
      static_assert(THREAD_BLOCK_DIM <= 1024);
      auto res{lhs};
      return std::forward<device_matrix<T>>((res). template operator-= <THREAD_BLOCK_DIM>(rhs));
    }
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    device_matrix<T> operator*(device_matrix<T>&& res, const T rhs) {
      std::cout <<"scalar operator* rvalue\n";
      static_assert(THREAD_BLOCK_DIM <= 1024);
      return std::forward<device_matrix<T>>((res). template operator*= <THREAD_BLOCK_DIM>(rhs));
    }
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    device_matrix<T> operator*(const device_matrix<T>& lhs, const T rhs) {
      std::cout <<"scalar operator* lvalue\n";
      static_assert(THREAD_BLOCK_DIM <= 1024);
      auto res{lhs};
      return std::forward<device_matrix<T>>((res). template operator*= <THREAD_BLOCK_DIM>(rhs));
    }
   
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    device_matrix<T> operator*(const T lhs, const device_matrix<T>& rhs) {
      std::cout <<"scalar operator*\n";
      static_assert(THREAD_BLOCK_DIM <= 1024);
      auto res{rhs};
      return std::forward<device_matrix<T>>((res). template operator*= <THREAD_BLOCK_DIM>(lhs));
    }

    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    device_matrix<T> operator*(device_matrix<T>&& res, const device_matrix<T>& rhs) {
      std::cout <<"operator* rvalue\n";
      static_assert(THREAD_BLOCK_DIM <= 1024);
      return std::forward<device_matrix<T>>((res). template operator*= <THREAD_BLOCK_DIM>(rhs));
    }
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    device_matrix<T> operator*(const device_matrix<T>& lhs, const device_matrix<T>& rhs) {
      std::cout <<"operator* lvalue\n";
      static_assert(THREAD_BLOCK_DIM <= 1024);
      auto res{lhs};
      return std::forward<device_matrix<T>>((res). template operator*= <THREAD_BLOCK_DIM>(rhs));
    }
      
    template<std::floating_point T, typename size_type>
    __global__ void print_matrix_kernel (const T* arr, size_type rows, size_type cols) {
      for (int r= 0; r < rows; ++r) {
        printf ("[");
        for (int c= 0; c < cols; ++c)
            printf ("%f ", arr[r * cols +  c]);
        printf("]\n");
      }
      printf("\n");
    }
    
    template<std::floating_point T>
    void print_matrix(const device_matrix<T>& mat) {
      print_matrix_kernel<<<1,1>>> (mat.data(), mat.num_rows(), mat.num_cols());
      cudaDeviceSynchronize();
    }
}




