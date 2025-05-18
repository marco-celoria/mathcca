
#include <cstddef>  // std::size_t
#include <concepts> // std::floating_point 
#include <iostream> // std::cout

// StdPar Omp Thrust Cuda
#include <mathcca/execution_policy.h>

#ifdef __CUDACC__
 #include <cuda_runtime.h>
 #include <mathcca/device_helper.h>
 #ifdef _THRUST
  #include <thrust/sequence.h>
  #include <thrust/execution_policy.h>
 #endif
#endif

#ifdef _STDPAR
 #include <execution>
 #include <ranges>
#endif

#ifdef _OPENMP
 #include <omp.h>
#endif

namespace mathcca {
    
  namespace detail {
     
#ifdef _STDPAR
     
    template<std::floating_point T>
    void fill_iota(StdPar, T* first, T* last, const T v) {
      std::cout << "DEBUG STDPAR\n";
      using value_type= T;
      const auto size {static_cast<std::size_t>(last - first)};
      std::ranges::iota_view indices(static_cast<unsigned int>(0),static_cast<unsigned int>(size));
      std::for_each(std::execution::par_unseq,indices.begin(),indices.end(),[&](auto i) { first[i]= v + static_cast<value_type>(i); });
    }
         
#endif
    
    template<std::floating_point T>
    void fill_iota(Omp, T* first, T* last, const T v) {
      std::cout << "DEBUG OMP\n";
      using value_type= T;
      const auto size {static_cast<std::size_t>(last - first)};
      #pragma omp parallel for default(shared)
      for (std::size_t i= 0; i < size; ++i) {
        first[i]= v + static_cast<value_type>(i);
      }
    }
    
#ifdef __CUDACC__
    
#ifdef _THRUST
    
    template<std::floating_point T>
    void fill_iota(Thrust, T* first, T* last, const T v) {
      std::cout << "DEBUG THRUST\n";
      thrust::sequence(thrust::device, first, last, v);
    }
    
#endif
    
    template<std::floating_point T>
    __global__ void fill_iota_kernel(T* __restrict start, const std::size_t size, const T v) {
      const auto tid{static_cast<std::size_t>(threadIdx.x + blockIdx.x * blockDim.x)};
      if(tid < size) {
        start[tid] = v + tid;
      }
    }
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    void fill_iota(Cuda, T* first, T* last, const T v, cudaStream_t stream) {
       std::cout << "DEBUG CUDA\n";
      static_assert(THREAD_BLOCK_DIM <= 1024);
      using value_type= T;
      const auto size {static_cast<std::size_t>(last - first)};
      constexpr unsigned int threads{THREAD_BLOCK_DIM};
      const auto blocks{static_cast<unsigned int>((size + static_cast<std::size_t>(threads) - 1) / (static_cast<std::size_t>(threads)))};
      constexpr dim3 dimBlock(threads, 1, 1);
      dim3 dimGrid(blocks, 1, 1);
      fill_iota_kernel<value_type><<<dimGrid, dimBlock, 0, stream>>>(first, size, v);
    }
    
#endif
    
  }  
    
}    


