
#include <cstddef>  // std::size_t
#include <concepts> // std::floating_point
#include <iostream> // std::cout

// StdPar Omp Thrust CudaD
#include <mathcca/execution_policy.h>

#ifdef _STDPAR
 #include <execution>
 #include <ranges>
#endif

#ifdef __CUDACC__
 #include <cuda_runtime.h>
 #include <mathcca/device_helper.h>
 #ifdef _THRUST
  #include <thrust/fill.h>
  #include <thrust/execution_policy.h>
 #endif
#endif

#ifdef _OPENMP
 #include <omp.h>
#endif
    
namespace mathcca {
     
  namespace detail {
    
#ifdef _STDPAR
      
    template<std::floating_point T>
    void fill_const(StdPar, T* first, T* last, const T v) {
      std::cout << "DEBUG STDPAR\n";
      std::fill(std::execution::par_unseq, first, last, v);
    }
    
#endif
    
    template<std::floating_point T>
    void fill_const(Omp, T* first, T* last, const T v) {
      std::cout << "DEBUG OMP\n";
      const std::size_t size= static_cast<std::size_t>(last - first);
      #pragma omp parallel for default(shared)
      for (std::size_t i= 0; i < size; ++i) {
        first[i]= v;
      }
    }
    
#ifdef __CUDACC__
    
#ifdef _THRUST
    
    template<std::floating_point T>
    void fill_const(Thrust, T* first, T* last, const T v) {
      std::cout << "DEBUG THRUST\n";
      thrust::fill(thrust::device, first, last, v);
    }
    
#endif
    
    template<std::floating_point T>
    __global__ void fill_const_kernel(T* start, const std::size_t size, const T v) {
      const auto tid{static_cast<std::size_t>(threadIdx.x + blockIdx.x * blockDim.x)};
      if(tid < size) {
        start[tid] = v;
      }
    }
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    void fill_const(CudaD, T* first, T* last, const T v, cudaStream_t stream) {
      std::cout << "DEBUG CUDA\n";
      static_assert(THREAD_BLOCK_DIM <= 1024);
      using value_type= T;
      const auto size {static_cast<std::size_t>(last - first)};
      constexpr unsigned int threads{THREAD_BLOCK_DIM};
      const auto blocks{static_cast<unsigned int>((size + static_cast<std::size_t>(threads) - 1) / (static_cast<std::size_t>(threads)))};
      constexpr dim3 dimBlock(threads, 1, 1);
      dim3 dimGrid(blocks, 1, 1);
      fill_const_kernel<value_type><<<dimGrid, dimBlock, 0, stream>>>(first, size, v);
    }

#endif
    
  } 
     
}   


