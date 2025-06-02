/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include <cstddef>  // std::size_t
#include <concepts> // std::floating_point
#include <iostream> // std::cout

// StdPar Omp Thrust Cuda CudaHtoDcpy CudaDtoHcpy
#include <mathcca/execution_policy.h>

#ifdef _STDPAR
 #include <execution>
 #include <ranges>
 #include <algorithm>
#endif

#ifdef __CUDACC__
 #include <cuda_runtime.h> // cudaStream_t
 #include <mathcca/device_helper.h> // checkCudaErrors
 #ifdef _THRUST
  #include <thrust/copy.h>
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
    void copy(StdPar, const T* s_first, const T* s_last, T* d_first) {
      std::cout << "DEBUG COPY STDPAR\n";
      std::copy(std::execution::par_unseq, s_first, s_last, d_first);
    }
    
#endif
    
    template<std::floating_point T>
    void copy(Omp, const T* s_first, const T* s_last, T* d_first) {
      std::cout << "DEBUG COPY OMP\n";
      const auto size {static_cast<std::size_t>(s_last - s_first)};
      #pragma omp parallel for default(shared)
      for (std::size_t i= 0; i < size; ++i) {
        d_first[i]= s_first[i];
      }
    }
    
#ifdef __CUDACC__
     
#ifdef _THRUST
    
    template<std::floating_point T>
    void copy(Thrust, const T* s_first, const T* s_last, T* d_first) {
      std::cout << "DEBUG COPY THRUST\n";
      thrust::copy(thrust::device, s_first, s_last, d_first);
    }
    
#endif
    
    template<std::floating_point T>
    void copy(Cuda, const T* s_first, const T* s_last, T* d_first, cudaStream_t stream) {
      std::cout << "DEBUG COPY CUDADTODCPY\n";
      const auto size{static_cast<std::size_t>(s_last - s_first)};
      const std::size_t nbytes{size * sizeof(T)};
      checkCudaErrors(cudaMemcpy(d_first, s_first, nbytes, cudaMemcpyDeviceToDevice));
    }
    
    template<std::floating_point T>
    void copy(CudaDtoHcpy, const T* s_first, const T* s_last, T* d_first, cudaStream_t stream) {
      std::cout << "DEBUG COPY CUDADTOHCPY\n";
      const auto size{static_cast<std::size_t>(s_last - s_first)};
      const auto bytes{size * sizeof(T)};
#ifdef _PINNED 
      std::cout << "(DEBUG PINNED)\n";
      checkCudaErrors(cudaMemcpyAsync(d_first, s_first, bytes, cudaMemcpyDeviceToHost, stream));
#else
      checkCudaErrors(cudaMemcpy(d_first, s_first, bytes, cudaMemcpyDeviceToHost));
#endif
    }
    
    template<std::floating_point T>
    void copy(CudaHtoDcpy, const T* s_first, const T* s_last, T* d_first, cudaStream_t stream) {
      std::cout << "DEBUG COPY CUDAHTODCPY\n";
      const auto size{static_cast<std::size_t>(s_last - s_first)};
      const auto bytes{size * sizeof(T)};
#ifdef _PINNED 
      std::cout << "(DEBUG PINNED)\n";
      checkCudaErrors(cudaMemcpyAsync(d_first, s_first, bytes, cudaMemcpyHostToDevice, stream));
#else   
      checkCudaErrors(cudaMemcpy(d_first, s_first, bytes, cudaMemcpyHostToDevice));
#endif
    }

#endif

  }

}



