#ifndef COPY_IMPL_H_
#define COPY_IMPL_H_
#pragma once

#include <concepts> // std::floating_point

// StdPar Omp Thrust CudaDtoDcpy CudaHtoHcpy CudaHtoDcpy CudaDtoHcpy
#include <mathcca/common_algorithm.h> 

#ifdef __CUDACC__
#include <cuda_runtime.h> // cudaStream_t
#endif

namespace mathcca {
    
  namespace detail {
    
#ifdef _STDPAR
    template<std::floating_point T>
    void copy(StdPar, const T* s_first, const T* s_last, T* d_first); 
#endif
    
    template<std::floating_point T>
    void copy(Omp, const T* s_first, const T* s_last, T* d_first); 
    
#ifdef __CUDACC__
    
#ifdef _THRUST      
    template<std::floating_point T>
    void copy(Thrust, const T* s_first, const T* s_last, T* d_first);
#endif
    
    template<std::floating_point T>
    void copy(CudaDtoDcpy, const T* s_first, const T* s_last, T* d_first, cudaStream_t stream=0);
    
    template<std::floating_point T>
    void copy(CudaHtoHcpy, const T* s_first, const T* s_last, T* d_first, cudaStream_t stream=0);
    
    template<std::floating_point T>
    void copy(CudaDtoHcpy, const T* s_first, const T* s_last, T* d_first, cudaStream_t stream=0); 
    
    template<std::floating_point T>
    void copy(CudaHtoDcpy, const T* s_first, const T* s_last, T* d_first, cudaStream_t stream=0);
    
#endif
    
  }  
    
}    
    

#include <mathcca/detail/copy_impl.inl>

#endif

