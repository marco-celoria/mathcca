#ifndef FILL_CONST_IMPL_H_
#define FILL_CONST_IMPL_H_
#pragma once

#include <concepts> // std::floating_point

// StdPar Omp Thrust CudaD
#include <mathcca/execution_policy.h>

#ifdef __CUDACC__
#include <cuda_runtime.h> // cudaStream_t
#endif

namespace mathcca {
    
  namespace detail {
    
#ifdef _STDPAR
    template<std::floating_point T>
    void fill_const(StdPar, T* first, T* last, const T v); 
#endif
    
    template<std::floating_point T>
    void fill_const(Omp, T* first, T* last, const T v);
    
#ifdef __CUDACC__
    
#ifdef _THRUST 
    template<std::floating_point T>
    void fill_const(Thrust, T* first, T* last, const T v); 
#endif
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM=128>
    void fill_const(CudaD, T* first, T* last, const T v, cudaStream_t stream=0);

#endif
    
  }  
    
}    

#include<mathcca/detail/fill_const_impl.inl>

#endif


