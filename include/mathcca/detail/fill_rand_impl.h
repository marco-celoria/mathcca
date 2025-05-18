#ifndef FILL_RAND_IMPL_H_
#define FILL_RAND_IMPL_H_
#pragma once

#include <concepts> //std::floating_point

// StdPar Omp Thrust Cuda
#include <mathcca/execution_policy.h>

#ifdef __CUDACC__
#include <cuda_runtime.h> // cudaStream_t
#endif

namespace mathcca {
    
  namespace detail {
    
#ifdef _STDPAR
    template<std::floating_point T>
    void fill_rand(StdPar, T* first, T* last); 
#endif
    
    template<std::floating_point T>
    void fill_rand(Omp, T* first, T* last); 
    
#ifdef __CUDACC__
    
#ifdef _THRUST
    template<std::floating_point T>
    void fill_rand(Thrust, T* first, T* last); 
#endif
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    void fill_rand(Cuda, T* first, T* last, cudaStream_t stream);     
    
#endif
    
  }  
  
}    

#include <mathcca/detail/fill_rand_impl.inl>

#endif


