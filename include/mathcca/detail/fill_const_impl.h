#ifndef FILL_CONST_IMPL_H_
#define FILL_CONST_IMPL_H_
#pragma once


#include <concepts>
#include <mathcca/common_algorithm.h>


namespace mathcca {
  namespace detail {
#ifdef _PARALG
    template<std::floating_point T>
    void fill_const(StdPar, T* first, T* last, const T v); 
#endif
    template<std::floating_point T>
    void fill_const(Omp, T* first, T* last, const T v) ;
  }
}


#ifdef __CUDACC__

namespace mathcca {
  namespace detail {	
    
    template<std::floating_point T>
    void fill_const(Thrust, T* first, T* last, const T v); 
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM=128>
    void fill_const(Cuda, T* first, T* last, const T v, cudaStream_t stream=0);
  }
}

#endif
#include<mathcca/detail/fill_const_impl.inl>
#endif
