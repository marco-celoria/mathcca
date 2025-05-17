#ifndef FILL_RAND_IMPL_H_
#define FILL_RAND_IMPL_H_
#pragma once

#include <concepts>
#include <mathcca/common_algorithm.h>


namespace mathcca {
  namespace detail {

#ifdef _PARALG
    template<std::floating_point T>
    void fill_rand(StdPar, T* first, T* last); 
#endif

    template<std::floating_point T>
    void fill_rand(Omp, T* first, T* last); 
  }
}

#ifdef __CUDACC__
namespace mathcca {
  namespace detail {
   
    template<std::floating_point T>
    void fill_rand(Thrust, T* first, T* last); 

    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    void fill_rand(Cuda, T* first, T* last, cudaStream_t stream);     
  }
}
#endif

#include <mathcca/detail/fill_rand_impl.inl>

#endif


