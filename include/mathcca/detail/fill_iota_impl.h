#ifndef FILL_IOTA_IMPL_H_
#define FILL_IOTA_IMPL_H_
#pragma once


#include <concepts>
#include <mathcca/common_algorithm.h>

namespace mathcca {
namespace detail {
#ifdef _PARALG
    template<std::floating_point T>
    void fill_iota(StdPar, T* first, T* last, const T v);
#endif
    template<std::floating_point T>
    void fill_iota(Omp, T* first, T* last, const T v); }
}


#ifdef __CUDACC__
namespace mathcca {
namespace detail {
    template<std::floating_point T>
    void fill_iota(Thrust, T* first, T* last, const T v); 

    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    void fill_iota(Cuda, T* first, T* last, const T v, cudaStream_t stream) ;
}
}

#include <mathcca/detail/fill_iota_impl.inl>

#endif


