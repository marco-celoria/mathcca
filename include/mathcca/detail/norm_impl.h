#ifndef NORM_IMPL_H_
#define NORM_IMPL_H_
#pragma once

#include <concepts>
#include <mathcca/host_matrix.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <mathcca/device_matrix.h>
#endif


namespace mathcca {
 namespace detail {

#ifdef _MKL
  template<std::floating_point T>
  constexpr decltype(auto) frobenius_norm_Mkl(const host_matrix<T>& x) ;
#endif

    template<std::floating_point T>
  constexpr decltype(auto) frobenius_norm_Base (const host_matrix<T>& A);
}}


#ifdef __CUDACC__
namespace mathcca {
 namespace detail { 
#ifdef _CUBLAS
    template<std::floating_point T>
    constexpr decltype(auto) frobenius_norm_Cublas(const device_matrix<T>& x);
#endif
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    constexpr decltype(auto) frobenius_norm_Base(const device_matrix<T>& x, cudaStream_t stream);
}
}
#endif

#include <mathcca/detail/norm_impl.inl>
#endif
