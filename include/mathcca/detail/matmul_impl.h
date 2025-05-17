#ifndef MATMUL_IMPL_H_
#define MATMUL_IMPL_H_
#pragma once

#include <concepts>
#include <mathcca/host_matrix.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <mathcca/device_matrix.h>
#endif

namespace mathcca {
namespace detail {

        template<std::floating_point T>
    constexpr void mm_parallel_Base(const host_matrix<T>& A, const host_matrix<T>& B, host_matrix<T>& C) ;

          template<std::floating_point T, unsigned int LINEAR_TILE_DIM>
  constexpr void mm_parallel_Tiled(const host_matrix<T>& A, const host_matrix<T>& B, host_matrix<T>& C) ;

            #ifdef _MKL
  template<std::floating_point T>
  constexpr void mm_parallel_Mkl(const host_matrix<T>& A, const host_matrix<T>& B, host_matrix<T>& C) ;
#endif 
}
}

#ifdef __CUDACC__ 
namespace mathcca {
 namespace detail {

      
    template <std::floating_point T, unsigned int LINEAR_THREAD_BLOCK_DIM>
    void mm_device_Base(const device_matrix<T>& A, const device_matrix<T>& B, device_matrix<T>& C, cudaStream_t stream) ;     
    template <std::floating_point T, unsigned int LINEAR_THREAD_BLOCK_DIM>
    void mm_device_Tiled(const device_matrix<T>& A, const device_matrix<T>& B, device_matrix<T>& C, cudaStream_t stream);     
#ifdef _CUBLAS
    template <std::floating_point T>
    void  mm_device_Cublas(const device_matrix<T>& A, const device_matrix<T>& B, device_matrix<T>& C) ; 

#endif
 }   
}
#endif

#include <mathcca/detail/matmul_impl.inl>

#endif
