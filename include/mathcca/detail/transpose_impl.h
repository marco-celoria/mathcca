#ifndef TRANSPOSE_IMPL_H_
#define TRANSPOSE_IMPL_H_
#pragma once

#include <concepts> // std::floating_point

#include <mathcca/host_matrix.h>

#ifdef __CUDACC__
#include <cuda_runtime.h> // cudaStream_t
#include <mathcca/device_matrix.h>
#endif

namespace mathcca {

  namespace Trans {
    class Base{};
    class Tiled{};
#ifdef _MKL
    class Mkl{};
#endif
#ifdef _CUBLAS
    class Cublas{};
#endif
  }

  namespace detail {
      
    template<std::floating_point T>
    constexpr void transpose(Trans::Base, const host_matrix<T>& A, host_matrix<T>& B);
      
    template<std::floating_point T, unsigned int LINEAR_TILE_DIM>
    constexpr void transpose(Trans::Tiled, const host_matrix<T>& A, host_matrix<T>& B);
      
#ifdef _MKL
      
    template<std::floating_point T>
    constexpr void transpose(Trans::Mkl, const host_matrix<T>& A, host_matrix<T>& B); 
      
#endif
      
#ifdef __CUDACC__
      
    template <std::floating_point T, unsigned int LINEAR_THREAD_BLOCK_DIM>
    void transpose(Trans::Base, const device_matrix<T>& A, device_matrix<T>& B, cudaStream_t stream);
      
    template <std::floating_point T, unsigned int LINEAR_THREAD_BLOCK_DIM>
    void transpose(Trans::Tiled, const device_matrix<T>& A, device_matrix<T>& B, cudaStream_t stream);     
      
#ifdef _CUBLAS
      
    template <std::floating_point T>
    void  transpose(Trans::Cublas, const device_matrix<T>& A, device_matrix<T>& B);
      
#endif
      
#endif
      
  }  
  
}

#include<mathcca/detail/transpose_impl.inl>

#endif


