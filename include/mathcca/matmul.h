#ifndef MATMUL_H_
#define MATMUL_H_
#pragma once

#include <concepts> // std::floating_point

#include <mathcca/host_matrix.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <mathcca/device_matrix.h>
#endif

#include <mathcca/detail/matmul_impl.h>

namespace mathcca {
    
  enum class HostMM {
    Base
    , Tiled
#ifdef _MKL
    , Mkl
#endif
  };
      
  template<std::floating_point T, HostMM O, unsigned int LINEAR_TILE_DIM= 32>
  void matmul(const host_matrix<T>& A, const host_matrix<T>& B, host_matrix<T>& C) {
    static_assert(O == HostMM::Base || O == HostMM::Tiled
#ifdef _MKL       
                  || O == HostMM::Mkl
#endif            
                 );
    if constexpr(O == HostMM::Tiled) {
      detail::mm_parallel_Tiled<T, LINEAR_TILE_DIM>(A, B, C);
    }
#ifdef _MKL
    else if constexpr(O == HostMM::Mkl) {
      detail::mm_parallel_Mkl<T>(A, B, C);
    }
#endif
    else {
      detail::mm_parallel_Base<T>(A, B, C);
    }
  }
      
  template<std::floating_point T, HostMM O, unsigned int LINEAR_TILE_DIM= 32>
  constexpr decltype(auto) matmul(const host_matrix<T>& A, const host_matrix<T>& B) {
    host_matrix<T> C{A.num_rows(), B.num_cols(), static_cast<T>(0)};
    matmul<T, O, LINEAR_TILE_DIM>(A, B, C);
    return C;
  }

#ifdef __CUDACC__

  enum class DevMM {
    Base
    , Tiled
#ifdef _CUBLAS
    , Cublas
#endif
  };

  template<std::floating_point T, DevMM O, unsigned int LINEAR_THREAD_BLOCK_DIM= 16 >
  void matmul(const device_matrix<T>& A, const device_matrix<T>& B, device_matrix<T>& C, cudaStream_t stream= 0) {
    static_assert(O == DevMM::Base || O == DevMM::Tiled
#ifdef _CUBLAS
                  || O == DevMM::Cublas
#endif
                 );
    if constexpr(O == DevMM::Tiled) {
      detail::mm_device_Tiled<T, LINEAR_THREAD_BLOCK_DIM>(A, B, C, stream);
    }
#ifdef _CUBLAS
    else if constexpr(O == DevMM::Cublas) {
      detail::mm_device_Cublas<T>(A, B, C);
    }
#endif
    else {
      detail::mm_device_Base<T, LINEAR_THREAD_BLOCK_DIM>(A, B, C, stream);
    }
  }
      
  template<std::floating_point T, DevMM O, unsigned int LINEAR_THREAD_BLOCK_DIM= 16 >
  constexpr decltype(auto) matmul(const device_matrix<T>& A, const device_matrix<T>& B, cudaStream_t stream= 0) {
    device_matrix<T> C{A.num_rows(), B.num_cols(), static_cast<T>(0)};
    matmul<T, O, LINEAR_THREAD_BLOCK_DIM>(A, B, C, stream);
    return C;
  }

#endif

}

#endif


