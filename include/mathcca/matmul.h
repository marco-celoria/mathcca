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
      
  template<std::floating_point T, typename Implementation, unsigned int LINEAR_TILE_DIM= 32>
  void matmul(const host_matrix<T>& A, const host_matrix<T>& B, host_matrix<T>& C, Implementation) {
    static_assert(std::is_same_v<Implementation, MM::Base> || std::is_same_v<Implementation, MM::Tiled>
#ifdef _MKL       
                  || std::is_same_v<Implementation, MM::Mkl>
#endif            
                 );
    if constexpr(std::is_same_v< Implementation, MM::Base>) {
      detail::matmul<T>(MM::Base(), A, B, C);
    }
    else if constexpr(std::is_same_v< Implementation, MM::Tiled>) {
      detail::matmul<T, LINEAR_TILE_DIM>(MM::Tiled(), A, B, C);
    }
#ifdef _MKL
    else if constexpr(std::is_same_v< Implementation, MM::Mkl>) {
      detail::matmul<T>(MM::Mkl(), A, B, C);
    }
#endif
  }
      
  template<std::floating_point T, typename Implementation, unsigned int LINEAR_TILE_DIM= 32>
  constexpr decltype(auto) matmul(const host_matrix<T>& A, const host_matrix<T>& B, Implementation) {
    host_matrix<T> C{A.num_rows(), B.num_cols(), static_cast<T>(0)};
    matmul<T, Implementation, LINEAR_TILE_DIM>(A, B, C, Implementation());
    return C;
  }

#ifdef __CUDACC__

  template<std::floating_point T, typename Implementation, unsigned int LINEAR_THREAD_BLOCK_DIM= 16 >
  void matmul(const device_matrix<T>& A, const device_matrix<T>& B, device_matrix<T>& C, Implementation, cudaStream_t stream= 0) {
    static_assert(std::is_same_v<Implementation, MM::Base> || std::is_same_v<Implementation, MM::Tiled>
#ifdef _CUBLAS       
                  || std::is_same_v<Implementation, MM::Cublas>
#endif
                 );
    if constexpr(std::is_same_v< Implementation, MM::Base>) {
      detail::matmul<T, LINEAR_THREAD_BLOCK_DIM>(MM::Base(), A, B, C, stream);
    }
    else if constexpr(std::is_same_v< Implementation, MM::Tiled>) {
      detail::matmul<T, LINEAR_THREAD_BLOCK_DIM>(MM::Tiled(), A, B, C, stream);
    }
#ifdef _CUBLAS
    else if constexpr(std::is_same_v< Implementation, MM::Cublas>) {
      detail::matmul<T>(MM::Cublas(), A, B, C);
    }
#endif
  }
      
  template<std::floating_point T, typename Implementation, unsigned int LINEAR_THREAD_BLOCK_DIM= 16 >
  constexpr decltype(auto) matmul(const device_matrix<T>& A, const device_matrix<T>& B, Implementation, cudaStream_t stream= 0) {
    device_matrix<T> C{A.num_rows(), B.num_cols(), static_cast<T>(0)};
    matmul<T, Implementation, LINEAR_THREAD_BLOCK_DIM>(A, B, C, Implementation(), stream);
    return C;
  }

#endif

}

#endif


