#ifndef TRANSPOSE_H_
#define TRANSPOSE_H_
#pragma once

#include <concepts> // std::floating_point

#include <mathcca/host_matrix.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <mathcca/device_matrix.h>
#endif

#include <mathcca/detail/transpose_impl.h>

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
     
  template<std::floating_point T, typename Implementation, unsigned int LINEAR_TILE_DIM= 32 >
  void transpose(const host_matrix<T>& A, host_matrix<T>& B, Implementation) {
    static_assert(std::is_same_v<Implementation, Trans::Base> || std::is_same_v<Implementation, Trans::Tiled>
#ifdef _MKL       
                  || std::is_same_v<Implementation, Trans::Mkl>
#endif
                 );
    if constexpr(std::is_same_v< Implementation, Trans::Base>) {
      detail::transpose_parallel_Base<T>(A, B);
    }
    else if constexpr(std::is_same_v< Implementation, Trans::Tiled>) {
      detail::transpose_parallel_Tiled<T, LINEAR_TILE_DIM>(A, B);
    }
#ifdef _MKL
    else if constexpr(std::is_same_v< Implementation, Trans::Mkl>) {
      detail::transpose_parallel_Mkl<T>(A, B);
    }
#endif
  }
      
  template<std::floating_point T, typename Implementation, unsigned int LINEAR_TILE_DIM= 32>
  constexpr decltype(auto) transpose(const host_matrix<T>& A, Implementation) {
    host_matrix<T> B{A.num_cols(), A.num_rows(), static_cast<T>(0)};
    transpose<T, Implementation, LINEAR_TILE_DIM>(A, B, Implementation());
    return B;
  }
     
#ifdef __CUDACC__
      
        
  template<std::floating_point T, typename Implementation, unsigned int LINEAR_THREAD_BLOCK_DIM= 16 >
  void transpose(const device_matrix<T>& A, device_matrix<T>& B, Implementation, cudaStream_t stream= 0) {
    static_assert(std::is_same_v<Implementation, Trans::Base> || std::is_same_v<Implementation, Trans::Tiled>
#ifdef _CUBLAS       
                  || std::is_same_v<Implementation, Trans::Cublas>
#endif
                 );
    if constexpr (std::is_same_v< Implementation, Trans::Base>) {
      detail::transpose_device_Base<T, LINEAR_THREAD_BLOCK_DIM>(A, B, stream);
    }
    else if constexpr(std::is_same_v< Implementation, Trans::Tiled>) {
      detail::transpose_device_Tiled<T, LINEAR_THREAD_BLOCK_DIM>(A, B, stream);
    }
#ifdef _CUBLAS
    else if constexpr(std::is_same_v< Implementation, Trans::Cublas>) {
      detail::transpose_device_Cublas<T>(A, B);
    }
#endif
  }
       
  template<std::floating_point T, typename Implementation, unsigned int LINEAR_THREAD_BLOCK_DIM= 16 >
  constexpr decltype(auto) transpose(const device_matrix<T>& A, Implementation, cudaStream_t stream= 0) {
    device_matrix<T> B{A.num_cols(), A.num_rows(), static_cast<T>(0)};
    transpose<T, Implementation, LINEAR_THREAD_BLOCK_DIM>(A, B, Implementation(), stream);
    return B;
  }

#endif

}

#endif


