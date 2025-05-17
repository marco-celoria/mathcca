#ifndef TRANSPOSE_H_
#define TRANSPOSE_H_
#pragma once

#include <concepts>
#include <mathcca/host_matrix.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <mathcca/device_matrix.h>
#endif

#include <mathcca/detail/transpose_impl.h>

namespace mathcca {

    enum class HostT {
      Base,
      Tiled
#ifdef _MKL
      , Mkl
#endif
    };

    template<std::floating_point T, HostT O, unsigned int LINEAR_TILE_DIM= 32 >
    void transpose(const host_matrix<T>& A, host_matrix<T>& B) {
      static_assert(O == HostT::Base || O == HostT::Tiled
#ifdef _MKL
                    || O == HostT::Mkl
#endif
                   );
      if constexpr(O == HostT::Tiled) {
	      detail::transpose_parallel_Tiled<T, LINEAR_TILE_DIM>(A, B);
      }
      #ifdef _MKL
      else if constexpr(O == HostT::Mkl) {
	      detail::transpose_parallel_Mkl<T>(A, B);
      }
      #endif
      else {
	      detail::transpose_parallel_Base<T>(A, B);
      }
    }

    template<std::floating_point T, HostT O, unsigned int LINEAR_TILE_DIM= 32>
    constexpr decltype(auto) transpose(const host_matrix<T>& A) {
      host_matrix<T> B{A.num_cols(), A.num_rows(), static_cast<T>(0)};
      transpose<T, O, LINEAR_TILE_DIM>(A, B);
      return B;
    }

}

#ifdef __CUDACC__
namespace mathcca {

    enum class DevT {
      Base,
      Tiled
#ifdef _CUBLAS
      , Cublas
#endif
    };

    template<std::floating_point T, DevT O, unsigned int LINEAR_THREAD_BLOCK_DIM= 16 >
    void transpose(const device_matrix<T>& A, device_matrix<T>& B, cudaStream_t stream= 0) {
      static_assert(O == DevT::Base || O == DevT::Tiled
#ifdef _CUBLAS
                    || O == DevT::Cublas
#endif
                   );
      if constexpr(O == DevT::Tiled) {
	      detail::transpose_device_Tiled<T, LINEAR_THREAD_BLOCK_DIM>(A, B, stream);
      }
#ifdef _CUBLAS
      else if constexpr(O == DevT::Cublas) {
	      detail::transpose_device_Cublas<T>(A, B);
      }
#endif
      else {
	      detail::transpose_device_Base<T, LINEAR_THREAD_BLOCK_DIM>(A, B, stream);
      }
    }

    template<std::floating_point T, DevT O, unsigned int LINEAR_THREAD_BLOCK_DIM= 16 >
    constexpr decltype(auto) transpose(const device_matrix<T>& A, cudaStream_t stream= 0) {
      device_matrix<T> B{A.num_cols(), A.num_rows(), static_cast<T>(0)};
      transpose<T, O, LINEAR_THREAD_BLOCK_DIM>(A, B, stream);
      return B;
    }

}
#endif

//#ifdef __CUDACC__
//#include <mathcca/transpose.inl>
//#endif

//#include <mathcca/transpose.inl>

#endif

