#ifndef DEVICE_TRANSPOSE_H_
#define DEVICE_TRANSPOSE_H_
#pragma once

#include <concepts>
#include <cuda_runtime.h>
#include <mathcca/device_matrix.h>

namespace mathcca {

  namespace matricca {
    
    enum class DevT {
      Base,
      Tiled
#ifdef _CUBLAS
      , Cublas
#endif
    };
    
    template <std::floating_point T, unsigned int LINEAR_THREAD_BLOCK_DIM= 16>
    void transpose_device_Base(const device_matrix<T>& A, device_matrix<T>& B, cudaStream_t stream= 0);
    
    template <std::floating_point T, unsigned int LINEAR_THREAD_BLOCK_DIM= 16>
    void transpose_device_Tiled(const device_matrix<T>& A, device_matrix<T>& B, cudaStream_t stream= 0);
    
#ifdef _CUBLAS
    template <std::floating_point T>
    void transpose_device_Cublas(const device_matrix<T>& A, device_matrix<T>& B);
#endif
    
    template<std::floating_point T, DevT O, unsigned int LINEAR_THREAD_BLOCK_DIM= 16 >
    void transpose(const device_matrix<T>& A, device_matrix<T>& B, cudaStream_t stream= 0) {
      static_assert(O == DevT::Base || O == DevT::Tiled
#ifdef _CUBLAS
                    || O == DevT::Cublas
#endif
                   );
      if constexpr(O == DevT::Tiled) {
        transpose_device_Tiled<T, LINEAR_THREAD_BLOCK_DIM>(A, B, stream);
      }
#ifdef _CUBLAS
      else if constexpr(O == DevT::Cublas) {
        transpose_device_Cublas<T>(A, B);
      }
#endif
      else {
        transpose_device_Base<T, LINEAR_THREAD_BLOCK_DIM>(A, B, stream);
      }
    }
    
    template<std::floating_point T, DevT O, unsigned int LINEAR_THREAD_BLOCK_DIM= 16 >
    constexpr decltype(auto) transpose(const device_matrix<T>& A, cudaStream_t stream= 0) {
      device_matrix<T> B{A.num_cols(), A.num_rows(), static_cast<T>(0)};
      transpose<T, O, LINEAR_THREAD_BLOCK_DIM>(A, B, stream);
      return B;
    }

  }

}

#include <mathcca/detail/device_transpose.inl>

#endif


