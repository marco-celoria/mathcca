#ifndef DEVICE_MATMUL_H_
#define DEVICE_MATMUL_H_
#pragma once

#include <concepts>
#include <cuda_runtime.h>
#include <mathcca/device_matrix.h>

namespace mathcca {

  namespace matricca {
  
    enum class DevMM {
      Base,
      Tiled
#ifdef _CUBLAS
      , Cublas
#endif
    };
     
    template <std::floating_point T, unsigned int LINEAR_THREAD_BLOCK_DIM= 16>
    void mm_device_Base(const device_matrix<T>& A, const device_matrix<T>& B, device_matrix<T>& C, cudaStream_t stream= 0);

    template <std::floating_point T, unsigned int LINEAR_THREAD_BLOCK_DIM= 16>
    void mm_device_Tiled(const device_matrix<T>& A, const device_matrix<T>& B, device_matrix<T>& C, cudaStream_t stream= 0);
    
    template <std::floating_point T>
    void  mm_device_Cublas(const device_matrix<T>& A, const device_matrix<T>& B, device_matrix<T>& C);
    
    template<std::floating_point T, DevMM O, unsigned int LINEAR_THREAD_BLOCK_DIM= 16 >
    void matmul(const device_matrix<T>& A, const device_matrix<T>& B, device_matrix<T>& C, cudaStream_t stream= 0) {
      static_assert(O == DevMM::Base || O == DevMM::Tiled
#ifdef _CUBLAS
                    || O == DevMM::Cublas
#endif
                   );
      if constexpr(O == DevMM::Tiled) {
        mm_device_Tiled<T, LINEAR_THREAD_BLOCK_DIM>(A, B, C, stream);
      }
#ifdef _CUBLAS
      else if constexpr(O == DevMM::Cublas) {
        mm_device_Cublas<T>(A, B, C);
      }
#endif
      else {
        mm_device_Base<T, LINEAR_THREAD_BLOCK_DIM>(A, B, C, stream);
      }
    }
    
    template<std::floating_point T, DevMM O, unsigned int LINEAR_THREAD_BLOCK_DIM= 16 >
    constexpr decltype(auto) matmul(const device_matrix<T>& A, const device_matrix<T>& B, cudaStream_t stream= 0) {
      device_matrix<T> C{A.num_rows(), B.num_cols(), static_cast<T>(0)};
      matmul<T, O, LINEAR_THREAD_BLOCK_DIM>(A, B, C, stream);
      return C;
    }

  }

}

#include <mathcca/detail/device_matmul.inl>

#endif

