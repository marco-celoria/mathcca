#ifndef HOST_MATMUL_H_
#define HOST_MATMUL_H_
#pragma once

#include <concepts>
#include <mathcca/host_matrix.h>

namespace mathcca {
      
    enum class HostMM {
      Base,
      Tiled
#ifdef _MKL
      , Mkl
#endif
    };
     
    template<std::floating_point T>
    constexpr void mm_parallel_Base(const host_matrix<T>& A, const host_matrix<T>& B, host_matrix<T>& C);
    
    template<std::floating_point T, unsigned int LINEAR_TILE_DIM= 32>
    constexpr void mm_parallel_Tiled(const host_matrix<T>& A, const host_matrix<T>& B, host_matrix<T>& C);
    
#ifdef _MKL
    template<std::floating_point T>
    constexpr void mm_parallel_Mkl(const host_matrix<T>& A, const host_matrix<T>& B, host_matrix<T>& C);
#endif
    
    template<std::floating_point T, HostMM O, unsigned int LINEAR_TILE_DIM= 32>
    void matmul(const host_matrix<T>& A, const host_matrix<T>& B, host_matrix<T>& C) {
      static_assert(O == HostMM::Base || O == HostMM::Tiled
#ifdef _MKL
                    || O == HostMM::Mkl
#endif
                   );
      if constexpr(O == HostMM::Tiled) {
        mm_parallel_Tiled<T, LINEAR_TILE_DIM>(A, B, C);
      }
#ifdef _MKL
      else if constexpr(O == HostMM::Mkl) {
        mm_parallel_Mkl<T>(A, B, C);
      }
#endif
      else {
        mm_parallel_Base<T>(A, B, C);
      }
    }
    
    template<std::floating_point T, HostMM O, unsigned int LINEAR_TILE_DIM= 32>
    constexpr decltype(auto) matmul(const host_matrix<T>& A, const host_matrix<T>& B) {
      host_matrix<T> C{A.num_rows(), B.num_cols(), static_cast<T>(0)};
      matmul<T, O, LINEAR_TILE_DIM>(A, B, C);
      return C;
    }
    
}

#include <mathcca/detail/host_matmul.inl>

#endif

