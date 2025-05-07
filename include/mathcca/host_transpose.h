#ifndef HOST_TRANSPOSE_H_
#define HOST_TRANSPOSE_H_
#pragma once

#include <concepts>
#include <mathcca/host_matrix.h>

namespace mathcca {
     
  namespace matricca {
    
    template<std::floating_point T>
    constexpr void transpose_parallel_Base(const host_matrix<T>& A, host_matrix<T>& B);
     
     
    template<std::floating_point T, unsigned int LINEAR_TILE_DIM>
    constexpr void transpose_parallel_Tiled(const host_matrix<T>& A, host_matrix<T>& B);
    
    
#ifdef _MKL
    template<std::floating_point T>
    constexpr void transpose_parallel_Mkl(const host_matrix<T>& A, host_matrix<T>& B);
#endif
    
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
        transpose_parallel_Tiled<T, LINEAR_TILE_DIM>(A, B);
      }
      #ifdef _MKL
      else if constexpr(O == HostT::Mkl) {
        transpose_parallel_Mkl<T>(A, B);
      }
      #endif
      else {
        transpose_parallel_Base<T>(A, B);
      }
    }
    
    template<std::floating_point T, HostT O, unsigned int LINEAR_TILE_DIM= 32>
    constexpr decltype(auto) transpose(const host_matrix<T>& A) {
      host_matrix<T> B{A.num_cols(), A.num_rows(), static_cast<T>(0)};
      transpose<T, O, LINEAR_TILE_DIM>(A, B);
      return B;
    }
    
  }
}

#include <mathcca/detail/host_transpose.inl>

#endif

