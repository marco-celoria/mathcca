/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef TRANSPOSE_H_
#define TRANSPOSE_H_
#pragma once

#include <mathcca/host_matrix.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <mathcca/device_matrix.h>
#endif

#include <mathcca/detail/transpose_impl.h>

namespace mathcca {
     
    template<typename Matrix>
    constexpr inline auto check_transposition_compatible_size(const Matrix& lhs, const Matrix& rhs) {
      if ((lhs.num_cols() == rhs.num_rows()) && (lhs.num_rows() == rhs.num_cols()))
        return true;
      return false;
    }


  template<typename Matrix, typename Implementation, unsigned int LINEAR_TILE_DIM= 32 >
#ifdef __CUDACC__
  void transpose(const Matrix& A, Matrix& B, Implementation, cudaStream_t stream= 0) {
#else
  void transpose(const Matrix& A, Matrix& B, Implementation) {
#endif
    static_assert(std::is_same_v<Implementation, Trans::Base> || std::is_same_v<Implementation, Trans::Tiled>
#ifdef _MKL       
                  || std::is_same_v<Implementation, Trans::Mkl>
#endif
#ifdef __CUDACC__
#ifdef _CUBLAS       
                  || std::is_same_v<Implementation, Trans::Cublas>
#endif
#endif
                 );
     if (!check_transposition_compatible_size(A, B))
       throw std::length_error{"Incompatible sizes for matrix transposition"};
    using value_type= Matrix::value_type;
    if constexpr(std::is_same_v< typename decltype(A.get_allocator())::execution, Omp> && std::is_same_v< Implementation, Trans::Base>) {
      detail::transpose<value_type>(Omp(), A.num_rows(), A.num_cols(), A.data(), B.data(), Trans::Base());
    }
    else if constexpr(std::is_same_v< typename decltype(A.get_allocator())::execution, Omp> && std::is_same_v< Implementation, Trans::Tiled>) {
      detail::transpose<value_type, LINEAR_TILE_DIM>(Omp(), A.num_rows(), A.num_cols(), A.data(), B.data(), Trans::Tiled());
    }
#ifdef _MKL
    else if constexpr(std::is_same_v< typename decltype(A.get_allocator())::execution, Omp> && std::is_same_v< Implementation, Trans::Mkl>) {
      detail::transpose<value_type>(Omp(), A.num_rows(), A.num_cols(), A.data(), B.data(), Trans::Mkl());
    }
#endif
#ifdef __CUDACC__
    else if constexpr (std::is_same_v< typename decltype(A.get_allocator())::execution, Cuda> && std::is_same_v< Implementation, Trans::Base>) {
      detail::transpose<value_type, LINEAR_TILE_DIM>(Cuda(), A.num_rows(), A.num_cols(), A.data(), B.data(), Trans::Base(), stream);
    }
    else if constexpr(std::is_same_v< typename decltype(A.get_allocator())::execution, Cuda> && std::is_same_v< Implementation, Trans::Tiled>) {
      detail::transpose<value_type, LINEAR_TILE_DIM>(Cuda(), A.num_rows(), A.num_cols(), A.data(), B.data(), Trans::Tiled(), stream);
    }
#ifdef _CUBLAS
    else if constexpr(std::is_same_v< typename decltype(A.get_allocator())::execution, Cuda> && std::is_same_v< Implementation, Trans::Cublas>) {
      detail::transpose<value_type>(Cuda(), A.num_rows(), A.num_cols(), A.data(), B.data(), Trans::Cublas());
    }
#endif
#endif
  }
      
  template<typename Matrix, typename Implementation, unsigned int LINEAR_THREAD_BLOCK_DIM= 16 >
#ifdef __CUDACC__
  constexpr decltype(auto) transpose(const Matrix& A, Implementation, cudaStream_t stream= 0) {
#else
  constexpr decltype(auto) transpose(const Matrix& A, Implementation) {
#endif
    using value_type= Matrix::value_type;
    Matrix B{A.num_cols(), A.num_rows(), static_cast<value_type>(0)};
#ifdef __CUDACC__    
    transpose<Matrix, Implementation, LINEAR_THREAD_BLOCK_DIM>(A, B, Implementation(), stream);
#else
    transpose<Matrix, Implementation, LINEAR_THREAD_BLOCK_DIM>(A, B, Implementation());
#endif
    return B;
  }

}

#endif


