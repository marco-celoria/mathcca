#ifndef MATMUL_H_
#define MATMUL_H_
#pragma once

#include <mathcca/host_matrix.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <mathcca/device_matrix.h>
#endif

#include <mathcca/detail/matmul_impl.h>

namespace mathcca {
     /* 
  template<typename Matrix, typename Implementation, unsigned int LINEAR_TILE_DIM= 32>
  void matmul(const Matrix& A, const Matrix& B, Matrix& C, Implementation) {
    static_assert(std::is_same_v<Implementation, MM::Base> || std::is_same_v<Implementation, MM::Tiled>
#ifdef _MKL       
                  || std::is_same_v<Implementation, MM::Mkl>
#endif            
                 );
    using value_type= Matrix::value_type;
    if constexpr(std::is_same_v< typename decltype(A.get_allocator())::execution, Omp> && std::is_same_v< Implementation, MM::Base>) {
     detail::matmul<value_type, LINEAR_THREAD_BLOCK_DIM> (Omp(), A.num_rows(), B.num_cols(), A.num_cols(), A.data(), B.data(), C.data(), MM::Base());
    }
    else if constexpr(std::is_same_v< typename decltype(A.get_allocator())::execution, Omp> && std::is_same_v< Implementation, MM::Tiled>) {
      detail::matmul<value_type, LINEAR_THREAD_BLOCK_DIM> (Omp(), A.num_rows(), B.num_cols(), A.num_cols(), A.data(), B.data(), C.data(), MM::Tiled());
    }
#ifdef _MKL
    else if constexpr(std::is_same_v< typename decltype(A.get_allocator())::execution, Omp> && std::is_same_v< Implementation, MM::Mkl>) {
detail::matmul<value_type> (Omp(), A.num_rows(), B.num_cols(), A.num_cols(), A.data(), B.data(), C.data(), MM::Mkl());
    }
#endif
  }
*/
  template<typename Matrix>
  constexpr inline auto check_matmul_compatible_size(const Matrix& lhs, const Matrix& rhs) {
      if (lhs.num_cols() == rhs.num_rows())
        return true;
      return false;
    }


  template<typename Matrix, typename Implementation, unsigned int LINEAR_THREAD_BLOCK_DIM= 16 >
#ifdef __CUDACC__
  void matmul(const Matrix& A, const Matrix& B, Matrix& C, Implementation, cudaStream_t stream= 0) {
#else
  void matmul(const Matrix& A, const Matrix& B, Matrix& C, Implementation) {
#endif
    static_assert(std::is_same_v<Implementation, MM::Base> || std::is_same_v<Implementation, MM::Tiled>
#ifdef _MKL       
                  || std::is_same_v<Implementation, MM::Mkl>
#endif  
#ifdef __CUDACC__
#ifdef _CUBLAS       
                  || std::is_same_v<Implementation, MM::Cublas>
#endif
#endif
                 );
    if (!check_matmul_compatible_size(A, B))
        throw std::length_error{"Incompatible length matrix-matrix product"};
    using value_type= Matrix::value_type;
    if constexpr(std::is_same_v< typename decltype(A.get_allocator())::execution, Omp> && std::is_same_v< Implementation, MM::Base>) {
     detail::matmul<value_type> (Omp(), A.num_rows(), B.num_cols(), A.num_cols(), A.data(), B.data(), C.data(), MM::Base());
    }
    else if constexpr(std::is_same_v< typename decltype(A.get_allocator())::execution, Omp> && std::is_same_v< Implementation, MM::Tiled>) {
      detail::matmul<value_type, LINEAR_THREAD_BLOCK_DIM> (Omp(), A.num_rows(), B.num_cols(), A.num_cols(), A.data(), B.data(), C.data(), MM::Tiled());
    }
#ifdef _MKL
    else if constexpr(std::is_same_v< typename decltype(A.get_allocator())::execution, Omp> && std::is_same_v< Implementation, MM::Mkl>) {
detail::matmul<value_type> (Omp(), A.num_rows(), B.num_cols(), A.num_cols(), A.data(), B.data(), C.data(), MM::Mkl());
    }
#endif
#ifdef __CUDACC__
    if constexpr( std::is_same_v< typename decltype(A.get_allocator())::execution, Cuda> && std::is_same_v<Implementation, MM::Base>) {
     detail::matmul<value_type, LINEAR_THREAD_BLOCK_DIM> (Cuda(), A.num_rows(), B.num_cols(), A.num_cols(), A.data(), B.data(), C.data(), MM::Base(),stream);
    }
    else if constexpr(std::is_same_v< typename decltype(A.get_allocator())::execution, Cuda> && std::is_same_v<Implementation, MM::Tiled>) {
	    detail::matmul<value_type, LINEAR_THREAD_BLOCK_DIM> (Cuda(), A.num_rows(), B.num_cols(), A.num_cols(), A.data(), B.data(), C.data(), MM::Tiled(), stream);
    }
#ifdef _CUBLAS
    else if constexpr(std::is_same_v< typename decltype(A.get_allocator())::execution, Cuda> && std::is_same_v< Implementation, MM::Cublas>) {
	    detail::matmul<value_type>(Cuda(), A.num_rows(), B.num_cols(), A.num_cols(), A.data(), B.data(), C.data(), MM::Cublas());
    }
#endif
#endif
  }
      
  template<typename Matrix, typename Implementation, unsigned int LINEAR_THREAD_BLOCK_DIM= 16 >
  constexpr decltype(auto) matmul(const Matrix& A, const Matrix& B, Implementation, cudaStream_t stream= 0) {
    using value_type= Matrix::value_type;
    Matrix C{A.num_rows(), B.num_cols(), static_cast<value_type>(0)};
    matmul<Matrix, Implementation, LINEAR_THREAD_BLOCK_DIM>(A, B, C, Implementation(), stream);
    return C;
  }


}

#endif


