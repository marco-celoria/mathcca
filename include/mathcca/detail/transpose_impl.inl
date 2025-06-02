/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include <cstddef>   // std::size_t
#include <concepts>  // std::floating_point
#include <iostream>  // std::cout
#include <stdexcept> // std::length_error

#include <mathcca/host_matrix.h>

#ifdef _MKL
 #include <mkl.h>
#endif

#ifdef __CUDACC__
 #include <cuda_runtime.h>
 #include <mathcca/device_helper.h>
 #include <mathcca/device_matrix.h>
 #ifdef _CUBLAS
  #include <cublas_v2.h>
 #endif
#endif

#ifdef _OPENMP
 #include <omp.h>
#endif

namespace mathcca {
     
  namespace detail {
    /* 
    template<std::floating_point T>
    constexpr inline auto check_transposition_compatible_size(const host_matrix<T>& lhs, const host_matrix<T>& rhs) {
      if ((lhs.num_cols() == rhs.num_rows()) && (lhs.num_rows() == rhs.num_cols()))
        return true;
      return false;
    }
     */
    template<std::floating_point T>
    constexpr void transpose(Omp, const std::size_t A_num_rows, const std::size_t A_num_cols, const T* A, T* B,Trans::Base) {
      std::cout << "Base transposition\n";
      //if (!check_transposition_compatible_size(A, B))
      //  throw std::length_error{"Incompatible sizes for matrix transposition"};
      using size_type= std::size_t;//typename host_matrix<T>::size_type;
      //const auto i_end{A.num_rows()};
      //const auto j_end{A.num_cols()};
      auto idx_Ac = [&A_num_cols](size_type i, size_type j){ return i * A_num_cols + j; };
      auto idx_Bc = [&A_num_rows](size_type i, size_type j){ return i * A_num_rows + j; };
      #pragma omp parallel for collapse(2) default(shared)
      for (size_type i= 0; i < A_num_rows; ++i) {
        for (size_type j= 0; j < A_num_cols; ++j) {
          B[idx_Bc(j, i)]= A[idx_Ac(i, j)];
        }
      }
    }
     
    template<std::floating_point T, unsigned int LINEAR_TILE_DIM>
    constexpr void transpose(Omp, const std::size_t A_num_rows, const std::size_t A_num_cols, const T* A, T* B, Trans::Tiled) {
      std::cout << "Tiled transposition\n";
      //if (!check_transposition_compatible_size(A, B))
      //  throw std::length_error{"Incompatible sizes for matrix transposition"};
      using size_type= std::size_t;//typename host_matrix<T>::size_type;
      //const auto ii_end{A.num_rows()};
      //const auto jj_end{A.num_cols()};
      auto idx_Ac = [&A_num_cols](size_type i, size_type j){ return i * A_num_cols + j; };
      auto idx_Bc = [&A_num_rows](size_type i, size_type j){ return i * A_num_rows + j; };
      const auto Ar_blocksize = std::min(static_cast<unsigned int>(A_num_rows), LINEAR_TILE_DIM);
      const auto Ac_blocksize = std::min(static_cast<unsigned int>(A_num_cols), LINEAR_TILE_DIM);
      #pragma omp parallel for collapse(2) default(shared)
      for (size_type ii= 0; ii < A_num_rows; ii+= Ar_blocksize) {
        for (size_type jj= 0; jj < A_num_cols; jj+= Ac_blocksize) {
          size_type i_end= std::min(ii + Ar_blocksize, A_num_rows);
          size_type j_end= std::min(jj + Ac_blocksize, A_num_cols);
          for (size_type i= ii; i < i_end; ++i) {
            for (size_type j= jj; j < j_end; ++j) {
              B[idx_Bc(j, i)]= A[idx_Ac(i, j)];
            }
          }
        }
      }
    }
     
#ifdef _MKL
     
    template<std::floating_point T>
    constexpr void transpose(Omp, const std::size_t A_num_rows, const std::size_t A_num_cols, const T* A, T* B, Trans::Mkl) {
      std::cout << "Mkl transposition\n";
      //if (!check_transposition_compatible_size(A, B))
      //  throw std::length_error{"Incompatible sizes for matrix transposition"};
      const T alpha{static_cast<T>(1)};
      if constexpr(std::is_same_v<T, double>) {
        mkl_domatcopy ('R', 'T', A_num_rows, A_num_cols, alpha, A, A_num_cols, B, A_num_rows);
      }
      else {
        mkl_somatcopy ('R', 'T', A_num_rows, A_num_cols, alpha, A, A_num_cols, B, A_num_rows);
       }
    }
     
#endif
     
#ifdef __CUDACC__
     /*
    template<std::floating_point T>
    constexpr inline auto check_transposition_compatible_size(const device_matrix<T>& lhs, const device_matrix<T>& rhs) {
      if ((lhs.num_cols() == rhs.num_rows()) && (lhs.num_rows() == rhs.num_cols()))
        return true;
      return false;
    }
       */ 
    template<std::floating_point T>
    __global__ void transpose_device_Base_kernel(const std::size_t A_num_rows, const std::size_t A_num_cols, const T* __restrict A, T* __restrict B) {
      const auto b_c {blockIdx.x};
      const auto b_r {blockIdx.y};
      // Thread index: 
      const auto t_c {threadIdx.x};  // tile col index
      const auto t_r {threadIdx.y};  // tile row index
      const auto col{static_cast<std::size_t>(blockDim.x * b_c + t_c)};
      const auto row{static_cast<std::size_t>(blockDim.y * b_r + t_r)};
      if (col >= A_num_cols || row >= A_num_rows)
        return;
      auto idx_A = [&A_num_cols](std::size_t i, std::size_t j){ return i * A_num_cols + j; };
      auto idx_B = [&A_num_rows](std::size_t i, std::size_t j){ return i * A_num_rows + j; };
      B[idx_B(col,row)] = A[idx_A(row, col)];
    }
        
    template <std::floating_point T, unsigned int LINEAR_THREAD_BLOCK_DIM>
    void transpose(Cuda, const std::size_t A_num_rows, const std::size_t A_num_cols, const T* A, T* B, Trans::Base, cudaStream_t stream) {
      std::cout << "Base transposition\n";
      static_assert(LINEAR_THREAD_BLOCK_DIM * LINEAR_THREAD_BLOCK_DIM <= 1024);
      //if (!check_transposition_compatible_size(A, B))
      //  throw std::length_error{"Incompatible sizes for matrix transposition"};
      using size_type = std::size_t;//typename device_matrix<T>::size_type;
      using value_type = T;
      constexpr unsigned int tile{LINEAR_THREAD_BLOCK_DIM};
      constexpr dim3 dimBlock = {tile, tile, 1}; // square
      dim3 dimGrid = {static_cast<unsigned int>((A_num_cols + static_cast<size_type>(dimBlock.x) - 1) / static_cast<size_type>(dimBlock.x)),
                      static_cast<unsigned int>((A_num_rows + static_cast<size_type>(dimBlock.y) - 1) / static_cast<size_type>(dimBlock.y)),
                      1};
      transpose_device_Base_kernel<value_type><<<dimGrid, dimBlock, 0, stream>>>(A_num_rows, A_num_cols, A, B);
      getLastCudaError("transpose_device_Base_kernel() execution failed.\n");
    }
       
    template<std::floating_point T, unsigned int LINEAR_THREAD_BLOCK_DIM>
    __global__ void transpose_parallel_Tiled_kernel(const std::size_t A_num_rows, const std::size_t A_num_cols, const T* __restrict A, T* __restrict B) {
      __shared__ float block[LINEAR_THREAD_BLOCK_DIM][LINEAR_THREAD_BLOCK_DIM + 1];
      // read the matrix tile into shared memory
      // load one element per thread from device memory (idata) and store it
      // in transposed order in block[][]
      // Block index
      const auto b_c {blockIdx.x};
      const auto b_r {blockIdx.y};
      // Thread index: 
      const auto t_c {threadIdx.x};  // tile col index
      const auto t_r {threadIdx.y};  // tile row index
      // Identify the column and row of the C element to work on
      auto col {static_cast<std::size_t>(b_c * LINEAR_THREAD_BLOCK_DIM + t_c)};
      auto row {static_cast<std::size_t>(b_r * LINEAR_THREAD_BLOCK_DIM + t_r)};
      auto idx_A = [&A_num_cols](std::size_t i, std::size_t j){ return i * A_num_cols + j; };
      auto idx_B = [&A_num_rows](std::size_t i, std::size_t j){ return i * A_num_rows + j; };
      if((col < A_num_cols) && (row < A_num_rows)) {
        block[t_r][t_c] = A[idx_A(row, col)];
      }
      // synchronise to ensure all writes to block[][] have completed
      __syncthreads();
      col = b_r * LINEAR_THREAD_BLOCK_DIM + t_c;
      row = b_c * LINEAR_THREAD_BLOCK_DIM + t_r;
      // write the transposed matrix tile to global memory (odata) in linear order
      if((col < A_num_rows) && (row < A_num_cols)) {
        B[idx_B(row,col)] = block[t_c][t_r];
      }
    }
     
    template <std::floating_point T, unsigned int LINEAR_THREAD_BLOCK_DIM>
    void transpose(Cuda, const std::size_t A_num_rows, const std::size_t A_num_cols, const T* A, T* B, Trans::Tiled, cudaStream_t stream) {
      std::cout << "Tiled transposition\n";
      static_assert(LINEAR_THREAD_BLOCK_DIM * LINEAR_THREAD_BLOCK_DIM <= 1024);
      //if (!check_transposition_compatible_size(A, B))////////////////////////////////////////////
      //  throw std::length_error{"Incompatible sizes for matrix transposition"};
      using size_type = std::size_t;//typename device_matrix<T>::size_type;
      using value_type = T;
      constexpr unsigned int tile{LINEAR_THREAD_BLOCK_DIM};
      constexpr dim3 dimBlock = {tile, tile, 1}; // square
      dim3 dimGrid = {static_cast<unsigned int>((A_num_cols + static_cast<size_type>(dimBlock.x) - 1) / static_cast<size_type>(dimBlock.x)),
                      static_cast<unsigned int>((A_num_rows + static_cast<size_type>(dimBlock.y) - 1) / static_cast<size_type>(dimBlock.y)),
                      1};
      transpose_parallel_Tiled_kernel<value_type, tile><<<dimGrid, dimBlock, 0, stream>>>(A_num_rows, A_num_cols, A, B);
      getLastCudaError("transpose_parallel_Tiled_kernel() execution failed.\n");
    }
      
#ifdef _CUBLAS
     
    template <std::floating_point T>
    void  transpose(Cuda, const std::size_t A_num_rows, const std::size_t A_num_cols, const T* A, T* B, Trans::Cublas) {
      std::cout << "Cublas transposition\n";
      //if (!check_transposition_compatible_size(A, B))
      //  throw std::length_error{"Incompatible sizes for matrix transposition"};
      const T alpha{static_cast<T>(1)};
      const T  beta{static_cast<T>(0)};
      cublasHandle_t handle;
      checkCudaErrors(cublasCreate(&handle));
      if constexpr(std::is_same_v<T, double>) {
        checkCudaErrors(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                    A_num_rows, A_num_cols, &alpha, A, A_num_cols, &beta, A, A_num_cols, B, A_num_rows));
      }
      else {
        //std::cout << "A.num_rows()=" << A_num_rows << " A.num_cols()=" << A_num_cols << "\n";
        //std::cout << "B.num_rows()=" << A_num_cols << " B.num_cols()=" << A_num_rows << "\n";
        checkCudaErrors(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                    A_num_rows, A_num_cols, &alpha, A, A_num_cols, &beta, A, A_num_cols, B, A_num_rows));
      }
      checkCudaErrors(cublasDestroy(handle));
    }
     
#endif
     
#endif
     
  }
     
}     



