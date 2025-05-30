
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
    constexpr inline auto check_matmul_compatible_size(const host_matrix<T>& lhs, const host_matrix<T>& rhs) {
      if (lhs.num_cols() == rhs.num_rows())
        return true;
      return false;
    }
    */
    template<std::floating_point T>
    constexpr void matmul(Omp, const std::size_t A_num_rows, const std::size_t B_num_cols, const std::size_t A_num_cols, const T* A, const T* B, T* C, MM::Base) {
      std::cout << "OMP Base matmul\n";
      //if (!check_matmul_compatible_size(A, B))
      //  throw std::length_error{"Incompatible length matrix-matrix product"};
      using size_type= std::size_t;
      using value_type= T;
      auto idx_Ac = [&A_num_cols](size_type i, size_type j){ return i * A_num_cols + j; };
      auto idx_Bc = [&B_num_cols](size_type i, size_type j){ return i * B_num_cols + j; };
      #pragma omp parallel for collapse(2) default(shared)
      for (size_type i= 0; i < A_num_rows; ++i) {
        for (size_type j= 0; j < B_num_cols; ++j) {
          auto sum{static_cast<value_type>(0)};
          #pragma omp simd reduction(+:sum)
          for (size_type k= 0; k < A_num_cols; ++k) {
            sum+= A[idx_Ac(i, k)] * B[idx_Bc(k,j)];
          }
          C[idx_Bc(i,j)]= sum;
        }
      }
    }
       
    template<std::floating_point T, unsigned int LINEAR_TILE_DIM>
    constexpr void matmul(Omp, const std::size_t A_num_rows, const std::size_t B_num_cols, const std::size_t A_num_cols, const T* A, const T* B, T* C, MM::Tiled) {
      std::cout << "OMP Tiled matmul\n";
      //if (!check_matmul_compatible_size(A, B))
      //  throw std::length_error{"Incompatible length matrix-matrix product"};
      using size_type= std::size_t;
      auto idx_Ac = [&A_num_cols](size_type i, size_type j){ return i * A_num_cols + j; };
      auto idx_Bc = [&B_num_cols](size_type i, size_type j){ return i * B_num_cols + j; };
      const auto Ar_blocksize = std::min(static_cast<unsigned int>(A_num_rows), LINEAR_TILE_DIM);
      const auto Bc_blocksize = std::min(static_cast<unsigned int>(B_num_cols), LINEAR_TILE_DIM);
      const auto Ac_blocksize = std::min(static_cast<unsigned int>(A_num_cols), LINEAR_TILE_DIM);
      #pragma omp parallel for collapse(2) default(shared)
      for (size_type ii= 0; ii < A_num_rows; ii+= Ar_blocksize) {
        for (size_type jj= 0; jj < B_num_cols; jj+= Bc_blocksize) {
          size_type i_end= std::min(ii + Ar_blocksize, A_num_rows);
          size_type j_end= std::min(jj + Bc_blocksize, B_num_cols);
          for (size_type kk= 0; kk < A_num_cols; kk+= Ac_blocksize) {
            size_type k_end = std::min(kk + Ac_blocksize, A_num_cols);
            for (size_type i= ii; i < i_end; ++i) {
              for (size_type k= kk; k < k_end; ++k) {
                for (size_type j= jj; j < j_end; ++j) {
                  C[idx_Bc(i,j)]+= A[idx_Ac(i, k)] * B[idx_Bc(k,j)];
		}
              }
            }
          }
        }
      }
    }
    
#ifdef _MKL
       
    template<std::floating_point T>
    constexpr void matmul(Omp, const std::size_t A_num_rows, const std::size_t B_num_cols, const std::size_t A_num_cols, const T* A, const T* B, T* C, MM::Mkl) {
      std::cout << "OMP Mkl matmul\n";
      //if (!check_matmul_compatible_size(A, B))
      //  throw std::length_error{"Incompatible length matrix-matrix product"};
      //using size_type= typename host_matrix<T>::size_type;
      using value_type= T;
      value_type alpha{1};
      value_type beta{0};
      if constexpr(std::is_same_v<T,double>) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A_num_rows, B_num_cols, A_num_cols,
                 alpha, A, A_num_cols, B, B_num_cols, beta, C, B_num_cols);
      } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A_num_rows, B_num_cols, A_num_cols,
                 alpha, A, A_num_cols, B, B_num_cols, beta, C, B_num_cols);
      }
    }
       
#endif
    
#ifdef __CUDACC__ 
    /*
    template<std::floating_point T>
    constexpr auto check_matmul_compatible_size(const device_matrix<T>& lhs, const device_matrix<T>& rhs) {
      if (lhs.num_cols() == rhs.num_rows())
        return true;
      return false;
    }
      */
    template < std::floating_point T>
    __global__ void mm_device_Base_kernel(const std::size_t A_num_rows, const std::size_t B_num_cols, const std::size_t A_num_cols,
                                          const T * __restrict A,       const T * __restrict B,       T * __restrict C) {
        
      const auto col {static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x)};
      const auto row {static_cast<std::size_t>(blockIdx.y * blockDim.y + threadIdx.y)};
      if(row >= A_num_rows || col >= B_num_cols)
         return;
      auto idx_Ac = [&A_num_cols](std::size_t i, std::size_t j){ return i * A_num_cols + j; };
      auto idx_Bc = [&B_num_cols](std::size_t i, std::size_t j){ return i * B_num_cols + j; };
      auto Cvalue{static_cast<T>(0)};
      for (std::size_t k= 0; k < A_num_cols; ++k) {
        Cvalue+= A[idx_Ac(row, k)] * B[idx_Bc(k, col)];
      }
      C[idx_Bc(row, col)]= Cvalue;
    } 
      
    template <std::floating_point T, unsigned int LINEAR_THREAD_BLOCK_DIM>
    void matmul(Cuda, const std::size_t A_num_rows, const std::size_t B_num_cols, const std::size_t A_num_cols, const T* A, const T* B, T* C, MM::Base, cudaStream_t stream) {
      std::cout << "CUDA Base matmul\n";
      static_assert(LINEAR_THREAD_BLOCK_DIM * LINEAR_THREAD_BLOCK_DIM <= 1024);
      //if (!check_matmul_compatible_size(A, B))
      //  throw std::length_error{"Incompatible length matrix-matrix product"};      
      using size_type= std::size_t;
      using value_type= T;
      constexpr unsigned int tile{LINEAR_THREAD_BLOCK_DIM};
      constexpr dim3 dimBlock = {tile, tile, 1}; // square
      dim3 dimGrid = {static_cast<unsigned int>((B_num_cols + static_cast<size_type>(dimBlock.x) - 1) / static_cast<size_type>(dimBlock.x)),
                      static_cast<unsigned int>((A_num_rows + static_cast<size_type>(dimBlock.y) - 1) / static_cast<size_type>(dimBlock.y)),
                      1};
      mm_device_Base_kernel<value_type><<<dimGrid, dimBlock, 0, stream>>>(A_num_rows, B_num_cols, A_num_cols, A, B, C);
      getLastCudaError("mm_device_Base_kernel() execution failed.\n");
    }  
      
    template < std::floating_point T, unsigned int LINEAR_THREAD_BLOCK_DIM >
    __global__ void mm_device_Tiled_kernel(const std::size_t A_num_rows, const std::size_t B_num_cols, const std::size_t A_num_cols, const T * __restrict A, const T * __restrict B, T * __restrict C) {
      // Declaration of the shared memory array As used to store the sub-matrix of A: tile A e.g. [16][16]
      __shared__ T Atile[LINEAR_THREAD_BLOCK_DIM][LINEAR_THREAD_BLOCK_DIM];
      // extern __shared__ T Atile[];
      // Declaration of the shared memory array Bs used to store the sub-matrix of B: tile B e.g. [16][16]
      __shared__ T Btile[LINEAR_THREAD_BLOCK_DIM][LINEAR_THREAD_BLOCK_DIM];
      // extern __shared__ T Btile[];
      // Block index
      const auto b_c {blockIdx.x};
      const auto b_r {blockIdx.y};
      // Thread index:
      const auto t_c {threadIdx.x};  // tile col index
      const auto t_r {threadIdx.y};  // tile row index
      // Identify the column and row of the C element to work on
      const auto col {static_cast<std::size_t>(b_c * LINEAR_THREAD_BLOCK_DIM + t_c)};
      const auto row {static_cast<std::size_t>(b_r * LINEAR_THREAD_BLOCK_DIM + t_r)};
      auto idx_Ac = [&A_num_cols](std::size_t i, std::size_t j){ return i * A_num_cols + j; };
      auto idx_Bc = [&B_num_cols](std::size_t i, std::size_t j){ return i * B_num_cols + j; };
      auto Cvalue{static_cast<T>(0)};
      // Loop over the A and B tiles required to compute the C element
      const auto num_tiles {static_cast<unsigned int>((A_num_cols + LINEAR_THREAD_BLOCK_DIM - 1)/(LINEAR_THREAD_BLOCK_DIM))};
      for(auto tile= 0; tile < num_tiles; ++tile) {
        const auto col_A {static_cast<std::size_t>(tile * LINEAR_THREAD_BLOCK_DIM + t_c)};
        const auto row_B {static_cast<std::size_t>(tile * LINEAR_THREAD_BLOCK_DIM + t_r)};
        if ((row < A_num_rows) && (col_A < A_num_cols))
          Atile[t_r][t_c]= A[idx_Ac(row, col_A)];  // copy A to shared mem
        else
          Atile[t_r][t_c]= static_cast<T>(0);
        if ((row_B < A_num_cols) && col < B_num_cols)
          Btile[t_r][t_c]= B[idx_Bc(row_B, col)];  // copy B to shared mem
        else
          Btile[t_r][t_c]= static_cast<T>(0);
        __syncthreads();
        for(auto k= 0; k < LINEAR_THREAD_BLOCK_DIM; k++) {
          Cvalue+= Atile[t_r][k] * Btile[k][t_c];
        }
        __syncthreads();
      }
      if ((row < A_num_rows) && (col < B_num_cols))
        C[idx_Bc(row, col)]= Cvalue;  // store complete result
    }
      
    template <std::floating_point T, unsigned int LINEAR_THREAD_BLOCK_DIM>
    void matmul(Cuda, const std::size_t A_num_rows, const std::size_t B_num_cols, const std::size_t A_num_cols, const T* A, const T* B, T* C, MM::Tiled, cudaStream_t stream)  {
      std::cout << "CUDA Tiled matmul\n";
      static_assert(LINEAR_THREAD_BLOCK_DIM * LINEAR_THREAD_BLOCK_DIM <= 1024);
      //if (!check_matmul_compatible_size(A, B))
      //  throw std::length_error{"Incompatible length matrix-matrix product"};
      
      using size_type = std::size_t;
      using value_type = T;
      constexpr unsigned int tile{LINEAR_THREAD_BLOCK_DIM};
      constexpr dim3 dimBlock = {tile, tile, 1}; // square
      dim3 dimGrid = {static_cast<unsigned int>((B_num_cols + static_cast<size_type>(dimBlock.x) - 1) / static_cast<size_type>(dimBlock.x)),
                      static_cast<unsigned int>((A_num_rows + static_cast<size_type>(dimBlock.y) - 1) / static_cast<size_type>(dimBlock.y)),
                      1};
      mm_device_Tiled_kernel<value_type, LINEAR_THREAD_BLOCK_DIM><<<dimGrid, dimBlock, 0, stream>>>(A_num_rows, B_num_cols, A_num_cols, A, B, C);
      getLastCudaError("mm_device_Tiled_kernel() execution failed.\n");
    }  
    
#ifdef _CUBLAS
    
    template <std::floating_point T>
    void matmul(Cuda, const std::size_t A_num_rows, const std::size_t B_num_cols, const std::size_t A_num_cols, const T* A, const T* B, T* C, MM::Cublas)  {
      std::cout << "CUDA Cublas matmul\n";
      //if (!check_matmul_compatible_size(A, B))
      //  throw std::length_error{"Incompatible length matrix-matrix product"};
      const T alpha{static_cast<T>(1)};
      const T  beta{static_cast<T>(0)};;
      cublasHandle_t handle;
      checkCudaErrors(cublasCreate(&handle));
      if constexpr(std::is_same_v<T, double>) {
        checkCudaErrors(cublasDgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, B_num_cols, A_num_rows, A_num_cols,
            &alpha, B, B_num_cols, A, A_num_cols, &beta,  C, B_num_cols));
      }
      else {
        checkCudaErrors(cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, B_num_cols, A_num_rows, A_num_cols,
            &alpha, B, B_num_cols, A, A_num_cols, &beta,  C, B_num_cols));
      }
      checkCudaErrors(cublasDestroy(handle));
    }
    
#endif
    
#endif
    
  }  
    
}


