#include <cstddef>

#ifdef _CUBLAS
 #include <cublas_v2.h>
#endif

namespace mathcca {
  
  namespace matricca {
  
    template<std::floating_point T>
    constexpr auto check_matmul_compatible_size(const device_matrix<T>& lhs, const device_matrix<T>& rhs) {
      if (lhs.num_cols() == rhs.num_rows())
        return true;
      return false;
    }

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
    void mm_device_Base(const device_matrix<T>& A, const device_matrix<T>& B, device_matrix<T>& C, cudaStream_t stream) {
      std::cout << "Base matmul\n";
      static_assert(LINEAR_THREAD_BLOCK_DIM * LINEAR_THREAD_BLOCK_DIM <= 1024);
      if (!check_matmul_compatible_size(A, B))
        throw std::length_error{"Incompatible length matrix-matrix product"};      
      using size_type = typename device_matrix<T>::size_type;
      using value_type = T;
      constexpr unsigned int tile{LINEAR_THREAD_BLOCK_DIM};
      constexpr dim3 dimBlock = {tile, tile, 1}; // square
      dim3 dimGrid = {static_cast<unsigned int>((B.num_cols() + static_cast<size_type>(dimBlock.x) - 1) / static_cast<size_type>(dimBlock.x)),
                      static_cast<unsigned int>((A.num_rows() + static_cast<size_type>(dimBlock.y) - 1) / static_cast<size_type>(dimBlock.y)),
                      1};
      mm_device_Base_kernel<value_type><<<dimGrid, dimBlock, 0, stream>>>(A.num_rows(), B.num_cols(), A.num_cols(), A.data(), B.data(), C.data());
    }  
      
    template < std::floating_point T, unsigned int LINEAR_THREAD_BLOCK_DIM >
    __global__ void mm_device_Tiled_kernel(const std::size_t A_num_rows, const std::size_t B_num_cols, const std::size_t A_num_cols,
                                           const T * __restrict A, const T * __restrict B, T * __restrict C) {
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
    void mm_device_Tiled(const device_matrix<T>& A, const device_matrix<T>& B, device_matrix<T>& C, cudaStream_t stream)  {
      std::cout << "Tiled matmul\n";
      static_assert(LINEAR_THREAD_BLOCK_DIM * LINEAR_THREAD_BLOCK_DIM <= 1024);
      if (!check_matmul_compatible_size(A, B))
        throw std::length_error{"Incompatible length matrix-matrix product"};
      
      using size_type = typename device_matrix<T>::size_type;
      using value_type = T;
      constexpr unsigned int tile{LINEAR_THREAD_BLOCK_DIM};
      constexpr dim3 dimBlock = {tile, tile, 1}; // square
      dim3 dimGrid = {static_cast<unsigned int>((B.num_cols() + static_cast<size_type>(dimBlock.x) - 1) / static_cast<size_type>(dimBlock.x)),
                      static_cast<unsigned int>((A.num_rows() + static_cast<size_type>(dimBlock.y) - 1) / static_cast<size_type>(dimBlock.y)),
                      1};
      mm_device_Tiled_kernel<value_type, LINEAR_THREAD_BLOCK_DIM><<<dimGrid, dimBlock, 0, stream>>>(A.num_rows(), B.num_cols(), A.num_cols(), A.data(), B.data(), C.data());
    }  
    
#ifdef _CUBLAS
    template <std::floating_point T>
    void  mm_device_Cublas(const device_matrix<T>& A, const device_matrix<T>& B, device_matrix<T>& C)  {
      std::cout << "Cublas matmul\n";
      if (!check_matmul_compatible_size(A, B))
        throw std::length_error{"Incompatible length matrix-matrix product"};
      const T alpha{static_cast<T>(1)};
      const T  beta{static_cast<T>(0)};;
      cublasHandle_t handle;
      checkCudaErrors(cublasCreate(&handle));
      if constexpr(std::is_same_v<T, double>) {
        checkCudaErrors(cublasDgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, B.num_cols(), A.num_rows(), A.num_cols(),
            &alpha, B.data(), B.num_cols(), A.data(), A.num_cols(),
            &beta,  C.data(), B.num_cols()));
      }
      else {
        checkCudaErrors(cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, B.num_cols(), A.num_rows(), A.num_cols(),
            &alpha, B.data(), B.num_cols(), A.data(), A.num_cols(),
            &beta,  C.data(), B.num_cols()));
      }
      checkCudaErrors(cublasDestroy(handle));
    }
#endif
   
  }

}
