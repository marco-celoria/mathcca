#include <cstddef>

#ifdef _CUBLAS
 #include <cublas_v2.h>
#endif

namespace mathcca {
  
  namespace matricca {
    
    template<std::floating_point T>
    constexpr inline auto check_transposition_compatible_size(const device_matrix<T>& lhs, const device_matrix<T>& rhs) {
      if ((lhs.num_cols() == rhs.num_rows()) && (lhs.num_rows() == rhs.num_cols()))
        return true;
      return false;
    }
    
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
    void transpose_device_Base(const device_matrix<T>& A, device_matrix<T>& B, cudaStream_t stream) {
      std::cout << "Base transposition\n";
      static_assert(LINEAR_THREAD_BLOCK_DIM * LINEAR_THREAD_BLOCK_DIM <= 1024);
      if (!check_transposition_compatible_size(A, B))
        throw std::length_error{"Incompatible sizes for matrix transposition"};
      using size_type = typename device_matrix<T>::size_type;
      using value_type = T;
      constexpr unsigned int tile{LINEAR_THREAD_BLOCK_DIM};
      constexpr dim3 dimBlock = {tile, tile, 1}; // square
      dim3 dimGrid = {static_cast<unsigned int>((A.num_cols() + static_cast<size_type>(dimBlock.x) - 1) / static_cast<size_type>(dimBlock.x)),
                      static_cast<unsigned int>((A.num_rows() + static_cast<size_type>(dimBlock.y) - 1) / static_cast<size_type>(dimBlock.y)),
                      1};
      transpose_device_Base_kernel<value_type><<<dimGrid, dimBlock, 0, stream>>>(A.num_rows(), A.num_cols(), A.data(), B.data());
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
    void transpose_device_Tiled(const device_matrix<T>& A, device_matrix<T>& B, cudaStream_t stream) {
      std::cout << "Tiled transposition\n";
      static_assert(LINEAR_THREAD_BLOCK_DIM * LINEAR_THREAD_BLOCK_DIM <= 1024);
      if (!check_transposition_compatible_size(A, B))////////////////////////////////////////////
        throw std::length_error{"Incompatible sizes for matrix transposition"};
      using size_type = typename device_matrix<T>::size_type;
      using value_type = T;
      constexpr unsigned int tile{LINEAR_THREAD_BLOCK_DIM};
      constexpr dim3 dimBlock = {tile, tile, 1}; // square
      dim3 dimGrid = {static_cast<unsigned int>((A.num_cols() + static_cast<size_type>(dimBlock.x) - 1) / static_cast<size_type>(dimBlock.x)),
                      static_cast<unsigned int>((A.num_rows() + static_cast<size_type>(dimBlock.y) - 1) / static_cast<size_type>(dimBlock.y)),
                      1};
      transpose_parallel_Tiled_kernel<value_type, tile><<<dimGrid, dimBlock, 0, stream>>>(A.num_rows(), A.num_cols(), A.data(), B.data());
    }
    
#ifdef _CUBLAS
    template <std::floating_point T>
    void  transpose_device_Cublas(const device_matrix<T>& A, device_matrix<T>& B) {
      std::cout << "Cublas transposition\n";
      if (!check_transposition_compatible_size(A, B))
        throw std::length_error{"Incompatible sizes for matrix transposition"};
      const T alpha{static_cast<T>(1)};
      const T  beta{static_cast<T>(0)};
      cublasHandle_t handle;
      checkCudaErrors(cublasCreate(&handle));
      if constexpr(std::is_same_v<T, double>) {
        checkCudaErrors(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                    A.num_rows(), A.num_cols(), &alpha, A.data(),
                    A.num_cols(), &beta, A.data(),
                    A.num_cols(), B.data(), A.num_rows()));
      }
      else {
        std::cout << "A.num_rows()=" << A.num_rows() << " A.num_cols()=" << A.num_cols() << "\n";
        std::cout << "B.num_rows()=" << B.num_rows() << " B.num_cols()=" << B.num_cols() << "\n";
        checkCudaErrors(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                    A.num_rows(), A.num_cols(), &alpha, A.data(),
                    A.num_cols(), &beta, A.data(),
                    A.num_cols(), B.data(), A.num_rows()));
      }
      checkCudaErrors(cublasDestroy(handle));
    }
#endif

  }

}

