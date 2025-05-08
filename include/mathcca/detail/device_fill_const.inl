#include <cstddef>


namespace mathcca {

    template<std::floating_point T>
    __global__ void fill_const_kernel(T* start, const std::size_t size, const T v) {
      const auto tid{static_cast<std::size_t>(threadIdx.x + blockIdx.x * blockDim.x)};
      if(tid < size) {
        start[tid] = v;
      }
    }

      template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    void fill_const(Cuda, T* first, T* last, const T v, cudaStream_t stream) {
      static_assert(THREAD_BLOCK_DIM <= 1024);
      using value_type= T;
      const auto size {static_cast<std::size_t>(last - first)};
      constexpr unsigned int threads{THREAD_BLOCK_DIM};
      const auto blocks{static_cast<unsigned int>((size + static_cast<std::size_t>(threads) - 1) / (static_cast<std::size_t>(threads)))};
      constexpr dim3 dimBlock(threads, 1, 1);
      dim3 dimGrid(blocks, 1, 1);
      fill_const_kernel<value_type><<<dimGrid, dimBlock, 0, stream>>>(first, size, v);
    }

}



