#include <cstddef>
#include <random>
#include <curand.h>
#include <curand_kernel.h>
#include <mathcca/device_helper.h>

namespace mathcca {
  
  namespace algocca {
  
    __global__ void init_seed(curandState* state, unsigned int seed, const std::size_t size) {
      auto idx{blockIdx.x * blockDim.x + threadIdx.x};
      if (idx < size) {
        curand_init(seed, idx, 0, &state[idx]);
      }
    }
    
    template<std::floating_point T>
    __global__ void fill_rand_kernel(curandState *state, T *randArray, const std::size_t size) {
      auto idx{blockIdx.x * blockDim.x + threadIdx.x};
      if (idx < size) {
        randArray[idx] = curand_uniform(&state[idx]);
      }
    }
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    void fill_rand(mathcca::iterator::device_iterator<T> first, mathcca::iterator::device_iterator<T> last, cudaStream_t stream) {
      static_assert(THREAD_BLOCK_DIM <= 1024);
      using value_type= T;
      const auto size {static_cast<std::size_t>(last - first)};
      constexpr unsigned int threads{THREAD_BLOCK_DIM};
      const auto blocks{static_cast<unsigned int>((size + static_cast<std::size_t>(threads) - 1) / (static_cast<std::size_t>(threads)))};
      constexpr dim3 dimBlock(threads, 1, 1);
      dim3 dimGrid(blocks, 1, 1);
      std::random_device rd;
      unsigned int seed = rd();
      curandState *d_state;
      checkCudaErrors(cudaMalloc(&d_state, (size * sizeof(curandState))));
      init_seed<<<dimGrid, dimBlock, 0, stream>>>(d_state, seed, size);
      fill_rand_kernel<value_type><<<dimGrid, dimBlock, 0, stream>>>(d_state, first.get(), size);
      checkCudaErrors(cudaFree(d_state));
    }
    
  }
  
}

