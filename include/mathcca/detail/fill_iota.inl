#include <cstddef>

#ifdef __CUDACC__
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#endif

#include <cstddef>
#ifdef _PARALG
#include <execution>
#include <ranges>
#endif

namespace mathcca {

#ifdef _PARALG
    template<std::floating_point T>
    void fill_iota(StdPar, T* first, T* last, const T v) {
      //std::cout << "DEBUG _PARALG\n";
      using value_type= T;
      const auto size {static_cast<std::size_t>(last - first)};
      std::ranges::iota_view indices(static_cast<unsigned int>(0),static_cast<unsigned int>(size));
      std::for_each(std::execution::par_unseq,indices.begin(),indices.end(),[&](auto i) { first[i]= v + static_cast<value_type>(i); });
    }
#endif
    template<std::floating_point T>
    void fill_iota(Omp, T* first, T* last, const T v) {
      //std::cout << "DEBUG NO _PARALG\n";
      using value_type= T;
      const auto size {static_cast<std::size_t>(last - first)};
      #pragma omp prallel for default(shared)
      for (std::size_t i= 0; i < size; ++i) {
        first[i]= v + static_cast<value_type>(i);
      }
    }

}


#ifdef __CUDACC__
namespace mathcca {

    template<std::floating_point T>
    void fill_iota(Thrust, T* first, T* last, const T v) {
      thrust::sequence(thrust::device, first, last, v);
    }

    template<std::floating_point T>
    __global__ void fill_iota_kernel(T* __restrict start, const std::size_t size, const T v) {
      const auto tid{static_cast<std::size_t>(threadIdx.x + blockIdx.x * blockDim.x)};
      if(tid < size) {
        start[tid] = v + tid;
      }
    }
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    void fill_iota(Cuda, T* first, T* last, const T v, cudaStream_t stream) {
      static_assert(THREAD_BLOCK_DIM <= 1024);
      using value_type= T;
      const auto size {static_cast<std::size_t>(last - first)};
      constexpr unsigned int threads{THREAD_BLOCK_DIM};
      const auto blocks{static_cast<unsigned int>((size + static_cast<std::size_t>(threads) - 1) / (static_cast<std::size_t>(threads)))};
      constexpr dim3 dimBlock(threads, 1, 1);
      dim3 dimGrid(blocks, 1, 1);
      fill_iota_kernel<value_type><<<dimGrid, dimBlock, 0, stream>>>(first, size, v);
    }
    
}
#endif


