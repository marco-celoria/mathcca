
#include <cstddef> // std::size_t
#include <concepts> //std::floating_point
#include <iostream> //std::cout
#include <random>

// StdPar Omp Thrust Cuda
#include <mathcca/common_algorithm.h>

#ifdef __CUDACC__
 #include <cuda_runtime.h>
 #include <mathcca/device_helper.h>
 #include <curand.h>
 #include <curand_kernel.h>
 #ifdef _THRUST
  #include <thrust/transform.h>
  #include <thrust/iterator/counting_iterator.h>
  #include <thrust/random.h>
 #endif
#endif

#ifdef _PARALG
 #include <execution>
 #include <ranges>
#endif

#ifdef _OPENMP
 #include<omp.h>
#endif

namespace mathcca {
    
  namespace detail {
    
#ifdef _STDPAR
    
    template<std::floating_point T>
      inline static T Uniform(T min, T max) {
      static thread_local std::mt19937 generator{std::random_device{}()};
      std::uniform_real_distribution<T> distribution{min,max};
      return distribution(generator);
    }
    
    template<std::floating_point T>
    void fill_rand(StdPar, T* first, T* last) {
    const auto size {static_cast<std::size_t>(last - first)};
      std::cout << "DEBUG STDPAR\n";
      std::ranges::iota_view r(static_cast<unsigned int>(0),static_cast<unsigned int>(size));
      std::for_each(std::execution::par_unseq,r.begin(), r.end(), [&](auto i) {first[i] = Uniform(static_cast<T>(0), static_cast<T>(1));});
    }
    
#endif
    
    template<std::floating_point T>
    void fill_rand(Omp, T* first, T* last) {
      std::cout << "DEBUG OMP\n";
      const auto size {static_cast<std::size_t>(last - first)};
      std::random_device rd;
      #pragma omp parallel default(shared)
      {
        #ifdef _OPENMP
         std::seed_seq seed_s{static_cast<int>(rd()), omp_get_thread_num()};
        #else
         std::seed_seq seed_s{static_cast<int>(rd()), 0};
        #endif
        std::mt19937 generator(seed_s);
        std::uniform_real_distribution<T> uniform(static_cast<T>(0), static_cast<T>(1));
        #pragma omp for
        for (std::size_t i = 0; i < size; ++i) {
          first[i] = uniform(generator);
        }
      }
    }
    
#ifdef __CUDACC__
    
#ifdef _THRUST    
    template<std::floating_point T>
    struct GenRand {
      
      unsigned int seed;
      T a, b;
      
      __host__ __device__
      GenRand(unsigned int s_, T a_ = static_cast<T>(0), T b_ = static_cast<T>(1)) : seed{s_}, a{a_}, b{b_} {};
      
      __device__
      T operator () (auto idx) {
        thrust::default_random_engine rng{seed};
        thrust::uniform_real_distribution<float> dist(a, b);
        rng.discard(idx);
        return dist(rng);
      }
    };
    
    template<std::floating_point T>
    void fill_rand(Thrust, T* first, T* last) {
      std::cout << "DEBUG THRUST\n";
      const auto size {static_cast<int>(last - first)};
      thrust::transform(thrust::device, thrust::make_counting_iterator(0), thrust::make_counting_iterator(size), first, GenRand<T>(std::random_device{}()));
    }
    
#endif
       
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
    void fill_rand(Cuda, T* first, T* last, cudaStream_t stream) {
      std::cout << "DEBUG CUDA\n";    
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
      fill_rand_kernel<value_type><<<dimGrid, dimBlock, 0, stream>>>(d_state, first, size);
      checkCudaErrors(cudaFree(d_state));
    }
    
#endif
    
  }  
    
}    
     
    
