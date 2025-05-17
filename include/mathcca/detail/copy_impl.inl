#include <cstddef>
//#include <mathcca/device_iterator.h>
//#include <mathcca/host_iterator.h>

#ifdef _PARALG
#include <execution>
#include <ranges>
#endif

#ifdef __CUDACC__
#include <mathcca/device_helper.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#endif

#include <cstddef>

namespace mathcca {
namespace detail {
#ifdef _PARALG
    template<std::floating_point T>
    void copy(StdPar, const T* s_first, const T* s_last, T* d_first) {
      //std::cout << "DEBUG _PARALG\n";
      std::copy(std::execution::par_unseq, s_first, s_last, d_first);
    }
#endif

    template<std::floating_point T>
    void copy(Omp, const T* s_first, const T* s_last, T* d_first) {
      //std::cout << "DEBUG NO _PARALG\n";
      const auto size {static_cast<std::size_t>(s_last - s_first)};
      #pragma omp parallel for default(shared)
      for (std::size_t i= 0; i < size; ++i) {
        d_first[i]= s_first[i];
      }
    }
}
}

#ifdef __CUDACC__
namespace mathcca {
namespace detail {
    template<std::floating_point T>
    void copy(Thrust, const T* s_first, const T* s_last, T* d_first) {
      thrust::copy(thrust::device, s_first, s_last, d_first);
    };


    template<std::floating_point T>
    void copy(Cuda, const T* s_first, const T* s_last, T* d_first, cudaStream_t stream) {
      const auto size{static_cast<std::size_t>(s_last - s_first)};
      const std::size_t nbytes{size * sizeof(T)};
      checkCudaErrors(cudaMemcpy(d_first, s_first, nbytes, cudaMemcpyDeviceToDevice));
    };

    template<std::floating_point T>
    void copy(CudaHtoHcpy, const T* s_first, const T* s_last, T* d_first, cudaStream_t stream) {
      const auto size{static_cast<std::size_t>(s_last - s_first)};
      const auto bytes{size * sizeof(T)};
      checkCudaErrors(cudaMemcpyAsync(d_first, s_first, bytes, cudaMemcpyHostToHost, stream));
    }

    template<std::floating_point T>
    void copy(CudaDtoHcpy, const T* s_first, const T* s_last, T* d_first, cudaStream_t stream) {
      const auto size{static_cast<std::size_t>(s_last - s_first)};
      const auto bytes{size * sizeof(T)};
      checkCudaErrors(cudaMemcpyAsync(d_first, s_first, bytes, cudaMemcpyDeviceToHost, stream));
    }


    template<std::floating_point T>
    void copy(CudaHtoDcpy, const T* s_first, const T* s_last, T* d_first, cudaStream_t stream) {
      const auto size{static_cast<std::size_t>(s_last - s_first)};
      const auto bytes{size * sizeof(T)};
      checkCudaErrors(cudaMemcpyAsync(d_first, s_first, bytes, cudaMemcpyHostToDevice, stream));
    }
}

}
#endif

