#include <cstddef>
#include <mathcca/device_helper.h>
//#include <mathcca/device_iterator.h>
//#include <mathcca/host_iterator.h>

namespace mathcca {
       
     
    template<std::floating_point T>
    void copy(CudaDtoDcpy, const T* s_first, const T* s_last, T* d_first, cudaStream_t stream) {
      const auto size{static_cast<std::size_t>(s_last - s_first)};
      const std::size_t nbytes{size * sizeof(T)};
      checkCudaErrors(cudaMemcpy(d_first, s_first, nbytes, cudaMemcpyDeviceToDevice));
    };
     
      
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

