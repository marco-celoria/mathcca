#include <cstddef>
#include <mathcca/device_helper.h>
#include <mathcca/device_iterator.h>
#include <mathcca/host_iterator.h>

namespace mathcca {
       
  namespace algocca {
     
    template<std::floating_point T>
    void copy(mathcca::iterator::device_iterator<const T> first, mathcca::iterator::device_iterator<const T> last, mathcca::iterator::device_iterator<T> d_first, cudaStream_t stream) {
      const auto size{static_cast<std::size_t>(last - first)};
      const std::size_t nbytes{size * sizeof(T)};
      checkCudaErrors(cudaMemcpy(d_first.get(), first.get(), nbytes, cudaMemcpyDeviceToDevice));
    };
     
      
    template<std::floating_point T>
    void copy(mathcca::iterator::device_iterator<const T> d_first, mathcca::iterator::device_iterator<const T> d_last, mathcca::iterator::host_iterator<T> h_first, cudaStream_t stream) {
      const auto size{static_cast<std::size_t>(d_last - d_first)};
      const auto bytes{size * sizeof(T)};
      checkCudaErrors(cudaMemcpyAsync(h_first.get(), d_first.get(), bytes, cudaMemcpyDeviceToHost, stream));
    }

    
    template<std::floating_point T>
    void copy(mathcca::iterator::host_iterator<const T> h_first, mathcca::iterator::host_iterator<const T> h_last, mathcca::iterator::device_iterator<T> d_first, cudaStream_t stream) {
      const auto size{static_cast<std::size_t>(h_last - h_first)};
      const auto bytes{size * sizeof(T)};
      checkCudaErrors(cudaMemcpyAsync(d_first.get(), h_first.get(), bytes, cudaMemcpyHostToDevice, stream));
    }

  }

}

