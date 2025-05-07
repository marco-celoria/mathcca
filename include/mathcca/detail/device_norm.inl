#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <mathcca/transform_reduce_sum.h>
#include <mathcca/device_helper.h>

#ifdef _CUBLAS
 #include <cublas_v2.h>
#endif

namespace mathcca {
  
  namespace matricca {

#ifdef _CUBLAS
    template<std::floating_point T>
    constexpr decltype(auto) frobenius_norm_Cublas(const device_matrix<T>& x) {
      std::cout << "Cublas frobenius norm\n";
      T result;
      const auto size{x.size()};
      const auto incx{1};
      cublasHandle_t handle;
      checkCudaErrors(cublasCreate(&handle));
      if constexpr(std::is_same_v<T, double>) {
        checkCudaErrors(cublasDnrm2(handle, size, x.data(), incx, &result));
      }
      else {
        checkCudaErrors(cublasSnrm2(handle, size, x.data(), incx, &result));
      }
      checkCudaErrors(cublasDestroy(handle));
      return result;
    }
#endif
    
    template<typename T>
    struct Square {
      __host__ __device__ T operator()(const T &x) const {
        return x * x;
      }
    };
    
    template<std::floating_point T>
    constexpr decltype(auto) frobenius_norm_Thrust(const device_matrix<T>& x) {
      std::cout << "Thrust frobenius norm\n";
      //thrust::device_ptr<const T> begin = thrust::device_pointer_cast(x.begin());
      //thrust::device_ptr<const T> end = thrust::device_pointer_cast(x.end());
      //thrust::device_vector<T> d_x(begin, end);
      const T res= thrust::transform_reduce(thrust::device, x.cbegin().get(), x.cend().get(), Square<T>(), static_cast<T>(0), thrust::plus<T>());
      //const T res= thrust::transform_reduce(d_x.begin(), d_x.end(), square<T>(), static_cast<T>(0), thrust::plus<T>());
      return std::sqrt(res);
    }
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    constexpr decltype(auto) frobenius_norm_Base(const device_matrix<T>& x, cudaStream_t stream) {
      std::cout << "Base frobenius norm\n";
      static_assert(THREAD_BLOCK_DIM <= 1024);
      //auto y{x*x};
      //return std::sqrt(algocca::device::reduce_sum<T, THREAD_BLOCK_DIM>(y.cbegin(), y.cend(), static_cast<T>(0), stream));
      return std::sqrt(algocca::transform_reduce_sum<T, Square<T>, THREAD_BLOCK_DIM>( x.cbegin(), x.cend(), Square<T>(), static_cast<T>(0), stream));
  }
  }
}


