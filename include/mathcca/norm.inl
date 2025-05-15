#ifdef __CUDACC__
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <mathcca/transform_reduce_sum.h>
#include <mathcca/device_helper.h>
#endif
#ifdef _CUBLAS
 #include <cublas_v2.h>
#endif

namespace mathcca {


          #ifdef _STDPAR

  template<typename T>
  struct Square {
    T operator()(const T &x) const {
      return x * x;
    }
  };


  template<std::floating_point T>
  constexpr decltype(auto) frobenius_norm_Pstl(const host_matrix<T>& x) {
    std::cout << "Parallel STL frobenius norm\n";
    return std::sqrt(std::transform_reduce(std::execution::par, x.cbegin(), x.cend(), static_cast<T>(0), std::plus<T>(), Square<T>()));
  }
#endif


#ifdef _MKL
  template<std::floating_point T>
  constexpr decltype(auto) frobenius_norm_Mkl(const host_matrix<T>& x) {
    std::cout << "Mkl frobenius norm\n";
    T result;
    const auto size{x.size()};
    const auto incx{1};
    if constexpr(std::is_same_v<T, double>) {
    //double cblas_dnrm2 (const MKL_INT n, const double *x, const MKL_INT incx);
    result= cblas_dnrm2(size, x.data(), incx);
    }
    else {
    //float cblas_snrm2 (const MKL_INT n, const float *x, const MKL_INT incx);
    result= cblas_snrm2(size, x.data(), incx);
    }

    return result;
  }
#endif

    template<std::floating_point T>
  constexpr decltype(auto) frobenius_norm_Base (const host_matrix<T>& A) {
    std::cout << "Base frobenius norm\n";
    using size_type= typename host_matrix<T>::size_type;
    //using value_type= T;
    const auto size{A.size()};
    auto sum{static_cast<T>(0)};
    #pragma omp parallel for reduction(+:sum) default(shared)
    for (size_type i= 0; i < size; ++i) {
        sum+= A[i] * A[i];
    }
    return std::sqrt(sum);
  }

}


#ifdef __CUDACC__
namespace mathcca {
  
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
   /* 
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
    */
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    constexpr decltype(auto) frobenius_norm_Base(const device_matrix<T>& x, cudaStream_t stream) {
      std::cout << "Base frobenius norm\n";
      static_assert(THREAD_BLOCK_DIM <= 1024);
      //auto y{x*x};
      using Iter= device_matrix<T>::const_iterator;//decltype(x.cbegin());
      //return std::sqrt(device::reduce_sum<T, THREAD_BLOCK_DIM>(y.cbegin(), y.cend(), static_cast<T>(0), stream));
      return std::sqrt(transform_reduce_sum<Iter, T, Square<T>, THREAD_BLOCK_DIM>( x.cbegin(), x.cend(), Square<T>(), static_cast<T>(0), stream));
  }
}

#endif
