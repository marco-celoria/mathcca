
#include <cstddef>   // std::size_t
#include <concepts>  // std::floating_point
#include <iostream>  // std::cout

#include <mathcca/transform_reduce_sum.h>

#include <mathcca/host_matrix.h>

#ifdef __CUDACC__
 #include <cuda_runtime.h>
 #include <mathcca/device_helper.h>
 #include <mathcca/device_matrix.h>
 #ifdef _CUBLAS
  #include <cublas_v2.h>
 #endif
#endif

#ifdef _OPENMP
 #include <omp.h>
#endif

namespace mathcca {
    
  namespace detail {
     
    template<typename T>
    struct Square {
#ifdef __CUDACC__ 
      __host__ __device__
#endif
      T operator()(const T &x) const {
        return x * x;
      }
    };
     
    template<std::floating_point T>
    constexpr decltype(auto) frobenius_norm_Base(const host_matrix<T>& x) {
      std::cout << "Base frobenius norm\n";
      return std::sqrt(transform_reduce_sum(x.cbegin(), x.cend(), Square<T>(), static_cast<T>(0)));
    }

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
    
#ifdef __CUDACC__
    
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
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    constexpr decltype(auto) frobenius_norm_Base(const device_matrix<T>& x, cudaStream_t stream) {
      std::cout << "Base frobenius norm\n";
      static_assert(THREAD_BLOCK_DIM <= 1024);
      using Iter= device_matrix<T>::const_iterator;
      return std::sqrt(transform_reduce_sum<Iter, T, Square<T>, THREAD_BLOCK_DIM>( x.cbegin(), x.cend(), Square<T>(), static_cast<T>(0), stream));
    }
    
#endif
    
  } 
    
}    


