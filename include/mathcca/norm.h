#ifndef NORM_H_
#define NORM_H_
#pragma once

#include <concepts>
#include <mathcca/host_matrix.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <mathcca/device_matrix.h>
#endif


namespace mathcca {

#ifdef _MKL
    template<std::floating_point T>
    constexpr decltype(auto) frobenius_norm_Mkl(const host_matrix<T>& x);
#endif

    template<std::floating_point T>
    constexpr decltype(auto) frobenius_norm_Base(const host_matrix<T>& x);

    enum class HostFN {
      Base
#ifdef _MKL
      , Mkl
#endif
     };

    template<std::floating_point T, HostFN O>
    T frobenius_norm (const host_matrix<T>& x) {
      static_assert(O == HostFN::Base
#ifdef _MKL
                      || O == HostFN::Mkl
#endif
                   );
      if constexpr(O == HostFN::Base) {
        return frobenius_norm_Base<T>(x);
      }
#ifdef _MKL
      else if constexpr(O == HostFN::Mkl) {
        return frobenius_norm_Mkl(x);
      }
#endif
    }

}


#ifdef __CUDACC__
namespace mathcca {

    enum class DevFN {
      Base
#ifdef _CUBLAS
      , Cublas
#endif
    };

#ifdef _CUBLAS
    template<std::floating_point T>
    constexpr decltype(auto) frobenius_norm_Cublas(const device_matrix<T>& x);
#endif

    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM= 128>
    constexpr decltype(auto) frobenius_norm_Base(const device_matrix<T>& x, cudaStream_t stream= 0);

    template<std::floating_point T, DevFN O, unsigned int THREAD_BLOCK_DIM= 128>
    constexpr decltype(auto) frobenius_norm (const device_matrix<T>& x, cudaStream_t stream= 0) {
      static_assert(O == DevFN::Base
#ifdef _CUBLAS
                   || O == DevFN::Cublas
#endif
                   );
      if constexpr(O == DevFN::Base) {
        return frobenius_norm_Base<T, THREAD_BLOCK_DIM>(x, stream);
      }
#ifdef _CUBLAS
      else if constexpr(O == DevFN::Cublas) {
        return frobenius_norm_Cublas<T>(x);
      }
#endif
    }

}
#endif

#include <mathcca/norm.inl>

//#ifdef __CUDACC__
//#include <mathcca/detail/device_norm.inl>
//#endif

#endif

