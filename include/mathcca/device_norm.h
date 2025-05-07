#ifndef DEVICE_NORM_H_
#define DEVICE_NORM_H_
#pragma once

#include <concepts>
#include <cuda_runtime.h>
#include <mathcca/device_matrix.h>

namespace mathcca {

  namespace matricca {
    
    enum class DevFN {
      Base,
      Thrust
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
    
    template<std::floating_point T>
    constexpr decltype(auto) frobenius_norm_Thrust(const device_matrix<T>& x);
    
    template<std::floating_point T, DevFN O, unsigned int THREAD_BLOCK_DIM= 128>
    constexpr decltype(auto) frobenius_norm (const device_matrix<T>& x, cudaStream_t stream= 0) {
      static_assert(O == DevFN::Base || O == DevFN::Thrust
#ifdef _CUBLAS
                   || O == DevFN::Cublas
#endif
                   );
      if constexpr(O == DevFN::Thrust) {
        return frobenius_norm_Thrust<T>(x);
      }
#ifdef _CUBLAS
      else if constexpr(O == DevFN::Cublas) {
        return frobenius_norm_Cublas<T>(x);
      }
#endif
      else {
        return frobenius_norm_Base<T, THREAD_BLOCK_DIM>(x, stream);
      }
    }

  }

}

#include <mathcca/detail/device_norm.inl>

#endif

