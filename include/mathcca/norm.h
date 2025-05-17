#ifndef NORM_H_
#define NORM_H_
#pragma once

#include <concepts> // std::floating_point

#include <mathcca/host_matrix.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <mathcca/device_matrix.h>
#endif

#include <mathcca/detail/norm_impl.h>

namespace mathcca {
    
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
      return detail::frobenius_norm_Base<T>(x);
    }
#ifdef _MKL
    else if constexpr(O == HostFN::Mkl) {
      return detail::frobenius_norm_Mkl(x);
    }
#endif
  }
    
#ifdef __CUDACC__
    
  enum class DevFN {
    Base
#ifdef _CUBLAS
    , Cublas
#endif
  };
     
  template<std::floating_point T, DevFN O, unsigned int THREAD_BLOCK_DIM= 128>
  constexpr decltype(auto) frobenius_norm (const device_matrix<T>& x, cudaStream_t stream= 0) {
    static_assert(O == DevFN::Base
#ifdef _CUBLAS     
                  || O == DevFN::Cublas
#endif             
                 );
    if constexpr(O == DevFN::Base) {
      return detail::frobenius_norm_Base<T, THREAD_BLOCK_DIM>(x, stream);
    }
#ifdef _CUBLAS
    else if constexpr(O == DevFN::Cublas) {
      return detail::frobenius_norm_Cublas<T>(x);
    }
#endif
  }

#endif

}

#endif


