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
      
  template<std::floating_point T, typename Implementation>
  T frobenius_norm (const host_matrix<T>& x, Implementation) {
    static_assert(std::is_same_v<Implementation, Norm::Base> 
#ifdef _MKL       
                  || std::is_same_v<Implementation, Norm::Mkl>
#endif
                 );
    if constexpr(std::is_same_v< Implementation, Norm::Base>) {
      return detail::frobenius_norm<T>(Norm::Base(), x);
    }
#ifdef _MKL
    else if constexpr(std::is_same_v< Implementation, Norm::Mkl>) {
      return detail::frobenius_norm(Norm::Mkl(), x);
    }
#endif
  }
    
#ifdef __CUDACC__
     
  template<std::floating_point T, typename Implementation, unsigned int THREAD_BLOCK_DIM= 128>
  constexpr decltype(auto) frobenius_norm (const device_matrix<T>& x, Implementation, cudaStream_t stream= 0) {
    static_assert(std::is_same_v<Implementation, Norm::Base>  
#ifdef _CUBLAS       
                  || std::is_same_v<Implementation, Norm::Cublas>
#endif
                 );
    if constexpr(std::is_same_v< Implementation, Norm::Base>) {
      return detail::frobenius_norm<T, THREAD_BLOCK_DIM>(Norm::Base(), x, stream);
    }
#ifdef _CUBLAS
    else if constexpr(std::is_same_v< Implementation, Norm::Cublas>) {
      return detail::frobenius_norm<T>(Norm::Cublas(), x);
    }
#endif
  }

#endif

}

#endif


