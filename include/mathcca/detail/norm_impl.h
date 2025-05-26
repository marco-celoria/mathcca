#ifndef NORM_IMPL_H_
#define NORM_IMPL_H_
#pragma once

#include <concepts> // std::floating_point

#include <mathcca/host_matrix.h>

#include <mathcca/execution_policy.h>
#include <mathcca/implementation_policy.h>

#ifdef __CUDACC__
 #include <cuda_runtime.h>
 #include <mathcca/device_matrix.h>
#endif

namespace mathcca {

  namespace detail {
    
#ifdef _MKL
    
    template<std::floating_point T>
    constexpr T frobenius_norm(Omp, const T* begin, const T* end, Norm::Mkl) ;
    
#endif
    
    template<std::floating_point T>
    constexpr T frobenius_norm(Omp, const T* begin, const T* end, Norm::Base);

#ifdef _STDPAR
    template<std::floating_point T>
    constexpr T frobenius_norm(StdPar, const T* begin, const T* end, Norm::Base);
#endif

#ifdef __CUDACC__
    
#ifdef _CUBLAS
    
    template<std::floating_point T>
    constexpr T frobenius_norm(Cuda, const T* begin, const T* end, Norm::Cublas);
    
#endif


#ifdef _THRUST
    template<std::floating_point T>
    constexpr T frobenius_norm(Thrust, const T* begin, const T* end, Norm::Base);
#endif
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    constexpr T frobenius_norm(Cuda, const T* begin, const T* end, Norm::Base, cudaStream_t stream);
    
#endif
    
  } 
    
}    

#include <mathcca/detail/norm_impl.inl>

#endif


