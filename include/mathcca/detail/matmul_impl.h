#ifndef MATMUL_IMPL_H_
#define MATMUL_IMPL_H_
#pragma once
    
#include <concepts> // std::floating_point

#include <mathcca/host_matrix.h>
    
#ifdef __CUDACC__
#include <cuda_runtime.h> // cudaStream_t
#include <mathcca/device_matrix.h>
#endif

#include <mathcca/implementation_policy.h>

namespace mathcca {

  namespace detail {
    
    template<std::floating_point T>
    constexpr void matmul(MM::Base, const host_matrix<T>& A, const host_matrix<T>& B, host_matrix<T>& C) ;
    
    template<std::floating_point T, unsigned int LINEAR_TILE_DIM>
    constexpr void matmul(MM::Tiled, const host_matrix<T>& A, const host_matrix<T>& B, host_matrix<T>& C) ;
    
#ifdef _MKL
    template<std::floating_point T>
    constexpr void matmul(MM::Mkl, const host_matrix<T>& A, const host_matrix<T>& B, host_matrix<T>& C) ;
#endif 
     
#ifdef __CUDACC__ 
      
    template <std::floating_point T, unsigned int LINEAR_THREAD_BLOCK_DIM>
    void matmul(MM::Base, const device_matrix<T>& A, const device_matrix<T>& B, device_matrix<T>& C, cudaStream_t stream) ;     
    
    template <std::floating_point T, unsigned int LINEAR_THREAD_BLOCK_DIM>
    void matmul(MM::Tiled, const device_matrix<T>& A, const device_matrix<T>& B, device_matrix<T>& C, cudaStream_t stream);     
    
#ifdef _CUBLAS
    template <std::floating_point T>
    void  matmul(MM::Cublas, const device_matrix<T>& A, const device_matrix<T>& B, device_matrix<T>& C) ; 
#endif
    
#endif
    
 }   
      
}   
    
#include <mathcca/detail/matmul_impl.inl>
    
#endif


