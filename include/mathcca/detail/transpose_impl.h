#ifndef TRANSPOSE_IMPL_H_
#define TRANSPOSE_IMPL_H_
#pragma once


namespace mathcca {
namespace detail {

            template<std::floating_point T>
  constexpr void transpose_parallel_Base(const host_matrix<T>& A, host_matrix<T>& B);

  template<std::floating_point T, unsigned int LINEAR_TILE_DIM>
  constexpr void transpose_parallel_Tiled(const host_matrix<T>& A, host_matrix<T>& B);

  #ifdef _MKL
  template<std::floating_point T>
  constexpr void transpose_parallel_Mkl(const host_matrix<T>& A, host_matrix<T>& B); 
#endif
}
}

#ifdef __CUDACC__
namespace mathcca {
 namespace detail {

    
    template <std::floating_point T, unsigned int LINEAR_THREAD_BLOCK_DIM>
    void transpose_device_Base(const device_matrix<T>& A, device_matrix<T>& B, cudaStream_t stream);
    
    template <std::floating_point T, unsigned int LINEAR_THREAD_BLOCK_DIM>
    void transpose_device_Tiled(const device_matrix<T>& A, device_matrix<T>& B, cudaStream_t stream);     
#ifdef _CUBLAS
    template <std::floating_point T>
    void  transpose_device_Cublas(const device_matrix<T>& A, device_matrix<T>& B);
#endif
 }
}
#endif

#include<mathcca/detail/transpose_impl.inl>


#endif
