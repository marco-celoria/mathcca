#ifndef FILL_CONST_H_
#define FILL_CONST_H_
#pragma once

#include <concepts> // std::floating_point
#include <type_traits> // std::is_same

// StdPar() Omp() Thrust() Cuda()
#include <mathcca/execution_policy.h>

#include <mathcca/host_iterator.h> // mathcca::host_iterator_tag()

#ifdef __CUDACC__
 #include <mathcca/device_matrix.h> // mathcca::device_iterator_tag()
 #include <cuda_runtime.h> // cudaStream_t
#endif

#include <mathcca/detail/fill_const_impl.h>

namespace mathcca {
     
  class host_iterator_tag;
  class Omp;
       
#ifdef __CUDACC__
     
  class device_iterator_tag;
  class Cuda;
    
#endif
     
#ifdef __CUDACC__
      
  template<typename Iter , std::floating_point T, unsigned int THREAD_BLOCK_DIM= 128>
  void fill_const(Iter first, Iter last, const T v, cudaStream_t stream= 0) {
    if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::host_iterator_tag()>) {
#ifdef _STDPAR
      detail::fill_const(StdPar(), first.get(), last.get(), v);
#else     
      detail::fill_const(Omp(), first.get(), last.get(), v);
#endif  
    }  
    if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::device_iterator_tag()>) {
#ifdef _THRUST
      detail::fill_const(Thrust(), first.get(), last.get(), v);
#else    
      detail::fill_const<T, THREAD_BLOCK_DIM>(Cuda(), first.get(), last.get(), v, stream);
#endif
    }
  }  
     
#else
     
  template<typename Iter, std::floating_point T>
  void fill_const(Iter first, Iter last, const T v){
    if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::host_iterator_tag()>) {
#ifdef _STDPAR
      detail::fill_const(StdPar(), first.get(), last.get(), v);
#else
      detail::fill_const(Omp(), first.get(), last.get(), v);
#endif
    }  
  }   
    
#endif
    
}   

#endif


