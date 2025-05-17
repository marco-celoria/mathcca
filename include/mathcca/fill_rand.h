#ifndef FILL_RAND_H_
#define FILL_RAND_H_
#pragma once

#include <type_traits> // std::is_same

// StdPar() Omp() Thrust() Cuda()
#include <mathcca/common_algorithm.h>

#include <mathcca/host_iterator.h> // mathcca::host_iterator_tag()

#ifdef __CUDACC__
 #include <mathcca/device_matrix.h> // mathcca::device_iterator_tag()
 #include <cuda_runtime.h> // cudaStream_t
#endif

#include <mathcca/detail/fill_rand_impl.h>

namespace mathcca {
       
  class host_iterator_tag;
     
  class Omp;
  class StdPar;
     
#ifdef __CUDACC__
      
  class device_iterator_tag;
     
  class Cuda;
  class Thrust;
     
#endif
     
#ifdef __CUDACC__
      
  template<typename Iter, unsigned int THREAD_BLOCK_DIM= 128>
  void fill_rand(Iter first, Iter last, cudaStream_t stream= 0) {
    if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::host_iterator_tag()>) {
#ifdef _STDPAR
      detail::fill_rand(StdPar(), first.get(), last.get());
#else    
      detail::fill_rand(Omp(), first.get(), last.get());
#endif  
    }   
    if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::device_iterator_tag()>) {
#ifdef _THRUST
      detail::fill_rand(Thrust(), first.get(), last.get());
#else     
      using T= typename Iter::value_type;
      detail::fill_rand<T, THREAD_BLOCK_DIM>(Cuda(), first.get(), last.get(), stream);
#endif  
    }  
  }     
          
#else   
           
  template<typename Iter>
  void fill_rand(Iter first, Iter last){
    if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::host_iterator_tag()>) {
#ifdef _STDPAR
      detail::fill_rand(StdPar(), first.get(), last.get());
#else       
      detail::fill_rand(Omp(), first.get(), last.get());
#endif
    }
  }

#endif

}


#endif


