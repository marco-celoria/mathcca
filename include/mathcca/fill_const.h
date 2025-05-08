#ifndef FILL_CONST_H_
#define FILL_CONST_H_
#pragma once


#include <concepts>
#include <mathcca/common_algorithm.h>
#include <mathcca/host_matrix.h>
#ifdef __CUDACC__
#include <mathcca/device_matrix.h>
#include <cuda_runtime.h>
#endif

namespace mathcca {

     class host_iterator_tag;
     class Omp;
     
#ifdef __CUDACC__
     class device_iterator_tag;
     class Cuda;
    
     template<typename Iter , std::floating_point T, unsigned int THREAD_BLOCK_DIM= 128>
    void fill_const(Iter first, Iter last, const T v, cudaStream_t stream= 0) {
      if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::host_iterator_tag()>){
        fill_const(Omp(), first.get(), last.get(), v);
      }
      if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::device_iterator_tag()>){
        fill_const<T, THREAD_BLOCK_DIM>(Cuda(), first.get(), last.get(), v, stream);
      }
    }
#else
    template<typename Iter, std::floating_point T>
    void fill_const(Iter first, Iter last, const T v){
      if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::host_iterator_tag()> ){
        fill_const(Omp(), first.get(), last.get(), v);
      }
    }
#endif

}



#ifdef __CUDACC__
#include <mathcca/detail/device_fill_const.inl>
#endif

#include <mathcca/detail/host_fill_const.inl>

#endif


