#ifndef REDUCE_SUM_H_
#define REDUCE_SUM_H_
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
#endif

 
#ifdef __CUDACC__
     template<typename Iter , std::floating_point T, unsigned int THREAD_BLOCK_DIM= 128>
    T reduce_sum(Iter first, Iter last, const T init, cudaStream_t stream= 0) {
      if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::host_iterator_tag()>) {
#ifdef _PARALG
        return reduce_sum(StdPar(), first.get(), last.get(), init);
#else
        return reduce_sum(Omp(), first.get(), last.get(), init);
#endif
      }
      if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::device_iterator_tag()>){
#ifdef _PARALG
        return reduce_sum(Thrust(), first.get(), last.get(), init);
#else
        return reduce_sum<T, THREAD_BLOCK_DIM>(Cuda(), first.get(), last.get(), init, stream);
#endif
      }
    }
#else
    template<typename Iter, std::floating_point T>
    T reduce_sum(Iter first, Iter last, const T init) {
      if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::host_iterator_tag()>){
#ifdef _PARALG
        return reduce_sum(StdPar(), first.get(), last.get(), init);
#else
        return reduce_sum(Omp(), first.get(), last.get(), init);
#endif
      }
    }
#endif

}


//#ifdef __CUDACC__
//#include <mathcca/detail/device_reduce_sum.inl>
//#endif

#include <mathcca/reduce_sum.inl>

#endif

