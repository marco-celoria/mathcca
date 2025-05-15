#ifndef TRANSFORM_REDUCE_SUM_H_
#define TRANSFORM_REDUCE_SUM_H_
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
     class StdPar;
     class Omp;
#ifdef __CUDACC__
     class device_iterator_tag;
     class Cuda;
     class Thrust;
#endif
     
#ifdef __CUDACC__
    template<typename Iter, std::floating_point T, typename UnaryFunction, unsigned int THREAD_BLOCK_DIM= 128>
    T transform_reduce_sum(Iter first, Iter last, UnaryFunction unary_op, const T init, cudaStream_t stream= 0) {
      if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::host_iterator_tag()>){
#ifdef _PARALG
        return transform_reduce_sum(StdPar(), first.get(), last.get(), unary_op, init);
#else
        return transform_reduce_sum(Omp(), first.get(), last.get(), unary_op, init);
#endif
      }
      if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::device_iterator_tag()>){
#ifdef _PARALG
        return transform_reduce_sum(Thrust(), first.get(), last.get(), unary_op, init);
#else
        return transform_reduce_sum<T, UnaryFunction, THREAD_BLOCK_DIM>(Cuda(), first.get(), last.get(), unary_op, init, stream);
#endif
      }
    }
#else
    template<typename Iter, std::floating_point T, typename UnaryFunction>
    T transform_reduce_sum(Iter first, Iter last, UnaryFunction unary_op, const T init) {
      if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::host_iterator_tag()>){
#ifdef _PARALG
        return transform_reduce_sum(StdPar(), first.get(), last.get(), unary_op, init);
#else
        return transform_reduce_sum(Omp(), first.get(), last.get(), unary_op, init);
#endif
      }
    }
#endif
}


//#ifdef __CUDACC__
//#include <mathcca/detail/device_transform_reduce_sum.inl>
//#endif

#include <mathcca/transform_reduce_sum.inl>

#endif

