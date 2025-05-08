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
     class Omp;

#ifdef __CUDACC__
     class device_iterator_tag;
     class Cuda;
    template<typename Iter, std::floating_point T, typename UnaryFunction, unsigned int THREAD_BLOCK_DIM= 128>
    T transform_reduce_sum(Iter first, Iter last, UnaryFunction unary_op, const T init, cudaStream_t stream= 0) {
      if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::host_iterator_tag()>){
        return transform_reduce_sum(Omp(), first.get(), last.get(), unary_op, init);
      }
      if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::device_iterator_tag()>){
        return transform_reduce_sum<T, UnaryFunction, THREAD_BLOCK_DIM>(Cuda(), first.get(), last.get(), unary_op, init, stream);
      }
    }
#else
    template<typename Iter, std::floating_point T, typename UnaryFunction>
    T transform_reduce_sum(Iter first, Iter last, UnaryFunction unary_op, const T init) {
      if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::host_iterator_tag()>){
        return transform_reduce_sum(Omp(), first.get(), last.get(), unary_op, init);
      }
    }
#endif

}


#ifdef __CUDACC__
#include <mathcca/detail/device_transform_reduce_sum.inl>
#endif

#include <mathcca/detail/host_transform_reduce_sum.inl>

#endif

