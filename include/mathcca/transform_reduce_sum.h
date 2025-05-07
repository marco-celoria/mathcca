#ifndef TRANSFORM_REDUCE_SUM_H_
#define TRANSFORM_REDUCE_SUM_H_
#pragma once

#include <concepts>
#include <mathcca/common_algorithm.h>
#include <mathcca/host_iterator.h>
//#ifdef __CUDACC__
#include <mathcca/device_iterator.h>
#include <cuda_runtime.h>
//#endif

namespace mathcca {

  namespace algocca {

//#ifdef __CUDACC__
    template<std::floating_point T, typename UnaryFunction, unsigned int THREAD_BLOCK_DIM= 128>
    T transform_reduce_sum(mathcca::iterator::device_iterator<const T> first, mathcca::iterator::device_iterator<const T> last, UnaryFunction unary_op, const T init, cudaStream_t stream= 0);
//#endif
    template<std::floating_point T, typename UnaryFunction>
    T transform_reduce_sum(mathcca::iterator::host_iterator<const T> first, mathcca::iterator::host_iterator<const T> last, UnaryFunction unary_op, const T init);
  }

}


//#ifdef __CUDACC__
#include <mathcca/detail/device_transform_reduce_sum.inl>
//#endif

#include <mathcca/detail/host_transform_reduce_sum.inl>

#endif

