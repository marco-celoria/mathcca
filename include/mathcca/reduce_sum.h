#ifndef REDUCE_SUM_H_
#define REDUCE_SUM_H_
#pragma once

#include <concepts>
#include <mathcca/common_algorithm.h>
#include <mathcca/host_iterator.h>
#ifdef __CUDACC__
#include <mathcca/device_iterator.h>
#include <cuda_runtime.h>
#endif

namespace mathcca {

  namespace algocca {

#ifdef __CUDACC__
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM= 128>
    T reduce_sum(mathcca::iterator::device_iterator<const T> first, mathcca::iterator::device_iterator<const T> last, const T init, cudaStream_t stream= 0);
#else
    template<std::floating_point T>
    T reduce_sum(mathcca::iterator::host_iterator<const T> first, mathcca::iterator::host_iterator<const T> last, const T init);
#endif
  }

}


#ifdef __CUDACC__
#include <mathcca/detail/device_reduce_sum.inl>
#endif

#include <mathcca/detail/host_reduce_sum.inl>

#endif

