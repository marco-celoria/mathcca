#ifndef FILL_CONST_H_
#define FILL_CONST_H_
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
    void fill_const(mathcca::iterator::device_iterator<T> first, mathcca::iterator::device_iterator<T> last, const T v, cudaStream_t stream= 0);
#endif
    template<std::floating_point T>
    void fill_const(mathcca::iterator::host_iterator<T> first, mathcca::iterator::host_iterator<T> last, const T v);
  }

}

#ifdef __CUDACC__
#include <mathcca/detail/device_fill_const.inl>
#endif

#include <mathcca/detail/host_fill_const.inl>

#endif


