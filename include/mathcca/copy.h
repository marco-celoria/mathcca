#ifndef COPY_H_
#define COPY_H_
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
    template<std::floating_point T>
    void copy(mathcca::iterator::device_iterator<const T> first, mathcca::iterator::device_iterator<const T> last, mathcca::iterator::device_iterator<T> d_first, cudaStream_t stream= 0);
    
    template<std::floating_point T>
    void copy(mathcca::iterator::device_iterator<const T> d_first, mathcca::iterator::device_iterator<const T> d_last, mathcca::iterator::host_iterator<T> h_first, cudaStream_t stream= 0);
    
    template<std::floating_point T>
    void copy(mathcca::iterator::host_iterator<const T> h_first, mathcca::iterator::host_iterator<const T> h_last, mathcca::iterator::device_iterator<T> d_first, cudaStream_t stream= 0);

#endif
    template<std::floating_point T>
    void copy(mathcca::iterator::host_iterator<const T> first, mathcca::iterator::host_iterator<const T> last, mathcca::iterator::host_iterator<T> h_first);
  }

}

#ifdef __CUDACC__
#include <mathcca/detail/device_copy.inl>
#endif

#include <mathcca/detail/host_copy.inl>

#endif

