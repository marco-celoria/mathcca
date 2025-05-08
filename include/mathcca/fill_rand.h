#ifndef FILL_RAND_H_
#define FILL_RAND_H_
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
    template<typename Iter, unsigned int THREAD_BLOCK_DIM= 128>
    void fill_rand(Iter first, Iter last, cudaStream_t stream= 0) {
      if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::host_iterator_tag()>){
        fill_rand(Omp(), first.get(), last.get());
      }
      if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::device_iterator_tag()>){
	using T= typename Iter::value_type;
        fill_rand<T, THREAD_BLOCK_DIM>(Cuda(), first.get(), last.get(), stream);
      }
    }
#else
    template<typename Iter>
    void fill_rand(Iter first, Iter last){
      if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::host_iterator_tag()> ){
        fill_rand(Omp(), first.get(), last.get());
      }
    }
#endif

}

#ifdef __CUDACC__
#include <mathcca/detail/device_fill_rand.inl>
#endif

#include <mathcca/detail/host_fill_rand.inl>

#endif

