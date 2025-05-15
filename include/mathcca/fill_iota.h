#ifndef FILL_IOTA_H_
#define FILL_IOTA_H_
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
    void fill_iota(Iter first, Iter last, const T v, cudaStream_t stream= 0) {
      if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::host_iterator_tag()>){
#ifdef _PARALG
        fill_iota(StdPar(), first.get(), last.get(), v);
#else
        fill_iota(Omp(), first.get(), last.get(), v);
#endif
      }
      if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::device_iterator_tag()>){
#ifdef _PARALG
        fill_iota(Thrust(), first.get(), last.get(), v);
#else
        fill_iota<T, THREAD_BLOCK_DIM>(Cuda(), first.get(), last.get(), v, stream);
#endif
      }
    }
#else
    template<typename Iter, std::floating_point T>
    void fill_iota(Iter first, Iter last, const T v){
      if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::host_iterator_tag()> ){
#ifdef _PARALG
        fill_iota(StdPar(), first.get(), last.get(), v);
#else
        fill_iota(Omp(), first.get(), last.get(), v);
#endif
      }
    }
#endif

}



//#ifdef __CUDACC__
//#include <mathcca/detail/device_fill_iota.inl>
//#endif

#include <mathcca/fill_iota.inl>

#endif

