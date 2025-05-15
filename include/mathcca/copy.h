#ifndef COPY_H_
#define COPY_H_
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
     class StdPar;

#ifdef __CUDACC__

     class device_iterator_tag;
     class CudaDtoDcpy;
     class CudaHtoDcpy;
     class CudaDtoHcpy;
     class CudaHtoHcpy;
#endif

#ifdef __CUDACC__
    template<typename Iter1, typename Iter2>
    void copy(Iter1 s_first, Iter1 s_last, Iter2 d_first, cudaStream_t stream=0) {
      if constexpr (std::is_same_v<typename Iter1::iterator_system(), mathcca::host_iterator_tag()> && std::is_same_v<typename Iter2::iterator_system(), mathcca::host_iterator_tag()>){
#ifdef _PARALG
      	      copy(StdPar(), s_first.get(), s_last.get(), d_first.get());
#else
      	      copy(CudaHtoHcpy(), s_first.get(), s_last.get(), d_first.get());
#endif
      }
      if constexpr (std::is_same_v<typename Iter1::iterator_system(), mathcca::device_iterator_tag()> && std::is_same_v<typename Iter2::iterator_system(), mathcca::device_iterator_tag()>) {
#ifdef _PARALG
        copy(Thrust(), s_first.get(), s_last.get(), d_first.get());
#else
        copy(Cuda(), s_first.get(), s_last.get(), d_first.get(), stream);
#endif
      } 
      else if constexpr (std::is_same_v<typename Iter1::iterator_system(), mathcca::host_iterator_tag()> && std::is_same_v<typename Iter2::iterator_system(), mathcca::device_iterator_tag()>) {
        copy(CudaHtoDcpy(), s_first.get(), s_last.get(), d_first.get(), stream);
      }
      else if constexpr (std::is_same_v<typename Iter1::iterator_system(), mathcca::device_iterator_tag()> && std::is_same_v<typename Iter2::iterator_system(), mathcca::host_iterator_tag()>) {
        copy(CudaDtoHcpy(), s_first.get(), s_last.get(), d_first.get(), stream);
      }
    }
    
#else
    template<typename Iter1, typename Iter2>
    void copy(Iter1 s_first, Iter1 s_last, Iter2 d_first){
      if constexpr (std::is_same_v<typename Iter1::iterator_system(), mathcca::host_iterator_tag()> && std::is_same_v<typename Iter2::iterator_system(), mathcca::host_iterator_tag()>){
#ifdef _PARALG
              copy(StdPar(), s_first.get(), s_last.get(), d_first.get());
#else
              copy(Omp(), s_first.get(), s_last.get(), d_first.get());
#endif
      }
   }
#endif

}

//#ifdef __CUDACC__
//#include <mathcca/detail/device_copy.inl>
//#endif

#include <mathcca/detail/copy.inl>

#endif

