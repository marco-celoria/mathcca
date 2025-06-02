/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef COPY_H_
#define COPY_H_
#pragma once

#include <type_traits> // std::is_same_v

#include <mathcca/execution_policy.h>

#include <mathcca/host_iterator.h>

#ifdef __CUDACC__
#include <mathcca/device_iterator.h>
#include <cuda_runtime.h>
#endif

#include <mathcca/detail/copy_impl.h>

namespace mathcca {
    
  class host_iterator_tag;
    
  class Omp;
  class StdPar;
    
#ifdef __CUDACC__
    
  class device_iterator_tag;
    
  class Cuda;
  class CudaHtoDcpy;
  class CudaDtoHcpy;

#ifdef _THRUST
  class Thrust;
#endif

#endif
        
#ifdef __CUDACC__
        
  template<typename Iter1, typename Iter2>
  void copy(Iter1 s_first, Iter1 s_last, Iter2 d_first, cudaStream_t stream=0) {
    if constexpr (std::is_same_v<typename Iter1::iterator_system(), mathcca::host_iterator_tag()> && 
                  std::is_same_v<typename Iter2::iterator_system(), mathcca::host_iterator_tag()>) {
#ifdef _STDPAR
      detail::copy(StdPar(), s_first.get(), s_last.get(), d_first.get());
#else  
      detail::copy(Omp(), s_first.get(), s_last.get(), d_first.get());
#endif
    }  
    if constexpr (std::is_same_v<typename Iter1::iterator_system(), mathcca::device_iterator_tag()> && 
                  std::is_same_v<typename Iter2::iterator_system(), mathcca::device_iterator_tag()>) {
#ifdef _THRUST
      detail::copy(Thrust(), s_first.get(), s_last.get(), d_first.get());
#else  
      detail::copy(Cuda(), s_first.get(), s_last.get(), d_first.get(), stream);
#endif
    }  
    else if constexpr (std::is_same_v<typename Iter1::iterator_system(), mathcca::host_iterator_tag()> && 
                       std::is_same_v<typename Iter2::iterator_system(), mathcca::device_iterator_tag()>) {
      detail::copy(CudaHtoDcpy(), s_first.get(), s_last.get(), d_first.get(), stream);
    }  
    else if constexpr (std::is_same_v<typename Iter1::iterator_system(), mathcca::device_iterator_tag()> && 
                       std::is_same_v<typename Iter2::iterator_system(), mathcca::host_iterator_tag()>) {
      detail::copy(CudaDtoHcpy(), s_first.get(), s_last.get(), d_first.get(), stream);
    }  
  }    
        
#else 
      
  template<typename Iter1, typename Iter2>
  void copy(Iter1 s_first, Iter1 s_last, Iter2 d_first) {
    if constexpr (std::is_same_v<typename Iter1::iterator_system(), mathcca::host_iterator_tag()> && 
                  std::is_same_v<typename Iter2::iterator_system(), mathcca::host_iterator_tag()>) {
#ifdef _STDPAR
      detail::copy(StdPar(), s_first.get(), s_last.get(), d_first.get());
#else  
      detail::copy(Omp(), s_first.get(), s_last.get(), d_first.get());
#endif 
    } 
  }    

#endif

}

#endif



