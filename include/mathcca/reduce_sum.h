/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef REDUCE_SUM_H_
#define REDUCE_SUM_H_
#pragma once

#include <concepts>    // std::floating_point
#include <type_traits> // std::is_same

// StdPar() Omp() Thrust() Cuda()
#include <mathcca/execution_policy.h>

#include <mathcca/host_iterator.h> // mathcca::host_iterator_tag()

#ifdef __CUDACC__
 #include <mathcca/device_matrix.h> // mathcca::device_iterator_tag()
 #include <cuda_runtime.h> // cudaStream_t
#endif

#include <mathcca/detail/reduce_sum_impl.h>

namespace mathcca {
     
  class host_iterator_tag;
  class Omp;
      
#ifdef __CUDACC__
      
  class device_iterator_tag;
  class Cuda;
      
#endif
        
#ifdef __CUDACC__
       
  template<typename Iter , std::floating_point T, unsigned int THREAD_BLOCK_DIM= 128>
  T reduce_sum(Iter first, Iter last, const T init, cudaStream_t stream= 0) {
    if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::host_iterator_tag()>) {
#ifdef _STDPAR
      return detail::reduce_sum(StdPar(), first.get(), last.get(), init);
#else       
      return detail::reduce_sum(Omp(), first.get(), last.get(), init);
#endif    
    }    
    if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::device_iterator_tag()>) {
#ifdef _THRUST
      return detail::reduce_sum(Thrust(), first.get(), last.get(), init);
#else     
      return detail::reduce_sum<T, THREAD_BLOCK_DIM>(Cuda(), first.get(), last.get(), init, stream);
#endif    
    }
  }
      
#else
         
  template<typename Iter, std::floating_point T>
  T reduce_sum(Iter first, Iter last, const T init) {
    if constexpr (std::is_same_v<typename Iter::iterator_system(), mathcca::host_iterator_tag()>) {
#ifdef _STDPAR
      return detail::reduce_sum(StdPar(), first.get(), last.get(), init);
#else
      return detail::reduce_sum(Omp(), first.get(), last.get(), init);
#endif
    }
  }

#endif

}

#endif


