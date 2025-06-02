/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef REDUCE_SUM_IMPL_H_
#define REDUCE_SUM_IMPL_H_
#pragma once

#include <concepts> // std::floating_point

// StdPar Omp Thrust Cuda
#include <mathcca/execution_policy.h>

#ifdef __CUDACC__
#include <cuda_runtime.h> // cudaStream_t
#endif
    
namespace mathcca {
    
  namespace detail {
    
#ifdef _STDPAR
    
    template<std::floating_point T>
    T reduce_sum(StdPar, const T* first, const T* last, const T init); 
    
#endif
    
    template<std::floating_point T>
    T reduce_sum(Omp, const T* first, const T* last, const T init); 
    
#ifdef __CUDACC__
    
#ifdef _THRUST
    
    template<std::floating_point T>
    T reduce_sum (Thrust, const T* first, const T* last, const T init);
    
#endif
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM= 128>
    T reduce_sum(Cuda, const T* first, const T* last, const T init, cudaStream_t stream=0); 
    
#endif
    
  } 
    
}    
    
#include <mathcca/detail/reduce_sum_impl.inl>
    
#endif


