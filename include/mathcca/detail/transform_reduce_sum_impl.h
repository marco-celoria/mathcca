/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef TRANSFORM_REDUCE_SUM_IMPL_H_
#define TRANSFORM_REDUCE_SUM_IMPL_H_
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
    
    template<std::floating_point T, typename UnaryFunction>
    T transform_reduce_sum(StdPar, const T* first, const T* last, UnaryFunction unary_op, const T init);
    
#endif
    
    template<std::floating_point T, typename UnaryFunction>
    T transform_reduce_sum(Omp, const T* first, const T* last, UnaryFunction unary_op, const T init); 
    
#ifdef __CUDACC__
    
#ifdef _THRUST
    
    template<std::floating_point T, typename UnaryFunction>
    T transform_reduce_sum (Thrust, const T* first, const T* last, UnaryFunction unary_op, const T init);
    
#endif
    
    template<std::floating_point T, typename UnaryFunction, unsigned int THREAD_BLOCK_DIM>
    T transform_reduce_sum(Cuda, const T* first, const T* last, UnaryFunction unary_op, const T init, cudaStream_t stream= 0); 
    
#endif    
    
  }  
      
}   

#include<mathcca/detail/transform_reduce_sum_impl.inl>

#endif


