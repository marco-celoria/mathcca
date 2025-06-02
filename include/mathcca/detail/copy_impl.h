/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef COPY_IMPL_H_
#define COPY_IMPL_H_
#pragma once

#include <concepts> // std::floating_point

// StdPar Omp Thrust Cuda CudaHtoDcpy CudaDtoHcpy
#include <mathcca/execution_policy.h> 

#ifdef __CUDACC__
#include <cuda_runtime.h> // cudaStream_t
#endif

namespace mathcca {
    
  namespace detail {
    
#ifdef _STDPAR
    template<std::floating_point T>
    void copy(StdPar, const T* s_first, const T* s_last, T* d_first); 
#endif
    
    template<std::floating_point T>
    void copy(Omp, const T* s_first, const T* s_last, T* d_first); 
    
#ifdef __CUDACC__
    
#ifdef _THRUST      
    template<std::floating_point T>
    void copy(Thrust, const T* s_first, const T* s_last, T* d_first);
#endif
    
    template<std::floating_point T>
    void copy(Cuda, const T* s_first, const T* s_last, T* d_first, cudaStream_t stream=0);
    
    template<std::floating_point T>
    void copy(CudaDtoHcpy, const T* s_first, const T* s_last, T* d_first, cudaStream_t stream=0); 
    
    template<std::floating_point T>
    void copy(CudaHtoDcpy, const T* s_first, const T* s_last, T* d_first, cudaStream_t stream=0);
    
#endif
    
  }  
    
}    
    

#include <mathcca/detail/copy_impl.inl>

#endif

