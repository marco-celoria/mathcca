/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef MATMUL_IMPL_H_
#define MATMUL_IMPL_H_
#pragma once
    
#include <concepts> // std::floating_point

#include <mathcca/host_matrix.h>
    
#ifdef __CUDACC__
#include <cuda_runtime.h> // cudaStream_t
#include <mathcca/device_matrix.h>
#endif

#include <mathcca/implementation_policy.h>

namespace mathcca {

  namespace detail {
    
    template<std::floating_point T>
    constexpr void matmul(Omp, const std::size_t A_num_rows, const std::size_t B_num_cols, const std::size_t A_num_cols, const T* A, const T* B, T* C, MM::Base) ;
    
    template<std::floating_point T, unsigned int LINEAR_TILE_DIM>
    constexpr void matmul(Omp, const std::size_t A_num_rows, const std::size_t B_num_cols, const std::size_t A_num_cols, const T* A, const T* B, T* C, MM::Tiled) ;
    
#ifdef _MKL
    template<std::floating_point T>
    constexpr void matmul(Omp, const std::size_t A_num_rows, const std::size_t B_num_cols, const std::size_t A_num_cols, const T* A, const T* B, T* C, MM::Mkl) ;
#endif 
     
#ifdef __CUDACC__ 
      
    template <std::floating_point T, unsigned int LINEAR_THREAD_BLOCK_DIM>
    void matmul(Cuda, const std::size_t A_num_rows, const std::size_t B_num_cols, const std::size_t A_num_cols, const T* A, const T* B, T* C, MM::Base, cudaStream_t stream) ;     
    
    template <std::floating_point T, unsigned int LINEAR_THREAD_BLOCK_DIM>
    void matmul(Cuda, const std::size_t A_num_rows, const std::size_t B_num_cols, const std::size_t A_num_cols, const T* A, const T* B, T* C, MM::Tiled, cudaStream_t stream);     
    
#ifdef _CUBLAS
    template <std::floating_point T>
    void  matmul(Cuda, const std::size_t A_num_rows, const std::size_t B_num_cols, const std::size_t A_num_cols, const T* A, const T* B, T* C, MM::Cublas) ; 
#endif
    
#endif
    
 }   
      
}   
    
#include <mathcca/detail/matmul_impl.inl>
    
#endif


