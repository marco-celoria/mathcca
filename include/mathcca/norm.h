/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef NORM_H_
#define NORM_H_
#pragma once

#include <mathcca/host_matrix.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <mathcca/device_matrix.h>
#endif

#include <mathcca/detail/norm_impl.h>

namespace mathcca {
    
#ifdef __CUDACC__     
  template<typename Matrix, typename Implementation, unsigned int THREAD_BLOCK_DIM= 128>
  auto frobenius_norm (const Matrix& A, Implementation, cudaStream_t stream= 0) {
#else
  template<typename Matrix, typename Implementation>
  auto frobenius_norm (const Matrix& A, Implementation) {	  
#endif
    static_assert(std::is_same_v<Implementation, Norm::Base>  
#ifdef _MKL       
                  || std::is_same_v<Implementation, Norm::Mkl>
#endif
#ifdef _CUBLAS       
                  || std::is_same_v<Implementation, Norm::Cublas>
#endif
                 );
    using value_type= Matrix::value_type;
    if constexpr(std::is_same_v< typename decltype(A.get_allocator())::execution, Omp> && std::is_same_v< Implementation, Norm::Base>) {
      return detail::frobenius_norm<value_type>(Omp(), A.cbegin().get(), A.cend().get(), Norm::Base());
    }
#ifdef _MKL
    else if constexpr(std::is_same_v< typename decltype(A.get_allocator())::execution, Omp> && std::is_same_v< Implementation, Norm::Mkl>) {
      return detail::frobenius_norm<value_type>(Omp(), A.cbegin().get(), A.cend().get(), Norm::Mkl());
    }
#endif
#ifdef __CUDACC__    
    else if constexpr(std::is_same_v< typename decltype(A.get_allocator())::execution, Cuda> && std::is_same_v< Implementation, Norm::Base>) {
      return detail::frobenius_norm<value_type, THREAD_BLOCK_DIM>(Cuda(), A.cbegin().get(), A.cend().get(), Norm::Base(), stream);
    }
#ifdef _CUBLAS
    else if constexpr(std::is_same_v< typename decltype(A.get_allocator())::execution, Cuda> && std::is_same_v< Implementation, Norm::Cublas>) {
      return detail::frobenius_norm<value_type>(Cuda(), A.cbegin().get(), A.cend().get(), Norm::Cublas());
    }
#endif
#endif
  }

}

#endif


