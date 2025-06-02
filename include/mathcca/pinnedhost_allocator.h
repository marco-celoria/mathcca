/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef PINNEDHOST_ALLOCATOR_H_
#define PINNEDHOST_ALLOCATOR_H_
#pragma once

#include <cstddef>  // std::size_t
#include <concepts> // std::floating_point

#include <cuda_runtime.h>
#include <mathcca/device_helper.h>

#include <mathcca/execution_policy.h>

namespace mathcca {
        
  template<std::floating_point T>
  class pinnedhost_allocator {
          
    public:
           
      using value_type= T;
      using reference= T&;
      using const_reference= const T&;
      using execution= Omp;

      pinnedhost_allocator()= default;
         
      pinnedhost_allocator(const pinnedhost_allocator& )= default;
         
      value_type* allocate(std::size_t size) {
        value_type* result{nullptr};
        const std::size_t nbytes{ size * sizeof(value_type) };
	checkCudaErrors(cudaHostAlloc((void **)& result, nbytes, cudaHostAllocDefault));
        return result;
      }  
         
      void deallocate(value_type* ptr, size_t) {
        checkCudaErrors(cudaFreeHost(ptr));
      }  
  };      

  template<std::floating_point T1, std::floating_point T2>
  bool operator==(const pinnedhost_allocator<T1>& lhs, const pinnedhost_allocator<T2>& rhs) {
    return true;
  }

  template<std::floating_point T1, std::floating_point T2>
  bool operator!=(const pinnedhost_allocator<T1>& lhs, const pinnedhost_allocator<T2>& rhs) {
    return !(lhs == rhs);
  }

}


#endif



