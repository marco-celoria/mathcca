/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

// Expression template
// CudaSynch
// cudaMemcpyAsync
// PinnedMemory
// CHECK all the function with tests
// Move count_if_diff_to device_matrix.h 
// In device_matrix.inl put implementation of operator overloading
// Create device_matmul.h device_matmul.inl
// Create device_transpose.h device_transpose.inl
// Create device_norm.h device_norm.inl
// Same with host
// Separate algorithms similarly
#ifndef DEVICE_MATRIX_H_
#define DEVICE_MATRIX_H_
#pragma once

#include <cstddef>   // std::size_t
#include <concepts>  // std::floating_point
#include <iostream>  // std::cout
#include <memory>    // std::allocator_traits
#include <utility>   // std::forward
#include <stdexcept> // std::length_error

#include <cuda_runtime.h>

#include <mathcca/execution_policy.h> // Cuda
      
#include <mathcca/detail/base_matrix.h>

#include <mathcca/device_helper.h>
#include <mathcca/device_iterator.h>
#include <mathcca/device_allocator.h>


namespace mathcca {
    
  template<std::floating_point T, typename Allocator>
  class device_matrix;
  
  template<std::floating_point T>
  __global__ void addTo_kernel(T* __restrict accululator, const T* __restrict to_be_op, const std::size_t size);
    
  template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
  __global__ void subTo_kernel(T* __restrict accululator, const T* __restrict to_be_op, const std::size_t size);
    
  template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
  __global__ void mulScalarTo_kernel(T* __restrict accululator, const T to_be_op, const std::size_t size);
  
  template<std::floating_point T, typename A, typename E, unsigned int THREAD_BLOCK_DIM>
  __global__ void mulTo_kernel(device_matrix<T,A>& accululator, const device_matrix<T,A>& to_be_op);
   
  template<std::floating_point T, typename Allocator = device_allocator<T>>
  class device_matrix : public detail::base_matrix<T, Allocator> {
    
    using self= device_matrix;
    typedef detail::base_matrix<T,Allocator> Parent;
          
    public:
        
      typedef typename Parent::size_type  size_type;
      typedef typename Parent::value_type  value_type;
      typedef typename Parent::reference reference;
      typedef typename Parent::const_reference const_reference;
      typedef typename Parent::pointer pointer;
      typedef typename Parent::const_pointer const_pointer;
          
      using iterator= device_iterator<T, false>;
      using const_iterator= device_iterator<T, true>;
      using traits_alloc= std::allocator_traits<Allocator>;
         
      device_matrix(Allocator a) : Parent(std::forward<Allocator>(a)) {}
         
      constexpr device_matrix(size_type r, size_type c) : Parent(r,c) {}
         
      constexpr device_matrix(size_type r, size_type c, const_reference v) : Parent(r,c,v)  {}
             
      constexpr ~device_matrix() {}
          
      constexpr device_matrix(self&& m): Parent(std::forward<device_matrix<T>>(m)) {}
          
      constexpr device_matrix(const self& m) : Parent(m) {}
         
      constexpr device_matrix<T>& operator=(device_matrix&& rhs) {
        Parent::operator=(std::forward<device_matrix<T>>(rhs));
        return *this;
      }
         
      constexpr device_matrix<T>& operator=(const device_matrix& rhs) {
        Parent::operator=(rhs);
        return *this;
      }
           
      constexpr iterator begin() noexcept { return iterator{this->data()}; }
      constexpr iterator end()   noexcept { return iterator{this->data() + this->size()}; }
        
      constexpr const_iterator begin() const  noexcept { return const_iterator{const_cast<pointer>(this->data())}; }
      constexpr const_iterator end()   const  noexcept { return const_iterator{const_cast<pointer>(this->data() + this->size())}; }
      
      constexpr const_iterator cbegin() const noexcept { return const_iterator{const_cast<pointer>(this->data())}; }
      constexpr const_iterator cend()   const noexcept { return const_iterator{const_cast<pointer>(this->data() + this->size())}; }
      // Standard kernel	
      template<unsigned int THREAD_BLOCK_DIM= 128>
      self& operator+=(const self& rhs) {
        std::cout <<"operator+= lvalue\n";
        static_assert(THREAD_BLOCK_DIM <= 1024);
        if (!check_equal_size((*this), rhs))
          throw std::length_error{"Incompatible sizes for matrix-matrix addition"};
        const size_type size{this->size()};
        constexpr unsigned int threads{THREAD_BLOCK_DIM};
        const auto blocks{static_cast<unsigned int>((size + static_cast<size_type>(threads) - 1)/(static_cast<size_type>(threads)))};
        constexpr dim3 dimBlock(threads, 1, 1);
        dim3 dimGrid(blocks, 1, 1);
        addTo_kernel<value_type><<<dimGrid, dimBlock>>>(this->data(), rhs.data(), size);
        getLastCudaError("addTo_kernel() execution failed.\n");
        return *this;
      }
            
      // Increasing ILP kernel 
      template<unsigned int THREAD_BLOCK_DIM= 128>
      self& operator-=(const self& rhs) {
        std::cout <<"operator-= lvalue\n";
        static_assert(THREAD_BLOCK_DIM <= 1024);
        if (!check_equal_size((*this), rhs))
            throw std::length_error{"Incompatible sizes for matrix-matrix subtraction"};
          const size_type size{this->size()};
          constexpr unsigned int threads{THREAD_BLOCK_DIM};
          const auto blocks{static_cast<unsigned int>((size + 2 * static_cast<size_type>(threads) - 1) / (2 * static_cast<size_type>(threads)))};
          constexpr dim3 dimBlock(threads, 1, 1);
          dim3 dimGrid(blocks, 1, 1);
          subTo_kernel<value_type, threads><<<dimGrid, dimBlock>>>(this->data(), rhs.data(), size);
          getLastCudaError("subTo_kernel() execution failed.\n");
          return *this;
      }
      
      // Increasing ILP kernel 
      template<unsigned int THREAD_BLOCK_DIM= 128>
      self& operator*=(const value_type rhs) {
        std::cout <<"scalar operator*= lvalue\n";
        static_assert(THREAD_BLOCK_DIM <= 1024);
        const size_type size{this->size()};
        constexpr unsigned int threads{THREAD_BLOCK_DIM};
        const auto blocks{static_cast<unsigned int>((size + 2 * static_cast<size_type>(threads) - 1) / (2 * static_cast<size_type>(threads)))};
        constexpr dim3 dimBlock(threads, 1, 1);
        dim3 dimGrid(blocks, 1, 1);
        mulScalarTo_kernel<value_type, threads><<<dimGrid, dimBlock>>>(this->data(), rhs, size);
        getLastCudaError("mulScalarTo_kernel() execution failed.\n");
        return *this;
      }
        
      // Increasing ILP kernel using the class inside the kernel
      template<unsigned int THREAD_BLOCK_DIM= 128> 
      self& operator*=(const self& rhs) {
        std::cout <<"operator*= lvalue\n";
        static_assert(THREAD_BLOCK_DIM <= 1024);
        if (!check_equal_size((*this), rhs))
          throw std::length_error{"Incompatible sizes for matrix-matrix Hadamard product"};
        device_matrix<value_type>* dp_this;
        device_matrix<value_type>* dp_rhs;
        checkCudaErrors(cudaMalloc((void**)& dp_this, sizeof(device_matrix<value_type>)));
        checkCudaErrors(cudaMalloc((void**)& dp_rhs,  sizeof(device_matrix<value_type>)));
        checkCudaErrors(cudaMemcpy(dp_this, this,     sizeof(device_matrix<value_type>), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(dp_rhs,  &rhs,     sizeof(device_matrix<value_type>), cudaMemcpyHostToDevice));
        const size_type size{this->size()};
        constexpr unsigned int threads{THREAD_BLOCK_DIM};
        const auto blocks{static_cast<unsigned int>((size + 2 * static_cast<size_type>(threads) - 1) / (2 * static_cast<size_type>(threads)))};
        constexpr dim3 dimBlock(threads, 1, 1);
        dim3 dimGrid(blocks, 1, 1);
        mulTo_kernel<value_type, Allocator, threads><<<dimGrid, dimBlock>>>(*dp_this, *dp_rhs);
        getLastCudaError("mulTo_kernel() execution failed.\n");
        checkCudaErrors(cudaFree(dp_this));
        checkCudaErrors(cudaFree(dp_rhs));
        return *this;
      }

      device_allocator<T> get_allocator() const { return this->get_allocator();}

  };

  template<std::floating_point T>
  void swap(device_matrix<T>& lhs, device_matrix<T>& rhs);
     
  template<std::floating_point T>
  constexpr bool check_equal_size(const device_matrix<T>& lhs, const device_matrix<T>& rhs);
    
  template<std::floating_point T, unsigned int THREAD_BLOCK_DIM= 128>
  constexpr bool operator==(const device_matrix<T>& lhs, const device_matrix<T>& rhs);
    
  template<std::floating_point T, unsigned int THREAD_BLOCK_DIM= 128>
  device_matrix<T> operator+(device_matrix<T>&& res, const device_matrix<T>& rhs);
    
  template<std::floating_point T, unsigned int THREAD_BLOCK_DIM= 128>
  device_matrix<T> operator+(const device_matrix<T>& lhs, const device_matrix<T>& rhs);
    
  template<std::floating_point T, unsigned int THREAD_BLOCK_DIM= 128>
  device_matrix<T> operator-(device_matrix<T>&& res, const device_matrix<T>& rhs);
    
  template<std::floating_point T, unsigned int THREAD_BLOCK_DIM= 128>
  device_matrix<T> operator-(const device_matrix<T>& lhs, const device_matrix<T>& rhs);
    
  template<std::floating_point T, unsigned int THREAD_BLOCK_DIM= 128>
  device_matrix<T> operator*(device_matrix<T>&& res, const T rhs);
    
  template<std::floating_point T, unsigned int THREAD_BLOCK_DIM= 128>
  device_matrix<T> operator*(const device_matrix<T>& lhs, const T rhs);
    
  template<std::floating_point T, unsigned int THREAD_BLOCK_DIM= 128>
  device_matrix<T> operator*(const T rhs, const device_matrix<T>& lhs);
    
  template<std::floating_point T, unsigned int THREAD_BLOCK_DIM= 128>
  device_matrix<T> operator*(device_matrix<T>&& res, const device_matrix<T>& rhs);
    
  template<std::floating_point T, unsigned int THREAD_BLOCK_DIM= 128>
  device_matrix<T> operator*(const device_matrix<T>& lhs, const device_matrix<T>& rhs);
    
  template<std::floating_point T>
  void print_matrix(const device_matrix<T>& mat);

}

#include <mathcca/device_matrix.inl>

#endif

