/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef HOST_MATRIX_H_
#define HOST_MATRIX_H_
#pragma once

#include <cstddef>   // std::size_t
#include <concepts>  // std::floating_point
#include <iostream>  // std::cout
#include <memory>    // std::allocator_traits
#include <utility>   // std::forward
#include <stdexcept> // std::length_error

#include <mathcca/detail/base_matrix.h>

#include <mathcca/execution_policy.h> // Omp
#ifdef _PINNED
#include <mathcca/pinnedhost_allocator.h>
#else
#include <mathcca/host_allocator.h>
#endif
#include <mathcca/host_iterator.h>

#ifdef _OPENMP
 #include <omp.h>
#endif

namespace mathcca {

#if defined (__CUDACC__) && defined(_PINNED)	
  template<std::floating_point T, typename Allocator = pinnedhost_allocator<T>>
#else	
  template<std::floating_point T, typename Allocator = host_allocator<T>>
#endif
  class host_matrix : public detail::base_matrix<T, Allocator > {
    
    using self= host_matrix; 
    typedef detail::base_matrix<T,Allocator> Parent;
      
    public:
        
      typedef typename Parent::size_type  size_type;
      typedef typename Parent::value_type  value_type;
      typedef typename Parent::reference reference;
      typedef typename Parent::const_reference const_reference;
      typedef typename Parent::pointer pointer;
      typedef typename Parent::const_pointer const_pointer;
         
      using iterator= host_iterator<T, false>;
      using const_iterator= host_iterator<T, true>;
      using traits_alloc= std::allocator_traits<Allocator>;
        
      host_matrix(Allocator a) : Parent(std::forward<Allocator>(a)) {}
        
      constexpr host_matrix(size_type r, size_type c) : Parent(r,c) {}
        
      constexpr host_matrix(size_type r, size_type c, const_reference v) : Parent(r,c,v)  {} 
         
      constexpr ~host_matrix() {}
        
      constexpr host_matrix(self&& m): Parent(std::forward<host_matrix<T>>(m)) {}
       
      constexpr host_matrix(const self& m) : Parent(m) {}	
        
      constexpr host_matrix<T>& operator=(host_matrix&& rhs) {
        Parent::operator=(std::forward<host_matrix<T>>(rhs));
        return *this;
      }
        
      constexpr host_matrix<T>& operator=(const host_matrix& rhs) {
        Parent::operator=(rhs);
        return *this;
      }
         
      constexpr reference operator[] (size_type i) noexcept {return this->data()[i]; }
      constexpr const_reference operator[] (size_type i) const noexcept {return  this->data()[i]; }
        
      constexpr reference operator() (size_type j, size_type i) noexcept {return  this->data()[j *  this->num_cols() + i]; }
      constexpr const_reference operator() (size_type j, size_type i) const noexcept {return this->data()[j *  this->num_cols() + i]; }

      iterator begin() noexcept { return iterator{ this->data()}; }
      iterator end()   noexcept { return iterator{ this->data() +  this->size()}; }
        
      const_iterator begin()  const noexcept { return const_iterator{ const_cast<pointer>(this->data())}; }
      const_iterator end()    const noexcept { return const_iterator{ const_cast<pointer>(this->data() +  this->size())}; }
        
      const_iterator cbegin() const noexcept { return const_iterator{ const_cast<pointer>(this->data())}; }
      const_iterator cend()   const noexcept { return const_iterator{ const_cast<pointer>(this->data() +  this->size()) }; }
        
      constexpr self& operator+=(const self& rhs) {
        std::cout <<"operator+= lvalue\n";
        if (!check_equal_size((*this), rhs))
          throw std::length_error{"Incompatible sizes for matrix-matrix addition"};
#ifdef _STDPAR
	std::transform(std::execution::par_unseq, begin().get(), end().get(), rhs.begin().get(), begin().get(), [=](value_type lhsi, value_type rhsi){ return lhsi + rhsi; });
#else	
        const auto size{this->size()};
        #pragma omp parallel for default(shared)
        for (size_type i= 0; i < size; ++i)
          (*this)[i] += rhs[i];
#endif  
        return *this;
      } 
         
      constexpr self& operator-=(const self& rhs) {
        std::cout <<"operator-= lvalue\n";
        if (!check_equal_size((*this), rhs))
          throw std::length_error{"Incompatible sizes for matrix-matrix subtraction"};
#ifdef _STDPAR
	std::transform(std::execution::par_unseq, begin().get(), end().get(), rhs.begin().get(), begin().get(), [=](value_type lhsi, value_type rhsi){ return lhsi - rhsi; });
#else    
        const auto size{this->size()}; 
        #pragma omp parallel for default(shared)
        for (size_type i= 0; i < size; ++i)
          (*this)[i] -= rhs[i];
#endif  
        return *this;
      } 
        
      constexpr self& operator*=(const value_type rhs) {
        std::cout <<"scalar operator*= lvalue\n";
#ifdef _STDPAR
        std::transform(std::execution::par_unseq, begin().get(), end().get(), begin().get(), [=](value_type lhsi){ return lhsi * rhs; });
#else
        const auto size{this->size()};
        #pragma omp parallel for default(shared)
        for (size_type i= 0; i < size; ++i)
          (*this)[i] *= rhs;
#endif  
        return *this;
      } 
        
      constexpr self& operator*=(const self& rhs) {
        std::cout <<"operator*= lvalue\n";
        if (!check_equal_size((*this), rhs))
          throw std::length_error{"Incompatible sizes for matrix-matrix Hadamard product"};
#ifdef _STDPAR
        std::transform(std::execution::par_unseq, begin().get(), end().get(), rhs.begin().get(), begin().get(), [=](value_type lhsi, value_type rhsi){ return lhsi * rhsi; });
#else   
        const auto size{this->size()};
        #pragma omp parallel for default(shared)
        for (size_type i= 0; i < size; ++i)
          (*this)[i] *= rhs[i];
#endif   
        return *this;
      } 

      Allocator get_allocator() const { return this->get_allocator();}

  };  
    
  /* Swap and checks */
     
  template<std::floating_point T>
  void swap(host_matrix<T>& a, host_matrix<T>& b);
    
  template<std::floating_point T>
  constexpr bool check_equal_size(const host_matrix<T>& lhs, const host_matrix<T>& rhs);
   
  /* Operator Overloadings */

  template<std::floating_point T>
  constexpr bool operator==(const host_matrix<T>& lhs, const host_matrix<T>& rhs);  
    
  template<std::floating_point T>
  constexpr host_matrix<T> operator+ (host_matrix<T>&& res, const host_matrix<T>& rhs);
  
  template<std::floating_point T>
  constexpr host_matrix<T> operator+ (const host_matrix<T>& lhs, const host_matrix<T>& rhs);
  
  template<std::floating_point T>
  constexpr host_matrix<T> operator- (host_matrix<T>&& res, const host_matrix<T>& rhs);
    
  template<std::floating_point T>
  constexpr host_matrix<T> operator- (const host_matrix<T>& lhs, const host_matrix<T>& rhs);
    
  template<std::floating_point T>
  constexpr host_matrix<T> operator*(host_matrix<T>&& A, const T B);
  
  template<std::floating_point T>
  constexpr host_matrix<T> operator*(const host_matrix<T>& A, const T B);
   
  template<std::floating_point T>
  constexpr host_matrix<T> operator*(const T B, const host_matrix<T>& A);
    
  template<std::floating_point T>
  constexpr host_matrix<T> operator*(host_matrix<T>&& A, const host_matrix<T>& B);
  
  template<std::floating_point T>
  constexpr host_matrix<T> operator*(const host_matrix<T>& A, const host_matrix<T>& B);
  
  template<std::floating_point T>
  void print_matrix(const host_matrix<T>& mat);

}


#include <mathcca/host_matrix.inl>


#endif



