#ifndef HOST_MATRIX_H_
#define HOST_MATRIX_H_
#pragma once

#include <cstddef>
#include <concepts>
#include <iostream>

#include <mathcca/base_matrix.h>

#include <mathcca/host_allocator.h>
#include <mathcca/host_iterator.h>

#ifdef __CUDACC__
 #include <mathcca/device_allocator.h>
 #include <mathcca/device_matrix.h>
#endif

#include <mathcca/copy.h>
#include <mathcca/fill_const.h>


#ifdef _OPENMP
 #include <omp.h>
#endif

#ifdef _MKL
 #include <mkl.h>
#endif

namespace mathcca {
    

#ifdef __CUDACC__
    template<std::floating_point T, typename Allocator, typename Execution>	
    class device_matrix;
#endif
    
    template<std::floating_point T, typename Allocator = host_allocator<T>, typename Execution= Omp>
    class host_matrix : public base_matrix<T, Allocator, Execution > {
    
      using self= host_matrix; 
      typedef base_matrix<T,Allocator, Execution> Parent;

      public:
        
        typedef typename Parent::size_type  size_type;
        typedef typename Parent::value_type  value_type;
        typedef typename Parent::reference reference;
        typedef typename Parent::const_reference const_reference;
        typedef typename Parent::pointer pointer;
        typedef typename Parent::const_pointer const_pointer;

        using iterator= host_iterator<T, false>;
        using const_iterator= host_iterator<T, true>;
	using traits_alloc = std::allocator_traits<Allocator>;
        
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
         
        constexpr reference operator[] (size_type i) noexcept {return Parent::data()[i]; }
        constexpr const_reference operator[] (size_type i) const noexcept {return  Parent::data()[i]; }
        
        constexpr reference operator() (size_type j, size_type i) noexcept {return  Parent::data()[j *  Parent::num_cols() + i]; }
        constexpr const_reference operator() (size_type j, size_type i) const noexcept {return  Parent::data()[j *  Parent::num_cols() + i]; }
         
	iterator begin() noexcept { return iterator{ Parent::data()}; }
        iterator end()   noexcept { return iterator{ Parent::data() +  Parent::size()}; }
        
        const_iterator begin()  const noexcept { return const_iterator{ Parent::data()}; }
        const_iterator end()    const noexcept { return const_iterator{ Parent::data() +  Parent::size()}; }
        
        const_iterator cbegin() const noexcept { return const_iterator{ const_cast<pointer>(Parent::data())}; }
        const_iterator cend()   const noexcept { return const_iterator{ const_cast<pointer>(Parent::data() +  Parent::size()) }; }
        
#ifdef __CUDACC__
        auto toDevice () const {
          device_matrix<T, device_allocator<T>, Cuda> devMat(Parent::num_rows(), Parent::num_cols());
	  copy(begin(), end(), devMat.begin());
          return devMat;
        }
#endif
        
        constexpr self& operator+=(const self& rhs) {
          std::cout <<"operator+= lvalue\n";
          if (!check_equal_size((*this), rhs))
            throw std::length_error{"Incompatible sizes for matrix-matrix addition"};
          const auto size{this->size()}; 
          #pragma omp parallel for default(shared)
          for (size_type i= 0; i < size; ++i)
            (*this)[i] += rhs[i];
          return *this;
        }
        
        constexpr self& operator-=(const self& rhs) {
          std::cout <<"operator-= lvalue\n";
          if (!check_equal_size((*this), rhs))
            throw std::length_error{"Incompatible sizes for matrix-matrix subtraction"};
          const auto size{this->size()}; 
          #pragma omp parallel for default(shared)
          for (size_type i= 0; i < size; ++i)
            (*this)[i] -= rhs[i];
          return *this;
        }
        
        constexpr self& operator*=(const value_type rhs) {
          std::cout <<"scalar operator*= lvalue\n";
          const auto size{this->size()};
          #pragma omp parallel for default(shared)
          for (size_type i= 0; i < size; ++i)
            (*this)[i] *= rhs;
          return *this;
        }
        
	constexpr self& operator*=(const self& rhs) {
          std::cout <<"operator*= lvalue\n";
          if (!check_equal_size((*this), rhs))
            throw std::length_error{"Incompatible sizes for matrix-matrix Hadamard product"};
          const auto size{this->size()};
          #pragma omp parallel for default(shared)
          for (size_type i= 0; i < size; ++i)
            (*this)[i] *= rhs[i];
          return *this;
        }
        
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
    constexpr host_matrix<T> operator*(host_matrix<T>&& A, const host_matrix<T>& B);
  
    template<std::floating_point T>
    constexpr host_matrix<T> operator*(const host_matrix<T>& A, const host_matrix<T>& B);
  
    template<std::floating_point T>
    void print_matrix(const host_matrix<T>& mat);

}

#include <mathcca/detail/host_matrix.inl>

#endif



