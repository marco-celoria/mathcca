#ifndef HOST_MATRIX_H_
#define HOST_MATRIX_H_
#pragma once

#include <cstddef>
#include <concepts>
#include <iostream>

#include <mathcca/host_iterator.h>
#include <mathcca/copy.h>
#include <mathcca/fill_const.h>

#ifdef __CUDACC__
 #include <mathcca/device_matrix.h>
#endif

#ifdef _OPENMP
 #include <omp.h>
#endif

#ifdef _MKL
 #include <mkl.h>
#endif

namespace mathcca {

  namespace matricca {

#ifdef __CUDACC__
    template<std::floating_point T>	
    class device_matrix;
#endif
    
    template<std::floating_point T>
    class host_matrix {
    
      using self= host_matrix; 
      
      public:
        
        using value_type= T;
        using size_type= std::size_t;
        using reference= T&;
        using const_reference= const T&;
        using pointer= T*;//device_ptr<T[]>;
        using const_pointer= const T*; //device_ptr<const T[]>;
        using iterator= mathcca::iterator::host_iterator<T>;
        using const_iterator= mathcca::iterator::host_iterator<const T>;
        
        constexpr host_matrix(size_type r, size_type c) : num_rows_{r}, num_cols_{c} { 
          data_ = new T[num_rows_ * num_cols_]{};      
          std::cout << "host_matrix custom ctor\n";
        }
        
        constexpr host_matrix(size_type r, size_type c, const_reference v) : host_matrix(r, c) {
          mathcca::algocca::fill_const(begin(), end(), v);
          std::cout << "(host_matrix delegating ctor)\n";
        } 
         
        constexpr ~host_matrix() {
          /**/       
          if (data_) {
            delete[] data_; 
            data_= nullptr; 
          }
          /**/
	  num_rows_= 0;
	  num_rows_= 0;
          std::cout << "host_matrix dtor\n"; 
        };
        
        constexpr host_matrix(self&& m):num_rows_{std::move(m.num_rows_)}, num_cols_{std::move(m.num_cols_)}, data_{std::move(m.data_)} {
          m.num_rows_= 0;
          m.num_rows_= 0;
          m.data_= nullptr; /**/
          std::cout << "host_matrix move ctor\n";
        }
        
        constexpr host_matrix(const self& m) : host_matrix{m.num_rows_, m.num_cols_} {
          algocca::copy(m.cbegin(), m.cend(), begin());
          std::cout << "host_matrix copy ctor\n";
        }

        constexpr host_matrix<T>& operator=(host_matrix&& rhs) {
          /**/
          if (data_)
            delete[] data_;
          /**/
          num_rows_= std::move(rhs.num_rows_);
          num_cols_= std::move(rhs.num_cols_);
          data_= std::move(rhs.data_);
          rhs.num_rows_= 0;
          rhs.num_cols_= 0;
          rhs.data_= nullptr; /**/
          std::cout << "host_matrix move assignment\n";
          return *this;
        }

        constexpr host_matrix<T>& operator=(const host_matrix& rhs) {
          if (this != &rhs) {
            std::cout << "host_matrix copy assignment (\n";
            if (this->size() != rhs.size()) {
              auto tmp{rhs};            // use copy ctor
              (*this)= std::move(tmp);  // finally move assignment
            }
            else {
              num_rows_= rhs.num_rows_;
              num_cols_= rhs.num_cols_;
              algocca::copy(rhs.cbegin(), rhs.cend(), begin());
            }
            std::cout << ")\n";
          }
          return *this;
        }
         
        constexpr reference operator[] (size_type i) noexcept {return data_[i]; }
        constexpr const_reference operator[] (size_type i) const noexcept {return data_[i]; }
        
        constexpr reference operator() (size_type j, size_type i) noexcept {return data_[j * num_cols_ + i]; }
        constexpr const_reference operator() (size_type j, size_type i) const noexcept {return data_[j * num_cols_ + i]; }
         
        constexpr size_type num_rows() const noexcept { return num_rows_; }
        constexpr size_type num_cols() const noexcept { return num_cols_; }
        
        constexpr size_type size() const noexcept { return num_rows_ * num_cols_; }
        
        constexpr pointer data() noexcept { return data_; } 
        constexpr const_pointer data() const noexcept { return data_; } 
        
        constexpr iterator begin() noexcept { return iterator{data_/*.get()*/}; }
        constexpr iterator end()   noexcept { return iterator{data_/*.get()*/ + size()}; }
        
        constexpr const_iterator begin()  const noexcept { return const_iterator{data_/*.get()*/}; }
        constexpr const_iterator end()    const noexcept { return const_iterator{data_/*.get()*/ + size()}; }
        
        constexpr const_iterator cbegin() const noexcept { return const_iterator{data_/*.get()*/}; }
        constexpr const_iterator cend()   const noexcept { return const_iterator{data_/*.get()*/ + size()}; }
        
#ifdef __CUDACC__
        auto toDevice () const {
          device_matrix<T> devMat(num_rows_, num_cols_);
	  algocca::copy(begin(), end(), devMat.begin());
          return devMat;
        }
#endif
        
        constexpr static auto tol() noexcept {
          if constexpr (std::is_same_v<value_type, double>) {
            return 1e-5;
          } else {
            return static_cast<float>(1e-2);
          }
        }
        
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
        
      private:
        
        size_type num_rows_{0};
        size_type num_cols_{0};
        pointer data_{nullptr};
        
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
  constexpr host_matrix<T> operator*(host_matrix<T>&& A, const host_matrix<T>& B);
  
  
  template<std::floating_point T>
  constexpr host_matrix<T> operator*(const host_matrix<T>& A, const host_matrix<T>& B);
  
  template<std::floating_point T>
  void print_matrix(const host_matrix<T>& mat);

  }
}

#include <mathcca/detail/host_matrix.inl>

#endif



