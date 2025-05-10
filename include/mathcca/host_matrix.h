#ifndef HOST_MATRIX_H_
#define HOST_MATRIX_H_
#pragma once

#include <cstddef>
#include <concepts>
#include <iostream>

#include <mathcca/host_iterator.h>
#include <mathcca/copy.h>
#include <mathcca/fill_const.h>

#include <mathcca/host_allocator.h>

#ifdef __CUDACC__
 #include <mathcca/device_matrix.h>
 #include <mathcca/device_allocator.h>
#endif

#ifdef _OPENMP
 #include <omp.h>
#endif

#ifdef _MKL
 #include <mkl.h>
#endif

namespace mathcca {
    

#ifdef __CUDACC__
    template<std::floating_point T, typename Allocator>	
    class device_matrix;
#endif
    
    template<std::floating_point T, typename Allocator = host_allocator<T>>
    class host_matrix {
    
      using self= host_matrix; 
      
      public:
        
//        template <bool IsConst>
//        class host_iterator;
        
        using value_type= T;
        using size_type= std::size_t;
        using reference= T&;
        using const_reference= const T&;
        using pointer= T*;//device_ptr<T[]>;
        using const_pointer= const T*; //device_ptr<const T[]>;
        using iterator= /*mathcca::iterator::*/host_iterator<T, false>;
        using const_iterator= /*mathcca::iterator::*/host_iterator<T, true>;
	using traits_alloc = std::allocator_traits<Allocator>;
        
        host_matrix(Allocator a) : allocator{std::move(a)} {}
        
        constexpr host_matrix(size_type r, size_type c) : num_rows_{r}, num_cols_{c} {
          data_ = traits_alloc::allocate(allocator, size());    	
          //std::cout << "host_matrix custom ctor\n";
          std::cout << "custom ctor\n";
        }
        
        constexpr host_matrix(size_type r, size_type c, const_reference v) : host_matrix(r, c) {
          mathcca::fill_const(begin(), end(), v);
          //std::cout << "(host_matrix delegating ctor)\n";
          std::cout << "(delegating ctor)\n";
        } 
         
        constexpr ~host_matrix() {
          /**/       
          if (data_) {
            traits_alloc::deallocate(allocator, data_, size());  
            data_= nullptr; 
          }
          /**/
	  num_rows_= 0;
	  num_rows_= 0;
          //std::cout << "host_matrix dtor\n"; 
          std::cout << "dtor\n"; 
        };
        
        constexpr host_matrix(self&& m):num_rows_{std::move(m.num_rows_)}, num_cols_{std::move(m.num_cols_)}, data_{std::move(m.data_)} {
          m.num_rows_= 0;
          m.num_rows_= 0;
          m.data_= nullptr; /**/
          std::cout << "move ctor\n";
          //std::cout << "host_matrix move ctor\n";
        }
        
        constexpr host_matrix(const self& m) : host_matrix{m.num_rows_, m.num_cols_} {
          copy(m.cbegin(), m.cend(), begin());
          std::cout << "copy ctor\n";
          //std::cout << "host_matrix copy ctor\n";
        }
        
        constexpr host_matrix<T>& operator=(host_matrix&& rhs) {
          /**/
          if (data_)
            traits_alloc::deallocate(allocator, data_, size());		  
          /**/
          num_rows_= std::move(rhs.num_rows_);
          num_cols_= std::move(rhs.num_cols_);
          data_= std::move(rhs.data_);
          rhs.num_rows_= 0;
          rhs.num_cols_= 0;
          rhs.data_= nullptr; /**/
          std::cout << "move assignment\n";
          //std::cout << "host_matrix move assignment\n";
          return *this;
        }
        
        constexpr host_matrix<T>& operator=(const host_matrix& rhs) {
          if (this != &rhs) {
            std::cout << "copy assignment (\n";
            //std::cout << "host_matrix copy assignment (\n";
            if (this->size() != rhs.size()) {
              auto tmp{rhs};            // use copy ctor
              (*this)= std::move(tmp);  // finally move assignment
            }
            else {
              num_rows_= rhs.num_rows_;
              num_cols_= rhs.num_cols_;
              copy(rhs.cbegin(), rhs.cend(), begin());
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
        
	iterator begin() noexcept { return iterator{data_/*.get()*/}; }
        iterator end()   noexcept { return iterator{data_/*.get()*/ + size()}; }
        
        const_iterator begin()  const noexcept { return const_iterator{data_/*.get()*/}; }
        const_iterator end()    const noexcept { return const_iterator{data_/*.get()*/ + size()}; }
        
        const_iterator cbegin() const noexcept { return const_iterator{data_/*.get()*/}; }
        const_iterator cend()   const noexcept { return const_iterator{data_/*.get()*/ + size()}; }
        
#ifdef __CUDACC__
        auto toDevice () const {
          device_matrix<T, device_allocator<T>> devMat(num_rows_, num_cols_);
	  copy(begin(), end(), devMat.begin());
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
        Allocator allocator;

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
/*
    template<std::floating_point T>
    template <bool IsConst>
    class host_matrix<T>::host_iterator {
      public:
        using value_type= T;
        using difference_type= std::ptrdiff_t;
        using pointer= T*;
        using reference= T&;
        using const_reference=const T&;
        using iterator_system= host_iterator_tag;
        using iterator_category= std::contiguous_iterator_tag;
        
        host_iterator() : ptr_{nullptr} {}
        
	explicit host_iterator(pointer x) : ptr_{x} {}
        
	host_iterator(const host_iterator&)= default;
        
        template<bool IsConst_ = IsConst, class = std::enable_if_t<IsConst_>>
        host_iterator(const host_iterator<false>& rhs) : ptr_(rhs.get()) {}  // OK
        
        template<bool IsConst_ = IsConst, class = std::enable_if_t<IsConst_>>
        host_iterator& operator=(const host_iterator<false>& rhs) { ptr_ = rhs.ptr_; return *this; }
        
	template <bool Q = IsConst>
        typename std::enable_if_t<Q, const_reference> operator*() const noexcept { return *ptr_; }
        
	template <bool Q = IsConst>
        typename std::enable_if_t<!Q, reference> operator*() const noexcept { return *ptr_; }
        
	template <bool Q = IsConst>
        typename std::enable_if_t<Q, const_reference> operator[](difference_type n) const noexcept { return *(ptr_ + n); }
	
	template <bool Q = IsConst>
        typename std::enable_if_t<!Q, reference> operator[](difference_type n) const noexcept { return *(ptr_ + n); }
        
	pointer operator->()  const noexcept { return ptr_; }
        pointer get() const noexcept { return ptr_; }
        
        host_iterator& operator++()   { ++ptr_; return *this; }
        host_iterator operator++(int) { auto tmp= *this; ++(*this); return tmp; }
        host_iterator& operator--()   { --ptr_; return *this; }
        host_iterator operator--(int) { auto tmp= *this; --(*this); return tmp; }
        
        host_iterator& operator+=(difference_type n) { ptr_+= n; return *this; }
        host_iterator& operator-=(difference_type n) { ptr_-= n; return *this; }
        
        friend bool operator==(const host_iterator& x, const host_iterator& y) { return x.get() == y.get(); }
        
        friend bool operator!=(const host_iterator& x, const host_iterator& y) { return !(x == y); }
        
        friend bool operator<(const host_iterator& lhs, const host_iterator& rhs) { return lhs.get() < rhs.get(); }
        
        friend bool operator>(const host_iterator& lhs, const host_iterator& rhs)  { return rhs < lhs; }
        
        friend bool operator<=(const host_iterator& lhs, const host_iterator& rhs) { return !(rhs < lhs); }
        
        friend bool operator>=(const host_iterator& lhs, const host_iterator& rhs) { return !(lhs < rhs); }
        
        friend  host_iterator operator+(const host_iterator& it, difference_type n) {
          host_iterator temp= it;
          temp+= n;
          return temp;
        }
        
        friend difference_type operator+(difference_type n, const host_iterator& it) { return it + n; }
        
        friend host_iterator operator-(const host_iterator& it, difference_type n) {
          host_iterator temp= it;
          temp-= n;
          return temp;
        }
        
        friend difference_type operator-(const host_iterator& lhs, const host_iterator& rhs) { return lhs.get() - rhs.get(); }
        
      private:
        
        pointer ptr_;
        
    };

*/

}

#include <mathcca/detail/host_matrix.inl>

#endif



