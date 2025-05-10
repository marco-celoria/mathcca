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

#include <cstddef>
#include <concepts>
#include <iostream>
#include <cuda_runtime.h>

#include <mathcca/copy.h>
#include <mathcca/fill_const.h>
#include <mathcca/device_helper.h>
#include <mathcca/host_matrix.h>
#include <mathcca/device_iterator.h>
#include <mathcca/device_allocator.h>

namespace mathcca {
    
    
    template<std::floating_point T, typename Allocator>
    class host_matrix;
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    __global__ void addTo_kernel(T* __restrict accululator, const T* __restrict to_be_op, const std::size_t size);
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    __global__ void subTo_kernel(T* __restrict accululator, const T* __restrict to_be_op, const std::size_t size);
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    __global__ void mulTo_kernel(T* __restrict accululator, const T* __restrict to_be_op, const std::size_t size);
    
    template<std::floating_point T, typename Allocator = device_allocator<T>> 
    class device_matrix {
      
      using self= device_matrix; 
      
      public:
          
//        template <bool IsConst>
//        class device_iterator;

        using value_type= T;
        using size_type= std::size_t;
        using reference= T&;
        using const_reference= const T&;
        using pointer= T*;                  //device_ptr<T[]>;
        using const_pointer= const T*;      //device_ptr<const T[]>;
        using iterator= /*mathcca::iterator::*/device_iterator<T, false>;
        using const_iterator= /*mathcca::iterator::*/device_iterator<T, true>;
        using traits_alloc = std::allocator_traits<Allocator>;
        
	device_matrix(Allocator a) : allocator{std::move(a)} {}

        device_matrix(size_type r, size_type c) : num_rows_{r}, num_cols_{c} {
          data_ = traits_alloc::allocate(allocator, size()); 
          std::cout << "custom ctor\n";
          //std::cout << "device_matrix custom ctor\n";
        }
        
        device_matrix(size_type r, size_type c, const_reference v) : device_matrix(r, c) {
          fill_const(begin(), end(), v);
          std::cout << "(delegating ctor)\n";
          //std::cout << "(device_matrix delegating ctor)\n";
        } 
   
        ~device_matrix() { 
          //
          if (data_) {
            traits_alloc::deallocate(allocator, data_, size()); 
	    data_=nullptr; 
	  }
	  // 
	  std::cout << "dtor\n"; 
	  //std::cout << "device_matrix dtor\n"; 
	}
        
        device_matrix(self && m) : num_rows_{std::move(m.num_rows_)}, num_cols_{std::move(m.num_cols_)}, data_{std::move(m.data_)} {
          std::cout << "move ctor\n";
          //std::cout << "device_matrix move ctor\n";
          m.num_rows_ = 0;
          m.num_cols_ = 0;
          m.data_ = nullptr; //
        }

        device_matrix(const self& m) : device_matrix(m.num_rows_, m.num_cols_) {
          std::cout << "copy ctor\n";
          //std::cout << "device_matrix copy ctor\n";
          copy(m.begin(), m.end(), begin());
        }

       self& operator=(device_matrix&& rhs) {
          std::cout << "move assignment\n";
          //std::cout << "device_matrix move assignment\n";
	  //
          if (data_) 
            traits_alloc::deallocate(allocator, data_, size());
          //
	  num_rows_= std::move(rhs.num_rows_);
          num_cols_= std::move(rhs.num_cols_);
          data_= std::move(rhs.data_);
          rhs.num_rows_= 0;
          rhs.num_cols_= 0;
          rhs.data_= nullptr; //
          return *this;
        }
        
        self& operator=(const device_matrix& rhs) {
          if (this != &rhs) {
            std::cout << "copy assignment (\n";
            //std::cout << "device_matrix copy assignment (\n";
            if (size() != rhs.size()) {
              auto tmp{rhs};                    // use copy ctor
              (*this)= std::move(tmp);          // finally move assignment
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
         
         size_type num_rows() const noexcept { return num_rows_; }
         size_type num_cols() const noexcept { return num_cols_; }
        
         size_type size() const noexcept { return num_rows_ * num_cols_; }
        
         pointer data() noexcept { return data_; } 
         const_pointer data() const noexcept { return data_; } 
       
         iterator begin() noexcept { return iterator{data_}; }
         iterator end()   noexcept { return iterator{data_ + size()}; }
        
         const_iterator cbegin() const noexcept { return const_iterator{data_}; }
         const_iterator cend()   const noexcept { return const_iterator{data_ + size()}; }
         
	 const_iterator begin() const  noexcept { return const_iterator{data_}; }
         const_iterator end()   const  noexcept { return const_iterator{data_ + size()}; }
       
        constexpr static auto tol() noexcept { 
          if constexpr (std::is_same_v<value_type, double>) {
            return 1e-5;
          } else { 
            return static_cast<float>(1e-2);
          }
        }
        
        auto toHost (cudaStream_t stream= 0) const {
          host_matrix<T> hostA(num_rows_, num_cols_);
          copy(begin(), end(), hostA.begin(), stream);
          return hostA;
        }
         
        template<unsigned int THREAD_BLOCK_DIM= 128>
        self& operator+=(const self& rhs) {
          std::cout <<"operator+= lvalue\n";
          static_assert(THREAD_BLOCK_DIM <= 1024);
          if (!check_equal_size((*this), rhs))
            throw std::length_error{"Incompatible sizes for matrix-matrix addition"};
          const size_type size{this->size()};
          constexpr unsigned int threads{THREAD_BLOCK_DIM};
          const auto blocks{static_cast<unsigned int>((size + 2 * static_cast<size_type>(threads) - 1) / (2 * static_cast<size_type>(threads)))};
          constexpr dim3 dimBlock(threads, 1, 1);
          dim3 dimGrid(blocks, 1, 1);
          addTo_kernel<value_type, threads><<<dimGrid, dimBlock>>>(data_, rhs.data(), size);
          return *this;
        }
        
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
          subTo_kernel<value_type, threads><<<dimGrid, dimBlock>>>(data_, rhs.data(), size);
          return *this;
        }
        
        template<unsigned int THREAD_BLOCK_DIM= 128> 
        self& operator*=(const self& rhs) {
          std::cout <<"operator*= lvalue\n";
          static_assert(THREAD_BLOCK_DIM <= 1024);
          if (!check_equal_size((*this), rhs))
            throw std::length_error{"Incompatible sizes for matrix-matrix Hadamard product"};
          const size_type size{this->size()};
          constexpr unsigned int threads{THREAD_BLOCK_DIM};
          const auto blocks{static_cast<unsigned int>((size + 2 * static_cast<size_type>(threads) - 1) / (2 * static_cast<size_type>(threads)))};
          constexpr dim3 dimBlock(threads, 1, 1);
          dim3 dimGrid(blocks, 1, 1);
          mulTo_kernel<value_type, threads><<<dimGrid, dimBlock>>>(data_, rhs.data(), size);
          return *this;
        }
        
      private:
        
        size_type num_rows_{0};
        size_type num_cols_{0};
        pointer data_{nullptr};
        Allocator allocator;
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
    device_matrix<T> operator*(device_matrix<T>&& res, const device_matrix<T>& rhs);
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM= 128>
    device_matrix<T> operator*(const device_matrix<T>& lhs, const device_matrix<T>& rhs);
    
    template<std::floating_point T>
    void print_matrix(const device_matrix<T>& mat);
    /* 
    template<std::floating_point T>
    template <bool IsConst>
    class device_matrix<T>::device_iterator {
      public:
        using value_type= T;
        using difference_type= std::ptrdiff_t;
        using pointer= T*;
        using reference= T&;
        using const_reference=const T&;
        using iterator_system = device_iterator_tag;
        using iterator_category = std::contiguous_iterator_tag;

        device_iterator() : ptr_{nullptr} {}

        explicit device_iterator(pointer x) : ptr_{x} {}

        device_iterator(const device_iterator&)= default;

        template<bool IsConst_ = IsConst, class = std::enable_if_t<IsConst_>>
        device_iterator(const device_iterator<false>& rhs) : ptr_(rhs.get()) {}  // OK
        template<bool IsConst_ = IsConst, class = std::enable_if_t<IsConst_>>
        device_iterator& operator=(const device_iterator<false>& rhs) { ptr_ = rhs.ptr_; return *this; }

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
        
        device_iterator& operator++()   { ++ptr_; return *this; }
        device_iterator operator++(int) { auto tmp= *this; ++(*this); return tmp; }
        device_iterator& operator--()   { --ptr_; return *this; }
        device_iterator operator--(int) { auto tmp= *this; --(*this); return tmp; }
        device_iterator& operator+=(difference_type n) { ptr_+= n; return *this; }
        device_iterator& operator-=(difference_type n) { ptr_-= n; return *this; }
        
        friend bool operator==(const device_iterator& x, const device_iterator& y) { return x.get() == y.get(); }
        friend bool operator!=(const device_iterator& x, const device_iterator& y) { return !(x == y); }
        friend bool operator<(const device_iterator& lhs, const device_iterator& rhs) { return lhs.get() < rhs.get(); }
        friend bool operator>(const device_iterator& lhs, const device_iterator& rhs)  { return rhs < lhs; }
        friend bool operator<=(const device_iterator& lhs, const device_iterator& rhs) { return !(rhs < lhs); }
        friend bool operator>=(const device_iterator& lhs, const device_iterator& rhs) { return !(lhs < rhs); }
        
	friend  device_iterator operator+(const device_iterator& it, difference_type n) {
          device_iterator temp= it;
          temp+= n;
          return temp;
        }
        
        friend difference_type operator+(difference_type n, const device_iterator& it) { return it + n; }
        
        friend device_iterator operator-(const device_iterator& it, difference_type n) {
          device_iterator temp= it;
          temp-= n;
          return temp;
        }
        
        friend difference_type operator-(const device_iterator& lhs, const device_iterator& rhs) {
          return lhs.get() - rhs.get();
        }
        
      private:
        
        pointer ptr_;
        
    };*/
}

#include <mathcca/detail/device_matrix.inl>

#endif

