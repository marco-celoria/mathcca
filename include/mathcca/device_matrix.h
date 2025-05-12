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
    
    
    template<std::floating_point T, typename Allocator, typename Execution>
    class host_matrix;
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    __global__ void addTo_kernel(T* __restrict accululator, const T* __restrict to_be_op, const std::size_t size);
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    __global__ void subTo_kernel(T* __restrict accululator, const T* __restrict to_be_op, const std::size_t size);
    
    template<std::floating_point T, unsigned int THREAD_BLOCK_DIM>
    __global__ void mulTo_kernel(T* __restrict accululator, const T* __restrict to_be_op, const std::size_t size);
   
       template<std::floating_point T, typename Allocator = device_allocator<T>, typename Execution= Cuda>
    class device_matrix : public base_matrix<T, Allocator, Execution > {

      using self= device_matrix;
      typedef base_matrix<T,Allocator, Execution> Parent;

      public:

        typedef typename Parent::size_type  size_type;
        typedef typename Parent::value_type  value_type;
        typedef typename Parent::reference reference;
        typedef typename Parent::const_reference const_reference;
        typedef typename Parent::pointer pointer;
        typedef typename Parent::const_pointer const_pointer;

        using iterator= device_iterator<T, false>;
        using const_iterator= device_iterator<T, true>;
        using traits_alloc = std::allocator_traits<Allocator>;

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
        
        constexpr iterator begin() noexcept { return iterator{Parent::data()}; }
        constexpr iterator end()   noexcept { return iterator{Parent::data() + Parent::size()}; }
        
        constexpr const_iterator cbegin() const noexcept { return const_iterator{const_cast<pointer>(Parent::data())}; }
        constexpr const_iterator cend()   const noexcept { return const_iterator{const_cast<pointer>(Parent::data() + Parent::size())}; }
         
        constexpr const_iterator begin() const  noexcept { return const_iterator{Parent::data()}; }
        constexpr const_iterator end()   const  noexcept { return const_iterator{Parent::data() + Parent::size()}; }
       
        constexpr static auto tol() noexcept { 
          if constexpr (std::is_same_v<value_type, double>) {
            return 1e-5;
          } else { 
            return static_cast<float>(1e-2);
          }
        }
        
        auto toHost (cudaStream_t stream= 0) const {
          host_matrix<T> hostA(Parent::num_rows(), Parent::num_cols());
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
          addTo_kernel<value_type, threads><<<dimGrid, dimBlock>>>(Parent::data(), rhs.data(), size);
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
          subTo_kernel<value_type, threads><<<dimGrid, dimBlock>>>(Parent::data(), rhs.data(), size);
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
          mulTo_kernel<value_type, threads><<<dimGrid, dimBlock>>>(Parent::data(), rhs.data(), size);
          return *this;
        }
        
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

