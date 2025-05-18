#ifndef BASE_MATRIX_H_
#define BASE_MATRIX_H_
#pragma once

#include <cstddef>   // std::size_t
#include <concepts>  // std::floating_point
#include <iostream>  // std::cout 
#include <memory>    // std::allocator_traits
#include <type_traits> // std::is_same

#include <mathcca/detail/base_iterator.h>
#include <mathcca/detail/copy_impl.h>
#include <mathcca/detail/fill_const_impl.h>
        
namespace mathcca {
        
  namespace detail { 
       
    template<std::floating_point T, typename Allocator, typename Execution>
    class base_matrix {
         
      using self= base_matrix; 
        
      public:
        
        using value_type= T;
        using size_type= std::size_t;
        using reference= T&;
        using const_reference= const T&;
        using pointer= T*;
        using const_pointer= const T*;
        using traits_alloc= std::allocator_traits<Allocator>;
        using iterator= base_iterator<T, false>;
        using const_iterator= base_iterator<T, true>;
        
        base_matrix(Allocator a) : allocator{std::move(a)} {}
        
        constexpr base_matrix(size_type r, size_type c) : num_rows_{r}, num_cols_{c} {
          std::cout << "custom ctor\n";
          data_ = traits_alloc::allocate(allocator, size());    	
        }
        
        constexpr base_matrix(size_type r, size_type c, const_reference v) : base_matrix(r, c) {
          std::cout << "(delegating ctor)\n";
          fill_const(Execution(), data(), data() + size(), v);
        } 
         
        constexpr ~base_matrix() {
          std::cout << "dtor\n"; 
          if (data_) {
            traits_alloc::deallocate(allocator, data_, size());  
            data_= nullptr; 
          }
          num_cols_= 0;
          num_rows_= 0;
        }
        
        constexpr base_matrix(self&& m):num_rows_{std::move(m.num_rows_)}, num_cols_{std::move(m.num_cols_)}, data_{std::move(m.data_)} {
          std::cout << "move ctor\n";
          m.data_= nullptr;
          m.num_cols_= 0;
          m.num_rows_= 0;
        }
        
        constexpr base_matrix(const self& m) : base_matrix{m.num_rows_, m.num_cols_} {
          std::cout << "copy ctor\n";
/*#ifdef __CUDACC__
	  if constexpr (std::is_same_v<Execution, Cuda>) { 
            
            copy(CudaDtoDcpy(), m.data(), m.data() + m.size(), data());
          }
#endif
	  if constexpr (std::is_same_v<Execution, Omp>) {
	    copy(Omp(), m.data(), m.data() + m.size(), data());
	  }*/
	  copy(Execution(), m.data(), m.data() + m.size(), data());
	}
        
        constexpr base_matrix<T, Allocator, Execution>& operator=(base_matrix&& rhs) {
          std::cout << "move assignment\n";
          num_rows_= std::move(rhs.num_rows_);
          num_cols_= std::move(rhs.num_cols_);
          if (data_) {
            traits_alloc::deallocate(allocator, data_, size());	    
          }  
          data_= std::move(rhs.data_);
          rhs.data_= nullptr;
          rhs.num_cols_= 0;
          rhs.num_rows_= 0;
          return *this;
        }
        
        constexpr base_matrix<T, Allocator, Execution>& operator=(const base_matrix& rhs) {
          if (this != &rhs) {
            std::cout << "copy assignment (\n";
            if (this->size() != rhs.size()) {
              auto tmp{rhs};            // use copy ctor
              (*this)= std::move(tmp);  // finally move assignment
            }
            else {
              num_rows_= rhs.num_rows_;
              num_cols_= rhs.num_cols_;
/*#ifdef __CUDACC__
	      if constexpr (std::is_same_v<Execution, Cuda>) { 
                copy(CudaDtoDcpy(), rhs.data(), rhs.data() + rhs.size(), data());
              }
#endif
	      if constexpr (std::is_same_v<Execution, Omp>) {
	        copy(Omp(), rhs.data(), rhs.data() + rhs.size(), data());
	      }*/
	      copy(Execution(), rhs.data(), rhs.data() + rhs.size(), data());
            }
            std::cout << ")\n";
          }
          return *this;
        }
        
#ifdef __CUDACC__
        __host__ __device__ 
#endif  
        constexpr size_type num_rows() const noexcept { return num_rows_; }
        
#ifdef __CUDACC__
        __host__ __device__ 
#endif  
        constexpr size_type num_cols() const noexcept { return num_cols_; }
        
#ifdef __CUDACC__
        __host__ __device__ 
#endif  
        constexpr size_type size() const noexcept { return num_rows_ * num_cols_; }
        
#ifdef __CUDACC__
        __host__ __device__ 
#endif  
        constexpr pointer data() noexcept { return data_; } 
        
#ifdef __CUDACC__
        __host__ __device__ 
#endif  
        constexpr const_pointer data() const noexcept { return data_; } 
        
#ifdef __CUDACC__
        __host__ __device__ 
#endif  
        constexpr reference operator[] (size_type i) noexcept {return data_[i]; }
        
#ifdef __CUDACC__
       	__host__ __device__ 
#endif  
        constexpr const_reference operator[] (size_type i) const noexcept {return  data_[i]; }
        
        constexpr iterator begin() noexcept { return iterator{ data()}; }
        constexpr iterator end()   noexcept { return iterator{ data() +  size()}; }
        
        constexpr const_iterator begin()  const noexcept { return const_iterator{ data()}; }
        constexpr const_iterator end()    const noexcept { return const_iterator{ data() +  size()}; }
        
        constexpr const_iterator cbegin() const noexcept { return const_iterator{ const_cast<pointer>(data())}; }
        constexpr const_iterator cend()   const noexcept { return const_iterator{ const_cast<pointer>(data() +  size()) }; }
        
        constexpr static auto tol() noexcept {
          if constexpr (std::is_same_v<value_type, float>) {
            return static_cast<float>(1e-2);
          } else {
            return static_cast<double>(1e-5);
          }
        }
        
      private:
        
        size_type num_rows_{0};
        size_type num_cols_{0};
        pointer data_{nullptr};
        Allocator allocator;
        
    };  
    
  }  
    
}   


#endif



