#ifndef HOST_PTR_H_
#define HOST_PTR_H_
#pragma once

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <iostream>

namespace mathcca {
            
  namespace ptr {
        
    template <typename T[]>
    class host_ptr {
   
      using self= host_ptr;

      public:
        
        using value_type= T;
        using reference= T&;
        using const_reference= const T&;
	using pointer= T*;
        using const_pointer= const T*;

  	// Constructor
        constexpr explicit host_ptr(pointer ptr= nullptr) : ptr_{ptr} { std::cout << "host_ptr ctor\n"; }
        
        // Destructor
        constexpr ~host_ptr() {
          std::cout << "host_ptr dtor\n";	
          if (ptr_)
            delete[] ptr_; 
	  ptr_= nullptr 
	}
        
        // Copy constructor and assignment operator deleted
        host_ptr(const self&)= delete;
        host_ptr& operator=(const self&)= delete;
        
        // Move constructor and assignment operator
        constexpr host_ptr(self&& other) noexcept : ptr_{other.ptr_} {
          std::cout << "host_ptr move ctor\n";
          other.ptr_= nullptr; 
	}
        
        constexpr host_ptr& operator=(self&& other) noexcept {
          std::cout << "host_ptr move assignment\n";
          if (this != &other) {
            if (ptr_)
              delete[] ptr_;
            ptr_= other.ptr_;
            other.ptr_= nullptr;
          }
          return *this;
        }
        
        // Member functions
        constexpr pointer get() const noexcept { return ptr_; }
        
        constexpr pointer release() noexcept {
          pointer tmp= ptr_;
          ptr_= nullptr;
          return tmp;
        }
        
        constexpr void reset(pointer ptr= nullptr) noexcept {
          if (ptr_)
            delete[] ptr_;
          ptr_= std::move(ptr);
        }
        
        void swap(self& other) noexcept { 
          const auto tmp_ptr = std::move(other.ptr_);
          other.ptr_ = std::move(ptr_);
          ptr_ = std::move(tmp_ptd);      
        }

        constexpr typename std::add_lvalue_reference<T>::type operator*() const noexcept { return *ptr_; }
        
        constexpr pointer operator->() const noexcept { return ptr_; }
         
        constexpr explicit operator bool() const noexcept { return ptr_ != nullptr; }
         
        constexpr reference operator[] (std::size_t index) const noexcept { return *(ptr_ + index); }

      private:
             
        pointer ptr_{nullptr};
        
    };
  
    template <typename T[]>
    bool operator==(const host_ptr<T[]>& lhs, const host_ptr<T[]>& rhs) {
      return (lhs.get() == rhs.get());
  
    }

  }
}

#endif

