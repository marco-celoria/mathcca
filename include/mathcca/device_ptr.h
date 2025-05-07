#ifndef DEVICE_PTR_H_
#define DEVICE_PTR_H_
#pragma once

#include <cstddef>
#include <iterator>
#include <type_traits>

#include <iostream>
#include <cuda_runtime.h>
#include <mathcca/device_helper.h>

namespace mathcca {

  namespace ptr {

    template <typename T[]>
    class device_ptr {

      using self= device_ptr;

      public:

        using value_type= T;
        using reference= T&;
        using const_reference= const T&;
        using pointer= T*;
        using const_pointer= const T*;

        // Constructor
        explicit device_ptr(pointer ptr = nullptr) : ptr_{ptr} { std::cout << "device_ptr ctor\n"; }
        
        // Destructor
        ~device_ptr() { 
          std::cout << "device_ptr dtor\n";
          if (ptr_) {
            checkCudaErrors(cudaFree(ptr_));
            ptr_ = nullptr;
          }
        }
        
        // Copy constructor and assignment operator deleted
        device_ptr(const self&) = delete;
        device_ptr& operator=(const device_ptr&) = delete;
        
        // Move constructor and assignment operator
        device_ptr(self&& other) noexcept : ptr_{other.ptr_} {
          std::cout << "device_ptr move ctor\n";       	
          other.ptr_ = nullptr; 
        }
        
        device_ptr& operator=(self&& other) noexcept {
          std::cout << "device_ptr move assignment\n";
          if (this != &other) {
            if (ptr_) 
              checkCudaErrors(cudaFree(ptr_));
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
          }
          return *this;
        }
        
        // Member functions
        pointer get() const noexcept { return ptr_; }
        
        /*  Releases ownership of its stored pointer, 
         *  by returning its value and replacing it with a null pointer.
         *  This call does not destroy the managed object, 
         *  but the unique_ptr object is released from the responsibility of deleting the object. 
         *  Some other entity must take responsibility for deleting the object at some point.
         *  To force the destruction of the object pointed, 
         *  either use member function reset or perform an assignment operation on it.
         */ 

        pointer release() noexcept {
          pointer tmp = ptr_;
          ptr_ = nullptr;
          return tmp;
        }
        
        void reset(pointer ptr = nullptr) noexcept {
          if (ptr_) 
            checkCudaErrors(cudaFree(ptr_));
          ptr_= ptr;
        }
       
        void swap(self& other) noexcept {
          const auto tmp_ptr = std::move(other.ptr_);
          other.ptr_ = std::move(ptr_);
          ptr_ = std::move(tmp_ptd);
        }

        typename std::add_lvalue_reference<T>::type operator*() const noexcept { return *ptr_; }
        
        pointer operator->() const noexcept { return ptr_; }
        
        explicit operator bool() const noexcept { return ptr_ != nullptr; }
        
        reference operator[] (std::size_t index) const noexcept { return *(ptr_ + index); }
        
      private:
         
        pointer ptr_{nullptr};

    };

    template <typename T[]>
    bool operator==(const device_ptr<T[]>& lhs, const device_ptr<T[]>& rhs) {
      return (lhs.get() == rhs.get());
    }

  }

}


#endif



