#ifndef HOST_ALLOCATOR_H_
#define HOST_ALLOCATOR_H_
#pragma once

#include <cstddef>  // std::size_t
#include <concepts> // std::floating_point

#include <mathcca/execution_policy.h>

namespace mathcca {
        
  template<std::floating_point T>
  class host_allocator {
          
    public:
           
      using value_type= T;
      using reference= T&;
      using const_reference= const T&;
      using execution= Omp;

      host_allocator()= default;
         
      host_allocator(const host_allocator& )= default;
         
      value_type* allocate(std::size_t size) {
        value_type* result{nullptr};
        result= new T[size]{};
        return result;
      }  
         
      void deallocate(value_type* ptr, size_t) {
        delete[] ptr;
      }  
  };      

  template<std::floating_point T1, std::floating_point T2>
  bool operator==(const host_allocator<T1>& lhs, const host_allocator<T2>& rhs) {
    return true;
  }

  template<std::floating_point T1, std::floating_point T2>
  bool operator!=(const host_allocator<T1>& lhs, const host_allocator<T2>& rhs) {
    return !(lhs == rhs);
  }

}


#endif



