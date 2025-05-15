#ifndef BASE_ITERATOR_H_
#define BASE_ITERATOR_H_
#pragma once
#include <cstddef>  // for std::ptrdiff_t
#include <iterator> // for std::random_access_iterator_tag
    
namespace mathcca {
  	
  template<std::floating_point T, bool IsConst>
  class base_iterator {
    public:
      using value_type= T;
      using difference_type= std::ptrdiff_t;
      using pointer= T*;
      using const_pointer= const T*;
      using reference= T&;
      using const_reference=const T&;
      using iterator_category= std::contiguous_iterator_tag;
      
      base_iterator() : ptr_{nullptr} {}
      
      explicit base_iterator(pointer x) : ptr_{x} {}
      
      base_iterator(const base_iterator&)= default;
      
      template<bool IsConst_ = IsConst, class = std::enable_if_t<IsConst_>>
      base_iterator(const base_iterator<T, false>& rhs) : ptr_(rhs.get()) {}
       
      template<bool IsConst_ = IsConst, class = std::enable_if_t<IsConst_>>
      base_iterator& operator=(const base_iterator<T, false>& rhs) { ptr_ = rhs.get(); return *this; }
      
      template <bool Q = IsConst>
      typename std::enable_if_t<Q, const_reference> operator*() const noexcept { return *ptr_; }
      
      template <bool Q = IsConst>
      typename std::enable_if_t<!Q, reference>      operator*() const noexcept { return *ptr_; }
        
      template <bool Q = IsConst>
      typename std::enable_if_t<Q, const_reference> operator[](difference_type n) const noexcept { return *(ptr_ + n); }
        
      template <bool Q = IsConst>
      typename std::enable_if_t<!Q, reference>      operator[](difference_type n) const noexcept { return *(ptr_ + n); }
      
      pointer operator->()  const noexcept { return ptr_; }
      
      pointer get()         const noexcept { return ptr_; }
      
      base_iterator& operator++()   { ++ptr_; return *this; }
      
      base_iterator operator++(int) { auto tmp= *this; ++(*this); return tmp; }
      
      base_iterator& operator--()   { --ptr_; return *this; }
      
      base_iterator operator--(int) { auto tmp= *this; --(*this); return tmp; }
      
      base_iterator& operator+=(difference_type n) { ptr_+= n; return *this; }
      
      base_iterator& operator-=(difference_type n) { ptr_-= n; return *this; }
      
    private:
        
      pointer ptr_;
      
  };
  
  template<std::floating_point T, bool IsConst>
  bool operator==(const base_iterator<T, IsConst>& lhs, const base_iterator<T, IsConst>& rhs) { return lhs.get() == rhs.get(); }
      
  template<std::floating_point T, bool IsConst>
  bool operator!=(const base_iterator<T, IsConst>& lhs, const base_iterator<T, IsConst>& rhs) { return !(lhs == rhs); }
      
  template<std::floating_point T, bool IsConst>
  bool operator<(const base_iterator<T, IsConst>& lhs,  const base_iterator<T, IsConst>& rhs) { return lhs.get() < rhs.get(); }
      
  template<std::floating_point T, bool IsConst>
  bool operator>(const base_iterator<T, IsConst>& lhs,  const base_iterator<T, IsConst>& rhs)  { return rhs < lhs; }
      
  template<std::floating_point T, bool IsConst>
  bool operator<=(const base_iterator<T, IsConst>& lhs, const base_iterator<T, IsConst>& rhs) { return !(rhs < lhs); }
      
  template<std::floating_point T, bool IsConst>
  bool operator>=(const base_iterator<T, IsConst>& lhs, const base_iterator<T, IsConst>& rhs) { return !(lhs < rhs); }
      
  template<std::floating_point T, bool IsConst>
  base_iterator<T, IsConst> operator+(const base_iterator<T, IsConst>& it, const typename base_iterator<T, IsConst>::difference_type n) {
    base_iterator tmp{it};
    tmp+= n;
    return tmp;
  }
  
  template<std::floating_point T, bool IsConst>
  typename base_iterator<T, IsConst>::difference_type operator+(const typename base_iterator<T, IsConst>::difference_type n, const base_iterator<T, IsConst>& it) {
    return it + n; 
  }
  
  template<std::floating_point T, bool IsConst>
  base_iterator<T, IsConst> operator-(const base_iterator<T, IsConst>& it, const typename base_iterator<T, IsConst>::difference_type n) {
    base_iterator tmp{it};
    tmp-= n;
    return tmp;
  }
  
  template<std::floating_point T, bool IsConst>
  typename base_iterator<T, IsConst>::difference_type operator-(const base_iterator<T, IsConst>& lhs, const base_iterator<T, IsConst>& rhs) {
    return lhs.get() - rhs.get();
  }

}

#endif
