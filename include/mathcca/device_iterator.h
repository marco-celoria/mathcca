#ifndef DEVICE_ITERATOR_H_
#define DEVICE_ITERATOR_H_
#pragma once

#include <concepts>  // std::floating_point
#include <cstddef>  // for std::ptrdiff_t
#include <iterator> // for std::random_access_iterator_tag
#include <mathcca/detail/base_iterator.h> 

namespace mathcca {
    
  class device_iterator_tag{};
  
  template<std::floating_point T, bool IsConst>
  class device_iterator : public detail::base_iterator<T, IsConst> {
    
    private:
        
      typedef detail::base_iterator<T, IsConst> Parent;
      
    public:
    
      typedef typename Parent::value_type value_type;
      typedef typename Parent::difference_type difference_type;
      typedef typename Parent::pointer pointer;
      typedef typename Parent::const_pointer const_pointer;
      typedef typename Parent::reference reference;
      typedef typename Parent::const_reference const_reference;
      typedef typename Parent::iterator_category iterator_category;
      
      using iterator_system = device_iterator_tag;
      
      device_iterator() : Parent() {} 
      
      explicit device_iterator(pointer x) : Parent(x) {}
      
      device_iterator(const device_iterator& x) : Parent(x) {};
      
      template<bool IsConst_ = IsConst, class = std::enable_if_t<IsConst_> >
      device_iterator(const device_iterator<T, false>& rhs) : Parent(rhs) {}
        
      template<bool IsConst_ = IsConst, class = std::enable_if_t<IsConst_> >
      device_iterator& operator=(const device_iterator<T, false>& rhs) { Parent::operator=(rhs); return *this; }
      
  };
  
  template<std::floating_point T, bool IsConst>
  bool operator==(const device_iterator<T, IsConst>& x, const device_iterator<T, IsConst>& y) { return x.get() == y.get(); }
      
  template<std::floating_point T, bool IsConst>
  bool operator!=(const device_iterator<T, IsConst>& x, const device_iterator<T, IsConst>& y) { return !(x == y); }
      
  template<std::floating_point T, bool IsConst>
  bool operator<(const device_iterator<T, IsConst>& lhs, const device_iterator<T, IsConst>& rhs) { return lhs.get() < rhs.get(); }
      
  template<std::floating_point T, bool IsConst>
  bool operator>(const device_iterator<T, IsConst>& lhs, const device_iterator<T, IsConst>& rhs)  { return rhs < lhs; }
      
  template<std::floating_point T, bool IsConst>
  bool operator<=(const device_iterator<T, IsConst>& lhs, const device_iterator<T, IsConst>& rhs) { return !(rhs < lhs); }
      
  template<std::floating_point T, bool IsConst>
  bool operator>=(const device_iterator<T, IsConst>& lhs, const device_iterator<T, IsConst>& rhs) { return !(lhs < rhs); }
      
  template<std::floating_point T, bool IsConst>
  device_iterator<T, IsConst> operator+(const device_iterator<T, IsConst>& it, const typename device_iterator<T, IsConst>::difference_type n) {
    device_iterator tmp{it};
    tmp+= n;
    return tmp;
  }

  template<std::floating_point T, bool IsConst>
  typename device_iterator<T, IsConst>::difference_type operator+(const typename device_iterator<T, IsConst>::difference_type n, const device_iterator<T, IsConst>& it) {
    return it + n;
  }
  
  template<std::floating_point T, bool IsConst>
  device_iterator<T, IsConst> operator-(const device_iterator<T, IsConst>& it, const typename device_iterator<T, IsConst>::difference_type n) {
    device_iterator tmp{it};
    tmp-= n;
    return tmp;
  }

  template<std::floating_point T, bool IsConst>
  typename device_iterator<T,IsConst>::difference_type operator-(const device_iterator<T, IsConst>& lhs, const device_iterator<T, IsConst>& rhs) {
    return lhs.get() - rhs.get();
  }

}

#endif
