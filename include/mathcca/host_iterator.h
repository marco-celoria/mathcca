#ifndef HOST_ITERATOR_H_
#define HOST_ITERATOR_H_
#pragma once
#include <cstddef>  // for std::ptrdiff_t
#include <iterator> // for std::random_access_iterator_tag
#include <mathcca/base_iterator.h> 

namespace mathcca {
    
  class host_iterator_tag{};
  
  template<std::floating_point T, bool IsConst>
  class host_iterator : public base_iterator<T, IsConst> {
    
    private:
        
      typedef base_iterator<T, IsConst> Parent;
      
    public:
    
      typedef typename Parent::value_type value_type;
      typedef typename Parent::difference_type difference_type;
      typedef typename Parent::pointer pointer;
      typedef typename Parent::const_pointer const_pointer;
      typedef typename Parent::reference reference;
      typedef typename Parent::const_reference const_reference;
      typedef typename Parent::iterator_category iterator_category;
      
      using iterator_system= host_iterator_tag;
      
      host_iterator() : Parent() {} 
      
      explicit host_iterator(pointer x) : Parent(x) {}
      
      host_iterator(const host_iterator& x) : Parent(x) {};
      
      template<bool IsConst_ = IsConst, class = std::enable_if_t<IsConst_> >
      host_iterator(const host_iterator<T, false>& rhs) : Parent(rhs) {}
        
      template<bool IsConst_ = IsConst, class = std::enable_if_t<IsConst_> >
      host_iterator& operator=(const host_iterator<T, false>& rhs) { Parent::operator=(rhs); return *this; } 
      
  };
  
  template<std::floating_point T, bool IsConst>
  bool operator==(const host_iterator<T, IsConst>& x, const host_iterator<T, IsConst>& y) { return x.get() == y.get(); }
      
  template<std::floating_point T, bool IsConst>
  bool operator!=(const host_iterator<T, IsConst>& x, const host_iterator<T, IsConst>& y) { return !(x == y); }
      
  template<std::floating_point T, bool IsConst>
  bool operator<(const host_iterator<T, IsConst>& lhs, const host_iterator<T, IsConst>& rhs) { return lhs.get() < rhs.get(); }
      
  template<std::floating_point T, bool IsConst>
  bool operator>(const host_iterator<T, IsConst>& lhs, const host_iterator<T, IsConst>& rhs)  { return rhs < lhs; }
      
  template<std::floating_point T, bool IsConst>
  bool operator<=(const host_iterator<T, IsConst>& lhs, const host_iterator<T, IsConst>& rhs) { return !(rhs < lhs); }
      
  template<std::floating_point T, bool IsConst>
  bool operator>=(const host_iterator<T, IsConst>& lhs, const host_iterator<T, IsConst>& rhs) { return !(lhs < rhs); }
      
  template<std::floating_point T, bool IsConst>
  host_iterator<T, IsConst> operator+(const host_iterator<T, IsConst>& it, const typename host_iterator<T, IsConst>::difference_type n) {
    host_iterator tmp{it};
    tmp+= n;
    return tmp;
  }

  template<std::floating_point T, bool IsConst>
  typename host_iterator<T, IsConst>::difference_type operator+(const typename host_iterator<T, IsConst>::difference_type n, const host_iterator<T, IsConst>& it) {
    return it + n;
  }
  
  template<std::floating_point T, bool IsConst>
  host_iterator<T, IsConst> operator-(const host_iterator<T, IsConst>& it, const typename host_iterator<T, IsConst>::difference_type n) {
    host_iterator tmp{it};
    tmp-= n;
    return tmp;
  }

  template<std::floating_point T, bool IsConst>
  typename host_iterator<T,IsConst>::difference_type operator-(const host_iterator<T, IsConst>& lhs, const host_iterator<T, IsConst>& rhs) {
    return lhs.get() - rhs.get();
  }

}

#endif
