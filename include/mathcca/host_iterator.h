#ifndef HOST_ITERATOR_H_
#define HOST_ITERATOR_H_
#pragma once
#include <cstddef>  // for std::ptrdiff_t
#include <iterator> // for std::random_access_iterator_tag
    
namespace mathcca {

  namespace iterator {

    template <typename T, bool IsConst>
    class host_iterator {
      public:
        using value_type= T;
        using difference_type= std::ptrdiff_t;
        using pointer= T*;
        using reference= T&;
        using const_reference=const T&;

        host_iterator() : ptr_{nullptr} {}
        explicit host_iterator(pointer x) : ptr_{x} {}

        host_iterator(const host_iterator&)= default;

        template<bool IsConst_ = IsConst, class = std::enable_if_t<IsConst_>>
        host_iterator(const host_iterator<T,false>& rhs) : ptr_(rhs.ptr_) {}  // OK


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

      private:
        pointer ptr_;
    };

   /* 
     template<typename T>
    bool operator==(const host_iterator<T>& x, const host_iterator<T>& y) { return x.get() == y.get(); }

    template<typename T>
    bool operator!=(const host_iterator<T>& x, const host_iterator<T>& y) { return !(x == y); }

    template<typename T>
    bool operator<(const host_iterator<T>& lhs, const host_iterator<T>& rhs) { return lhs.get() < rhs.get(); }

    template<typename T>
    bool operator>(const host_iterator<T>& lhs, const host_iterator<T>& rhs)  { return rhs < lhs; }

    template<typename T>
    bool operator<=(const host_iterator<T>& lhs, const host_iterator<T>& rhs) { return !(rhs < lhs); }

    template<typename T>
    bool operator>=(const host_iterator<T>& lhs, const host_iterator<T>& rhs) { return !(lhs < rhs); }

    template<typename T>
    host_iterator<T> operator+(const host_iterator<T>& it, typename host_iterator<T>::difference_type n) {
      host_iterator temp= it;
      temp+= n;
      return temp;
    }

    template<typename T>
    typename host_iterator<T>::difference_type operator+(typename host_iterator<T>::difference_type n, const host_iterator<T>& it) { return it + n; }

    template<typename T>
    host_iterator<T> operator-(const host_iterator<T>& it, typename host_iterator<T>::difference_type n) {
      host_iterator temp= it;
      temp-= n;
      return temp;
    }

    template<typename T>
    typename host_iterator<T>::difference_type operator-(const host_iterator<T>& lhs, const host_iterator<T>& rhs) {
      return lhs.get() - rhs.get();
    }
 */ 
  }
}

#endif
