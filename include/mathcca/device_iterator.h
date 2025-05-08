#ifndef DEVICE_ITERATOR_H_
#define DEVICE_ITERATOR_H_
#pragma once
#include <cstddef>  // for std::ptrdiff_t
#include <iterator> // for std::random_access_iterator_tag
    
namespace mathcca {
    class device_iterator_tag{};

    template<std::floating_point T, bool IsConst>
    class device_iterator {
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
        device_iterator(const device_iterator<T, false>& rhs) : ptr_(rhs.get()) {}  // OK
        
	template<bool IsConst_ = IsConst, class = std::enable_if_t<IsConst_>>
        device_iterator& operator=(const device_iterator<T, false>& rhs) { ptr_ = rhs.ptr_; return *this; }

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

      private:

        pointer ptr_;

    };

      template<std::floating_point T,  bool IsConst>
      bool operator==(const device_iterator<T, IsConst>& x, const device_iterator<T, IsConst>& y) { return x.get() == y.get(); }
      
      template<std::floating_point T,  bool IsConst>
      bool operator!=(const device_iterator<T, IsConst>& x, const device_iterator<T, IsConst>& y) { return !(x == y); }
      
      template<std::floating_point T,  bool IsConst>
      bool operator<(const device_iterator<T, IsConst>& lhs, const device_iterator<T, IsConst>& rhs) { return lhs.get() < rhs.get(); }
      
      template<std::floating_point T,  bool IsConst>
      bool operator>(const device_iterator<T, IsConst>& lhs, const device_iterator<T, IsConst>& rhs)  { return rhs < lhs; }
      
      template<std::floating_point T,  bool IsConst>
      bool operator<=(const device_iterator<T, IsConst>& lhs, const device_iterator<T, IsConst>& rhs) { return !(rhs < lhs); }
      
      template<std::floating_point T,  bool IsConst>
      bool operator>=(const device_iterator<T, IsConst>& lhs, const device_iterator<T, IsConst>& rhs) { return !(lhs < rhs); }
      
      template<std::floating_point T,  bool IsConst>
      device_iterator<T, IsConst> operator+(const device_iterator<T, IsConst>& it, typename device_iterator<T,IsConst>::difference_type n) {
          device_iterator temp= it;
          temp+= n;
          return temp;
      }

      template<std::floating_point T,  bool IsConst>
      typename device_iterator<T,IsConst>::difference_type operator+(typename device_iterator<T,IsConst>::difference_type n, const device_iterator<T, IsConst>& it) { return it + n; }

      template<std::floating_point T,  bool IsConst>
      device_iterator<T, IsConst> operator-(const device_iterator<T, IsConst>& it, typename device_iterator<T,IsConst>::difference_type n) {
          device_iterator temp= it;
          temp-= n;
          return temp;
      }

      template<std::floating_point T,  bool IsConst>
      typename device_iterator<T,IsConst>::difference_type operator-(const device_iterator<T, IsConst>& lhs, const device_iterator<T, IsConst>& rhs) {
          return lhs.get() - rhs.get();
      }

}

#endif
