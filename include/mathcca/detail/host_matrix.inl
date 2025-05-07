
#include <cmath>
#include <stdexcept>
#include <type_traits>

#ifdef PAR_UNSEQ
 #include <execution>
#endif

namespace mathcca {

  namespace matricca {

    template<std::floating_point T>
    void swap(host_matrix<T>& a, host_matrix<T>& b) {
      auto tmp= std::move(a);
      a= std::move(b);
      b= std::move(tmp);
    }
    
    template<std::floating_point T>
    constexpr inline bool check_equal_size(const host_matrix<T>& lhs, const host_matrix<T>& rhs) {
      if (lhs.num_rows() == rhs.num_rows() && lhs.num_cols() == rhs.num_cols())
        return true;
      return false;
    }

    template<std::floating_point T>
    constexpr bool operator==(const host_matrix<T>& lhs, const host_matrix<T>& rhs) { 
      using size_type= typename host_matrix<T>::size_type;
      if (!check_equal_size(lhs, rhs))
        return false;
      size_type errs{0};
      const auto size{lhs.size()};
      const auto tol{host_matrix<T>::tol()};
      #pragma omp parallel for reduction(+:errs)  default(shared) 
      for (size_type i= 0; i < size; ++i) {
        if (std::abs(lhs[i] - rhs[i]) > tol) { 
          ++errs;
        }
      }
      if (errs > 0)
        return false;
      return true;
    }

  template<std::floating_point T>
  constexpr host_matrix<T> operator+ (host_matrix<T>&& res, const host_matrix<T>& rhs) {
    std::cout <<"operator+ rvalue\n";
    return std::forward<host_matrix<T>>(res+= rhs);
  }

  template<std::floating_point T>
  constexpr host_matrix<T> operator+ (const host_matrix<T>& lhs, const host_matrix<T>& rhs) {
    std::cout <<"operator+ lvalue\n";
    auto res{lhs};
    return std::forward<host_matrix<T>>(res+= rhs);
  }

  template<std::floating_point T>
  constexpr host_matrix<T> operator- (host_matrix<T>&& res, const host_matrix<T>& rhs) {
    std::cout <<"operator- rvalue\n";
    return std::forward<host_matrix<T>>(res-= rhs);
  }

  template<std::floating_point T>
  constexpr host_matrix<T> operator- (const host_matrix<T>& lhs, const host_matrix<T>& rhs) {
    std::cout <<"operator- lvalue\n";
    auto res{lhs};
    return std::forward<host_matrix<T>>(res-= rhs);
  }
  
  template<std::floating_point T>
  constexpr host_matrix<T> operator* (host_matrix<T>&& res, const host_matrix<T>& rhs) {
    std::cout <<"operator* rvalue\n";
    return std::forward<host_matrix<T>>(res*= rhs);
  }

  template<std::floating_point T>
  constexpr host_matrix<T> operator* (const host_matrix<T>& lhs, const host_matrix<T>& rhs) {
    std::cout <<"operator* lvalue\n";
    auto res{lhs};
    return std::forward<host_matrix<T>>(res*= rhs);
  }

    template<std::floating_point T>
  void print_matrix(const host_matrix<T>& mat) {
    for (std::size_t r= 0; r < mat.num_rows(); ++r) {
      printf ("[");
      for (std::size_t c= 0; c < mat.num_cols(); ++c)
        std::cout <<  mat(r, c) << " ";
      std::cout << "]\n";
    }
    std::cout << "\n";
  }

  }
}

