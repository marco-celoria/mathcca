


namespace mathcca {
  namespace matricca {


	  #ifdef _PARALLELSTL

  template<typename T>
  struct Square {
    T operator()(const T &x) const {
      return x * x;
    }
  };


  template<std::floating_point T>
  constexpr decltype(auto) frobenius_norm_Pstl(const host_matrix<T>& x) {
    std::cout << "Parallel STL frobenius norm\n";
    return std::sqrt(std::transform_reduce(std::execution::par, x.cbegin(), x.cend(), static_cast<T>(0), std::plus<T>(), Square<T>()));
  }
#endif


#ifdef _MKL
  template<std::floating_point T>
  constexpr decltype(auto) frobenius_norm_Mkl(const host_matrix<T>& x) {
    std::cout << "Mkl frobenius norm\n";
    T result;
    const auto size{x.size()};
    const auto incx{1};
    if constexpr(std::is_same_v<T, double>) {
    //double cblas_dnrm2 (const MKL_INT n, const double *x, const MKL_INT incx);
    result= cblas_dnrm2(size, x.data(), incx);
    }
    else {
    //float cblas_snrm2 (const MKL_INT n, const float *x, const MKL_INT incx);
    result= cblas_snrm2(size, x.data(), incx);
    }

    return result;
  }
#endif

    template<std::floating_point T>
  constexpr decltype(auto) frobenius_norm_Base (const host_matrix<T>& A) {
    std::cout << "Base frobenius norm\n";
    using size_type= typename host_matrix<T>::size_type;
    //using value_type= T;
    const auto size{A.size()};
    auto sum{static_cast<T>(0)};
    #pragma omp parallel for reduction(+:sum) default(shared)
    for (size_type i= 0; i < size; ++i) {
        sum+= A[i] * A[i];
    }
    return std::sqrt(sum);
  }

  }
}
