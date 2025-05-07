#ifdef _PARALLELSTL
#include <execution>
#include <ranges>
#endif

#ifdef _OPENMP
 #include<omp.h>
#endif
namespace mathcca {

  namespace algocca {

    template<Arithmetic T, typename UnaryFunction>
    T transform_reduce_sum(mathcca::iterator::host_iterator<const T> first, mathcca::iterator::host_iterator<const T> last, UnaryFunction unary_op, const T init) {
#ifdef _PARALLELSTL
      return std::transform_reduce(std::execution::par, first, last, init, std::plus<T>(), unary_op());
#else
      using value_type= T;
      using size_type= std::size_t;
      const auto size {static_cast<std::size_t>(last - first)};
      auto res{static_cast<value_type>(init)};
      #pragma omp parallel for reduction(+:sum) default(shared)
      for (size_type i= 0; i < size; ++i) {
        res+= unary_op(first[i]);
      }
      return res;
#endif
    }

  }

}
