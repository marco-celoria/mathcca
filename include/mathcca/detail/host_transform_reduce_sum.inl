#include <cstddef>
#ifdef _STDPAR
#include <execution>
#include <ranges>
#endif

namespace mathcca {

#ifdef _STDPAR
    template<std::floating_point T, typename UnaryFunction>
    T transform_reduce_sum(Stdpar, const T* first, const T* last, UnaryFunction unary_op, const T init) {
      std::cout << "DEBUG _STDPAR\n";
      return std::transform_reduce(std::execution::par, first, last, init, std::plus<T>(), unary_op());
    }
#endif
    template<std::floating_point T, typename UnaryFunction>
    T transform_reduce_sum(Omp, const T* first, const T* last, UnaryFunction unary_op, const T init) {
      std::cout << "DEBUG NO _STDPAR\n";
      using value_type= T;
      const auto size {static_cast<std::size_t>(last - first)};
      auto res{static_cast<T>(init)};
      #pragma omp prallel for default(shared) reduction(+:res)
      for (std::size_t i= 0; i < size; ++i) {
        res+= unary_op(first[i]);
      }
      return res;
    }

}

