#include <cstddef>
#ifdef _PARALG
#include <execution>
#include <ranges>
#endif

namespace mathcca {

#ifdef _PARALG
    template<std::floating_point T>
    T reduce_sum(StdPar, const T* first, const T* last, const T init) {
      std::cout << "DEBUG _PARALG\n";
      return std::reduce(std::execution::par_unseq, first, last, static_cast<T>(init), std::plus<T>());
    }
#endif
    template<std::floating_point T>
    T reduce_sum(Omp, const T* first, const T* last, const T init) {
      std::cout << "DEBUG NO _PARALG\n";
      using value_type= T;
      const auto size {static_cast<std::size_t>(last - first)};
      auto res{static_cast<T>(init)};
      #pragma omp prallel for default(shared) reduction(+:res)
      for (std::size_t i= 0; i < size; ++i) {
        res+= first[i];
      }
      return res;
    }

}


