#ifdef _PARALLELSTL
#include <execution>
#include <ranges>
#endif

#include <cstddef>
#include <mathcca/host_iterator.h>

namespace mathcca {

  namespace algocca {

    template<std::floating_point T>
    void copy(mathcca::iterator::host_iterator<const T> first, mathcca::iterator::host_iterator<const T> last, mathcca::iterator::host_iterator<T> h_first) {
#ifdef _PARALLELSTL
      //std::cout << "DEBUG _PARALLELSTL\n";
      std::copy(std::execution::par_unseq, first, last, h_first);
#else
      //std::cout << "DEBUG NO _PARALLELSTL\n";
      const auto size {static_cast<std::size_t>(last - first)};
      #pragma omp prallel for default(shared)
      for (std::size_t i= 0; i < size; ++i) {
        h_first[i]= first[i];
      }
#endif
    }
  }

}
