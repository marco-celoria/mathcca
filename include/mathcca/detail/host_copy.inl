#ifdef _PARALG
#include <execution>
#include <ranges>
#endif

#include <cstddef>

namespace mathcca {

#ifdef _PARALG
    template<std::floating_point T>
    void copy(StdpPar, const T* s_first, const T* s_last, T* d_first) {
      //std::cout << "DEBUG _PARALG\n";
      std::copy(std::execution::par_unseq, s_first, s_last, d_first);
    }
#endif
    template<std::floating_point T> 
    void copy(Omp, const T* s_first, const T* s_last, T* d_first) {
      //std::cout << "DEBUG NO _PARALG\n";
      const auto size {static_cast<std::size_t>(s_last - s_first)};
      #pragma omp prallel for default(shared)
      for (std::size_t i= 0; i < size; ++i) {
        d_first[i]= s_first[i];
      }
    }

}
