#include <cstddef>
#ifdef _STDPAR
#include <execution>
#include <ranges>
#endif

namespace mathcca {
        
#ifdef _STDPAR
    template<std::floating_point T>
    void fill_const(Stdpar, T* first, T* last, const T v) {
      //std::cout << "DEBUG _PARALLELST\n";
      std::fill(std::execution::par_unseq, first, last, v);
    }
#endif
    template<std::floating_point T>
    void fill_const(Omp, T* first, T* last, const T v) {    
      //std::cout << "DEBUG NO _STDPAR\n";
      const std::size_t size= static_cast<std::size_t>(last - first);
      #pragma omp prallel for default(shared)
      for (std::size_t i= 0; i < size; ++i) {
        first[i]= v;
      }
    }

}



