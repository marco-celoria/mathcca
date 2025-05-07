#include <cstddef>

namespace mathcca {

  namespace algocca {
        
    template<std::floating_point T>
    void fill_const(mathcca::iterator::host_iterator<T> first, mathcca::iterator::host_iterator<T> last, const T v) {
#ifdef _PARALLELSTL
      //std::cout << "DEBUG _PARALLELST\n";
      std::fill(std::execution::par_unseq, first, last, v);
#else
      //std::cout << "DEBUG NO _PARALLELSTL\n";
      const std::size_t size= static_cast<std::size_t>(last - first);
      #pragma omp prallel for default(shared)
      for (std::size_t i= 0; i < size; ++i) {
        first[i]= v;
      }
#endif
    }

  }

}



