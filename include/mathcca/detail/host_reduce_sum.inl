
#ifdef _PARALLELSTL
#include <execution>
#include <ranges>
#endif

#ifdef _OPENMP
 #include<omp.h>
#endif
namespace mathcca {

  namespace algocca {
    
    template<std::floating_point T>
    T reduce_sum(mathcca::iterator::host_iterator<const T> first, mathcca::iterator::host_iterator<const T> last, const T init) {
#ifdef _PARALLELSTL
      //std::cout << "DEBUG _PARALLELSTL\n"; 
      return std::reduce(std::execution::par_unseq, first, last, static_cast<T>(init), std::plus<T>());
#else
      //std::cout << "DEBUG NO _PARALLELSTL\n"; 
      using value_type= T;
      const auto size {static_cast<std::size_t>(last - first)};
      auto res{static_cast<T>(init)};
      #pragma omp prallel for default(shared) reduction(+:res) 
      for (std::size_t i= 0; i < size; ++i) {
        res+= first[i];
      }
      return res;
#endif
    }

  }

}

