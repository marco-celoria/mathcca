#include <cstddef>

namespace mathcca {

  namespace algocca {
        
    template<std::floating_point T>
    void fill_iota(mathcca::iterator::host_iterator<T> first, mathcca::iterator::host_iterator<T> last, const T& v) {
      using value_type= T;
      const auto size {static_cast<std::size_t>(last - first)};
#ifdef _PARALLELSTL
      //std::cout << "DEBUG _PARALLELST\n"; 
      std::ranges::iota_view indices(static_cast<unsigned int>(0),static_cast<unsigned int>(size));
      std::for_each(std::execution::par_unseq,indices.begin(),indices.end(),[&](auto i) { first[i]= v + static_cast<value_type>(i); });

#else
      //std::cout << "DEBUG NO _PARALLELSTL\n";
      #pragma omp prallel for default(shared) 
      for (std::size_t i= 0; i < size; ++i) {
        first[i]= v + static_cast<value_type>(i);
      }
#endif
    }

  }

}



