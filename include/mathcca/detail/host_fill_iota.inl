#include <cstddef>
#ifdef _STDPAR
#include <execution>
#include <ranges>
#endif

namespace mathcca {

#ifdef _STDPAR
    template<std::floating_point T>
    void fill_iota(Stdpar, T* first, T* last, const T v) {
      //std::cout << "DEBUG _PARALLELST\n";
      using value_type= T;
      const auto size {static_cast<std::size_t>(last - first)};
      std::ranges::iota_view indices(static_cast<unsigned int>(0),static_cast<unsigned int>(size));
      std::for_each(std::execution::par_unseq,indices.begin(),indices.end(),[&](auto i) { first[i]= v + static_cast<value_type>(i); });
    }
#endif
    template<std::floating_point T>
    void fill_iota(Omp, T* first, T* last, const T v) {
      //std::cout << "DEBUG NO _STDPAR\n";
      using value_type= T;
      const auto size {static_cast<std::size_t>(last - first)};
      #pragma omp prallel for default(shared) 
      for (std::size_t i= 0; i < size; ++i) {
        first[i]= v + static_cast<value_type>(i);
      //std::cout << "DEBUG NO _STDPAR\n";
      }
    }

}


