#include <random>

#ifdef _PARALG
#include <execution>
#include <ranges>
#endif

#ifdef _OPENMP
 #include<omp.h>
#endif


#include <cstddef>
namespace mathcca {

    template<std::floating_point T>
      inline static T Uniform(T min, T max) {
      static thread_local std::mt19937 generator{std::random_device{}()};
      std::uniform_real_distribution<T> distribution{min,max};
      return distribution(generator);
    }

#ifdef _PARALG
    template<std::floating_point T>
    void fill_rand(Stdpar, T* first, T* last) {  
    const auto size {static_cast<std::size_t>(last - first)};
      //std::cout << "DEBUG _PARALG\n"; 
      std::ranges::iota_view r(static_cast<unsigned int>(0),static_cast<unsigned int>(size));
      std::for_each(std::execution::par_unseq,r.begin(), r.end(), [&](auto i) {first[i] = Uniform(static_cast<T>(0), static_cast<T>(1));});
    }

#endif
    template<std::floating_point T>
    void fill_rand(Omp, T* first, T* last) {
      //std::cout << "DEBUG NO _PARALG\n"; 
    const auto size {static_cast<std::size_t>(last - first)};
      std::random_device rd;
      #pragma omp parallel default(shared)
      {
        #ifdef _OPENMP
         std::seed_seq seed_s{static_cast<int>(rd()), omp_get_thread_num()};
        #else
         std::seed_seq seed_s{static_cast<int>(rd()), 0};
        #endif
        std::mt19937 generator(seed_s);
        std::uniform_real_distribution<T> uniform(static_cast<T>(0), static_cast<T>(1));
        #pragma omp for 
        for (std::size_t i = 0; i < size; ++i) {
          first[i] = uniform(generator);
        }
      }
    }


}



