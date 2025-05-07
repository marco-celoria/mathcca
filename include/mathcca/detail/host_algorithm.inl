#include <random>

//#ifdef _PARALLELSTL
#include <execution>
#include <ranges>
//#endif

#ifdef _OPENMP
 #include<omp.h>
#endif
namespace algocca {


  namespace host {


    template<std::floating_point T>
    void fill_const(T* first, T* last, const T& v) {
#ifdef _PARALLELSTL
      //std::cout << "DEBUG _PARALLELST\n";  
      std::fill(std::execution::par_unseq, first, last, v);
#else
      //std::cout << "DEBUG NO _PARALLELSTL\n";      
      const auto size {static_cast<std::size_t>(last - first)};
      #pragma omp prallel for default(shared) 
      for (std::size_t i= 0; i < size; ++i) {
        first[i]= v;
      } 
#endif
    }


    template<std::floating_point T>
    void fill_iota(T* first, T* last, const T& v) {
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


    template<std::floating_point T>
      inline static T Uniform(T min, T max) {
      static thread_local std::mt19937 generator{std::random_device{}()};
      std::uniform_real_distribution<T> distribution{min,max};
      return distribution(generator);
    }

    template<std::floating_point T>
    void fill_rand(T* first, T* last) {
      const auto size {static_cast<std::size_t>(last - first)};    
#ifdef _PARALLELSTL
      //std::cout << "DEBUG _PARALLELSTL\n"; 
      std::ranges::iota_view r(static_cast<unsigned int>(0),static_cast<unsigned int>(size));
      std::for_each(std::execution::par_unseq,r.begin(), r.end(), [&](auto i) {first[i] = Uniform(static_cast<T>(0), static_cast<T>(1));});
#else
      //std::cout << "DEBUG NO _PARALLELSTL\n"; 
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
#endif
    }


    template<std::floating_point T>
    void copy(const T* first, const T* last, T* h_first) {
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


    template<Arithmetic T>
    T reduce_sum(const T* first, const T* last, const T init) {
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

    
    template<Arithmetic T, typename UnaryFunction>
    T transform_reduce_sum(const T* first, const T* last, UnaryFunction unary_op, const T init) {
#ifdef _PARALLELSTL
      return std::transform_reduce(std::execution::par, first, last, init, std::plus<T>(), unary_op());
#else
      using value_type= T;
      using size_type= typename host_matrix<T>::size_type;
      const auto size {static_cast<std::size_t>(last - first)};
      auto res{static_cast<value_type>(init)};
      #pragma omp parallel for reduction(+:sum) default(shared)
      for (size_type i= 0; i < size; ++i) {
        res+= unary_op(first[i]);
      }
      return res;
#endif
    }


  }


}

