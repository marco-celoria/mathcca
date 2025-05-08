#ifndef COMMON_ALGORITHM_H_
#define COMMON_ALGORITHM_H_

#include <type_traits>

namespace mathcca {
  
    template<typename T>
    concept Arithmetic = std::is_arithmetic_v<T>;
    
    class Omp{};
#ifdef _STDPAR
    class Stdpar{};
#endif
#ifdef __CUDACC__
    class Cuda{};
    class CudaDtoHcpy{};
    class CudaHtoDcpy{};
    class CudaDtoDcpy{};
#endif

}

#endif

