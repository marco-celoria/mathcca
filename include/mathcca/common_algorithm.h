#ifndef COMMON_ALGORITHM_H_
#define COMMON_ALGORITHM_H_

#include <type_traits>

namespace mathcca {
  
    template<typename T>
    concept Arithmetic = std::is_arithmetic_v<T>;
    
    class Omp{};
#ifdef _PARALG
    class StdPar{};
#endif
#ifdef __CUDACC__
    class Cuda{};
    class Thrust{};
    class CudaDtoHcpy{};
    class CudaHtoHcpy{};
    class CudaHtoDcpy{};
    class CudaDtoDcpy{};
#endif

}

#endif

