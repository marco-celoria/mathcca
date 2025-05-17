#ifndef COMMON_ALGORITHM_H_
#define COMMON_ALGORITHM_H_

#include <type_traits>

namespace mathcca {
    
  template<typename T>
  concept Arithmetic = std::is_arithmetic_v<T>;
    
  class Omp{};
    
#ifdef _STDPAR
   
  class StdPar{};
    
#endif
     
#ifdef __CUDACC__
    
#ifdef _THRUST 
    
  class Thrust{};
    
#endif   
    
  class Cuda{};
       
  class CudaDtoHcpy{};
     
  class CudaHtoHcpy{};
    
  class CudaHtoDcpy{};
     
  class CudaDtoDcpy{};
     
#endif
     
}

#endif

