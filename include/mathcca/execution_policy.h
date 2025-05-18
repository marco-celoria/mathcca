#ifndef EXECUTION_POLICY_H_
#define EXECUTION_POLICY_H_

namespace mathcca {
    
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
     
  class CudaHtoDcpy{};
     
#endif
     
}

#endif

