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
    
  class CudaD{};
  
  class CudaPH{};
       
  class CudaDtoHcpy{};
     
  class CudaHtoHcpy{};
    
  class CudaHtoDcpy{};
     
  class CudaDtoDcpy{};
     
#endif
     
}

#endif

