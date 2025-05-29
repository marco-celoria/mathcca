#ifndef DEVICE_HELPER_H_
#define DEVICE_HELPER_H_
#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>

#ifdef _CUBLAS
 #include <cublas_v2.h>
#endif

#ifdef _THRUST
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#endif

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}
  
#ifdef _CUBLAS

static const char *_cudaGetErrorEnum(cublasStatus_t error) {
    
  switch (error) {
            
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
        
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
        
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
        
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
        
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
        
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
         
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
        
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
        
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
         
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    
  return "<unknown>";
    
}

#endif
  
template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
  if (result) {
    fprintf(stderr, "\nCUDA error in %s (%d) - %s : code=%d (%s)\n", 
            file, line, func, static_cast<unsigned int>(result), _cudaGetErrorEnum(result));
    exit(EXIT_FAILURE);
  }
}

#ifdef _THRUST

template <>
void check<cudaError_t>(cudaError_t code, char const *const func, const char *file, int line) {
  if(code != cudaSuccess) {
    std::string file_and_line{"\nCUDA error in " + std::string(file) + " (" + std::to_string(line) + ") - " 
		    + std::string(func) + " : code=" + std::to_string(code) + "\n"};
    throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
  }
}

#endif

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line) {
  cudaError_t code = cudaGetLastError();
  if(code != cudaSuccess) {
#ifdef _THRUST
    std::string file_and_line {"\ngetLastCudaError() in " + std::string(file) + " (" + std::to_string(line) + ") - " 
	    + std::string(errorMessage) + " : code=" + std::to_string(code) + " (" + cudaGetErrorString(code) + ")\n"};
    throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
#else
    fprintf(stderr, "\ngetLastCudaError() in %s (%i) - %s : code=%d (%s)\n",
            file, line, errorMessage, static_cast<int>(code), cudaGetErrorString(code));
    exit(EXIT_FAILURE);
#endif
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

#endif


