#ifndef COPY_IMPL_H_
#define COPY_IMPL_H_
#pragma once
#include <cstddef>
//#include <mathcca/device_iterator.h>
//#include <mathcca/host_iterator.h>

#ifdef _PARALG
#include <execution>
#include <ranges>
#endif

#ifdef __CUDACC__
#include <mathcca/device_helper.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#endif

#include <concepts>
#include <mathcca/common_algorithm.h>

namespace mathcca {
namespace detail {
#ifdef _PARALG
    template<std::floating_point T>
    void copy(StdPar, const T* s_first, const T* s_last, T* d_first); 
#endif

    template<std::floating_point T>
    void copy(Omp, const T* s_first, const T* s_last, T* d_first); 
}
}
#ifdef __CUDACC__
namespace mathcca {
namespace detail {
    template<std::floating_point T>
    void copy(Thrust, const T* s_first, const T* s_last, T* d_first);

    template<std::floating_point T>
    void copy(Cuda, const T* s_first, const T* s_last, T* d_first, cudaStream_t stream=0);

    template<std::floating_point T>
    void copy(CudaHtoHcpy, const T* s_first, const T* s_last, T* d_first, cudaStream_t stream=0);

    template<std::floating_point T>
    void copy(CudaDtoHcpy, const T* s_first, const T* s_last, T* d_first, cudaStream_t stream=0); 

    template<std::floating_point T>
    void copy(CudaHtoDcpy, const T* s_first, const T* s_last, T* d_first, cudaStream_t stream=0);
}
}
#endif

#include <mathcca/detail/copy_impl.inl>
#endif
