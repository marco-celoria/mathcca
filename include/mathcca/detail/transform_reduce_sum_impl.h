#ifndef TRANSFORM_REDUCE_SUM_IMPL_H_
#define TRANSFORM_REDUCE_SUM_IMPL_H_
#pragma once

#include <concepts>
#include <mathcca/common_algorithm.h>


namespace mathcca {
	namespace detail {
#ifdef _PARALG
    template<std::floating_point T, typename UnaryFunction>
    T transform_reduce_sum(StdPar, const T* first, const T* last, UnaryFunction unary_op, const T init);
#endif
    template<std::floating_point T, typename UnaryFunction>
    T transform_reduce_sum(Omp, const T* first, const T* last, UnaryFunction unary_op, const T init); 

  }
}


#ifdef __CUDACC__

namespace mathcca {
  namespace detail { 

    template<std::floating_point T, typename UnaryFunction>
    T transform_reduce_sum (Thrust, const T* first, const T* last, UnaryFunction unary_op, const T init);

    
    
    template<std::floating_point T, typename UnaryFunction, unsigned int THREAD_BLOCK_DIM>
    T transform_reduce_sum(Cuda, const T* first, const T* last, UnaryFunction unary_op, const T init, cudaStream_t stream); 
  }    
}


#endif
#include<mathcca/detail/transform_reduce_sum_impl.inl>
#endif


