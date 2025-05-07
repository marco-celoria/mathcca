#ifndef HOST_ALGORITHM_H
#define HOST_ALGORITHM_H

#include "common_algocca.h"


namespace algocca {


  namespace host {


    template<std::floating_point T>
    void fill_const(T* first, T* last, const T& value);


    template<std::floating_point T>
    void fill_iota(T* first, T* last, const T& value);
  

    template<std::floating_point T>
    void fill_rand(T* first, T* last);
  

    template<std::floating_point T>
    void copy(const T* first, const T* last, T* h_first);


    template<Arithmetic T>
    T reduce_sum(const T* first, const T* last, const T init);


    template<Arithmetic T, typename UnaryFunction>
    T transform_reduce_sum(const T* first, const T* last, UnaryFunction unary_op, const T init);

  
  }


}

#include "host_algorithm.inl"

#endif
