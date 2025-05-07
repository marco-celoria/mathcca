#ifndef COMMON_ALGORITHM_H_
#define COMMON_ALGORITHM_H_

#include <type_traits>

namespace mathcca {
  
  namespace algocca {

    template<typename T>
    concept Arithmetic = std::is_arithmetic<T>::value;

  }

}

#endif

