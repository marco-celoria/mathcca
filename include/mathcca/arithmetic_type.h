#ifndef ARITHMETIC_TYPE_H_
#define ARITHMETIC_TYPE_H_

#include <type_traits>

namespace mathcca {
    
  template<typename T>
  concept Arithmetic = std::is_arithmetic_v<T>;
     
}

#endif

