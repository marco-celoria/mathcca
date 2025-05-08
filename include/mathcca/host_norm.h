#ifndef HOST_NORM_H_
#define HOST_NORM_H_
#pragma once

#include <concepts>
#include <mathcca/host_matrix.h>

namespace mathcca {
     
#ifdef _MKL
    template<std::floating_point T>
    constexpr decltype(auto) frobenius_norm_Mkl(const host_matrix<T>& x);
#endif
    
#ifdef _STDPAR
    template<std::floating_point T>
    constexpr decltype(auto) frobenius_norm_Pstl(const host_matrix<T>& x) ;
#endif
    
    template<std::floating_point T>
    constexpr decltype(auto) frobenius_norm_Base(const host_matrix<T>& x);
    
    enum class HostFN {
      Base
#ifdef _STDPAR
      , Pstl
#endif
#ifdef _MKL
      , Mkl
#endif
     };
    
    template<std::floating_point T, HostFN O>
    T frobenius_norm (const host_matrix<T>& x) {
      static_assert(O == HostFN::Base
#ifdef _STDPAR
                      || O == HostFN::Pstl
#endif
#ifdef _MKL
                      || O == HostFN::Mkl
#endif
                   );
      if constexpr(O == HostFN::Base) {
        return frobenius_norm_Base<T>(x);
      }
#ifdef _MKL
      else if constexpr(O == HostFN::Mkl) {
        return frobenius_norm_Mkl(x);
      }
#endif
#ifdef _STDPAR
      else if constexpr(O == HostFN::Pstl) {
        return frobenius_norm_Pstl(x);
      }
#endif
    }

}

#include <mathcca/detail/host_norm.inl>

#endif


