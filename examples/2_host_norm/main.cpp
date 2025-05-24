#include <mathcca/host_matrix.h>
#include <mathcca/norm.h>
#include <mathcca/fill_const.h>
#include <mathcca/execution_policy.h>
#include <mathcca/detail/transform_reduce_sum_impl.h>
#include <iostream>
#include <cmath>
#include <mathcca/implementation_policy.h>
int main(int argc, char **argv)  {
  std::size_t r{15000};
  std::size_t c{2000};
  //std::size_t c{1100};
#ifdef _USE_DOUBLE_PRECISION
  mathcca::host_matrix<double> A{r, c};
#else
  mathcca::host_matrix<float> A{r, c};
#endif

  using value_type= typename decltype(A)::value_type;
  mathcca::fill_const(A.begin(), A.end(), static_cast<value_type>(1));
      
  auto fnb= mathcca::frobenius_norm(A, mathcca::Norm::Base());
#ifdef _MKL
  auto fnm= mathcca::frobenius_norm(A, mathcca::Norm::Mkl());
  std::cout << std::boolalpha << fnb << " " <<  fnm << std::noboolalpha << "\n";
#endif

#ifdef _STDPAR
  auto fns= std::sqrt(mathcca::detail::transform_reduce_sum(mathcca::StdPar(), A.cbegin().get(), A.cend().get(), mathcca::detail::Square<value_type>(), static_cast<value_type>(0)));
  std::cout << std::boolalpha << fnb << " " << fns << std::noboolalpha << "\n";
#endif

}


