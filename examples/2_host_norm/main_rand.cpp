#include <mathcca/host_matrix.h>
#include <mathcca/norm.h>
#include <mathcca/fill_const.h>
#include <mathcca/fill_rand.h>
#include <mathcca/detail/fill_rand_impl.h>
#include <mathcca/execution_policy.h>
#include <mathcca/detail/transform_reduce_sum_impl.h>
#include <mathcca/detail/reduce_sum_impl.h>
#include <mathcca/detail/norm_impl.h>
#include <iostream>
#include <cmath>
#include <mathcca/implementation_policy.h>
int main(int argc, char **argv)  {

  std::size_t r{40000};
  std::size_t c{2000};
  //std::size_t c{1100};
  std::size_t n{r*c};
#ifdef _USE_DOUBLE_PRECISION
  mathcca::host_matrix<double> A{r, c};
#else
  mathcca::host_matrix<float> A{r, c};
#endif

  using value_type= typename decltype(A)::value_type;

  mathcca::detail::fill_rand(mathcca::Omp(), A.begin().get(), A.end().get());
      
  auto res_fn= std::sqrt(static_cast<value_type>(n / 3.));
  auto res_mn= static_cast<value_type>(n / 2.);

  auto fnb= mathcca::frobenius_norm(A, mathcca::Norm::Base());

#ifdef _MKL
  auto fnm= mathcca::frobenius_norm(A, mathcca::Norm::Mkl());
  std::cout << fnb << " " <<  fnm << res_fn << "\n";
#endif

#ifdef _STDPAR
  auto fns= std::sqrt(mathcca::detail::transform_reduce_sum(mathcca::StdPar(), A.cbegin().get(), A.cend().get(), mathcca::detail::Square<value_type>(), static_cast<value_type>(0)));
  auto mns= mathcca::detail::reduce_sum(mathcca::StdPar(), A.cbegin().get(), A.cend().get(), static_cast<value_type>(0));
  std::cout << fnb << " " << fns << " " << res_fn << "\n";
  std::cout << mns << " " << res_mn << "\n";
#endif

}


