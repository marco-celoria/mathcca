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

double wtime() {
#ifdef _OPENMP
  return omp_get_wtime();
#else
  using clock = std::chrono::high_resolution_clock;
  auto time = clock::now();
  auto duration = std::chrono::duration<double>(time.time_since_epoch());
  return duration.count();
#endif
}


int main(int argc, char **argv)  {

  constexpr std::size_t r{15000};
  constexpr std::size_t c{50333};
  //std::size_t c{1100};
  constexpr std::size_t n{r*c};

#ifdef _USE_DOUBLE_PRECISION
  using value_type= double;
#else
  using value_type= float;
#endif

  mathcca::host_matrix<value_type> A{r, c};


  mathcca::detail::fill_rand(mathcca::Omp(), A.begin().get(), A.end().get());

  auto res_fn= std::sqrt(static_cast<value_type>(n / 3.));
  
  auto res_mn= static_cast<value_type>(n / 2.);

  auto tb= wtime();
  auto fnb= mathcca::frobenius_norm(A, mathcca::Norm::Base());
  tb= wtime() - tb;
  std::cout << "tb == " << tb << "\n";

#ifdef _MKL
  auto tm= wtime();
  auto fnm= mathcca::frobenius_norm(A, mathcca::Norm::Mkl());
  tm= wtime() - tm;
  std::cout << fnb << " " <<  fnm << res_fn << "\n";
  std::cout << "tm == " << tm << "\n";

#endif

#ifdef _STDPAR
  auto ts= wtime();
  auto fns= std::sqrt(mathcca::detail::transform_reduce_sum(mathcca::StdPar(), A.cbegin().get(), A.cend().get(), mathcca::detail::Square<value_type>(), static_cast<value_type>(0)));
  ts= wtime() - ts;
  std::cout << "ts == " << ts << "\n";
  auto mns= mathcca::detail::reduce_sum(mathcca::StdPar(), A.cbegin().get(), A.cend().get(), static_cast<value_type>(0));
  std::cout << fnb << " " << fns << " " << res_fn << "\n";
  std::cout << mns << " " << res_mn << "\n";
#endif

}


