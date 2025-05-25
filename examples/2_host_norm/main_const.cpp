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
#ifdef _USE_DOUBLE_PRECISION
  using value_type= double;
#else
  using value_type= float;
#endif
  
  mathcca::host_matrix<value_type> A{r, c};

  mathcca::fill_const(A.begin(), A.end(), static_cast<value_type>(1));

  auto tb= wtime();
  auto fnb= mathcca::frobenius_norm(A, mathcca::Norm::Base());
  tb= wtime() - tb;
  std::cout << "tb == " << tb << "\n";

#ifdef _MKL
  auto tm= wtime();
  auto fnm= mathcca::frobenius_norm(A, mathcca::Norm::Mkl());
  tm= wtime() - tm;
  std::cout << std::boolalpha << fnb << " " <<  fnm << std::noboolalpha << "\n";
  std::cout << "tm == " << tm << "\n";
#endif

#ifdef _STDPAR
  auto ts= wtime();
  auto fns= mathcca::detail::frobenius_norm(mathcca::StdPar(), A.cbegin().get(), A.cend().get(), mathcca::Norm::Base());
  ts= wtime() - ts;
  std::cout << std::boolalpha << fnb << " " << fns << std::noboolalpha << "\n";
  std::cout << "ts == " << ts << "\n";
#endif

}


