/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

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

#ifdef _OPENMP
 #include <omp.h>
#endif

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
  std::cout << "USE DOUBLE PRECISION\n";  
  using value_type= double;
#else
  std::cout << "USE SINGLE PRECISION\n";  
  using value_type= float;
#endif

#ifdef _OPENMP
  int num_threads = 0;
  #pragma omp parallel reduction(+:num_threads)
  num_threads += 1;
  std::cout << "Running with " << num_threads << " OMP threads\n";
#endif

  mathcca::host_matrix<value_type> A{r, c};

  mathcca::detail::fill_rand(mathcca::Omp(), A.begin().get(), A.end().get());

  auto res= std::sqrt(static_cast<value_type>(n / 3.));
  
  auto tb= wtime();
  auto fnb= mathcca::frobenius_norm(A, mathcca::Norm::Base());
  tb= wtime() - tb;
  std::cout << "\nDoes Base result agree with real result? " << fnb << " -:- " << res << " \tError: " << std::abs(fnb - res) << "\n";
  std::cout << "Base  time: " << tb << "\n";

#ifdef _MKL
  auto tm= wtime();
  auto fnm= mathcca::frobenius_norm(A, mathcca::Norm::Mkl());
  tm= wtime() - tm;
  std::cout << "\n" << "Does Mkl  result agree with real result? " << fnm << " -:- " << res << " \tError: " << std::abs(fnm - res) << "\n";
  std::cout         << "Does Base result agree with Mkl  result? " << fnb << " -:- " << fnm << " \tError: " << std::abs(fnb - fnm) << "\n";  
  std::cout << "Mkl  time: " << tm << "\n";
#endif

#ifdef _STDPAR
  auto ts= wtime();
  auto fns= mathcca::detail::frobenius_norm(mathcca::StdPar(), A.cbegin().get(), A.cend().get(), mathcca::Norm::Base());
  ts= wtime() - ts;
  std::cout << "\n" << "Does StdPar result agree with real   result? "  << fns << " -:- " << res << " \tError: " << std::abs(fns - res) << "\n";
  std::cout         << "Does Base   result agree with StdPar result? "  << fnb << " -:- " << fns << " \tError: " << std::abs(fnb - fns) << "\n";
  std::cout << "StdPar  time: " << ts << "\n";
#endif

}


