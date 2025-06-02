#include <mathcca/host_matrix.h>
#include <mathcca/matmul.h>
#include <iostream>
#include <mathcca/fill_rand.h>
#include <mathcca/fill_const.h>
#include <chrono>

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

  constexpr std::size_t l{4501};
  constexpr std::size_t m{4013};
  constexpr std::size_t n{5002};

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

  constexpr value_type gops = l * n * (2. * m - 1.) * 1.e-9;
  
  mathcca::host_matrix<value_type> A{l, m};
  mathcca::host_matrix<value_type> B{m, n};

  mathcca::fill_rand(A.begin(), A.end());
  mathcca::fill_const(B.begin(), B.end(), static_cast<value_type>(0.1));

  auto t0= wtime();  
  auto C0= mathcca::matmul(A, B, mathcca::MM::Base());
  t0 = wtime() - t0;

  auto t1= wtime(); 
  auto C1= mathcca::matmul(A, B, mathcca::MM::Tiled());
  t1 = wtime() - t1;

  std::cout << "Does Base result agree with Tiled result? " << std::boolalpha << (C0 == C1) << std::noboolalpha << "\n";

  std::cout << "Base  time: " << t0 << "; GFLOPSs = " << gops/t0 << "\n";
  std::cout << "Tiled time: " << t1 << "; GFLOPSs = " << gops/t1 << "\n";

#ifdef _MKL
  auto t2= wtime();
  auto C2 = mathcca::matmul(A, B, mathcca::MM::Mkl());
  t2 = wtime() - t2;
  std::cout << "Does Base  result agree with Mkl result? " <<  std::boolalpha << (C0 == C2) << std::noboolalpha << "\n";
  std::cout << "Does Tiled result agree with Mkl result? " <<  std::boolalpha << (C1 == C2) << std::noboolalpha << "\n";
  std::cout << "Mkl  time:" << t2 << "; GFLOPSs = " << gops/t2 << "\n";
#endif

}


