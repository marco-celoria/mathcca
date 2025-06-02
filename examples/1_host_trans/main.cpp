#include <mathcca/host_matrix.h>
#include <mathcca/transpose.h>
#include <iostream>
#include <mathcca/fill_rand.h>
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

  constexpr std::size_t r{35013};
  constexpr std::size_t c{30023};

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

  mathcca::fill_rand(A.begin(), A.end());
 
  auto t0= wtime();  
  auto B0= mathcca::transpose(A,  mathcca::Trans::Base());
  auto C0= mathcca::transpose(B0, mathcca::Trans::Base());
  t0= wtime() - t0;

  auto t1= wtime();
  auto B1= mathcca::transpose(A,  mathcca::Trans::Tiled());
  auto C1= mathcca::transpose(B1, mathcca::Trans::Tiled());
  t1= wtime() - t1;

  std::cout << "Does Base result agree with Tiled result? " << std::boolalpha << (B0 == B1) << std::noboolalpha << "\n";
  std::cout << "Is Base   result consistent? " << std::boolalpha << (C0 == A) << std::noboolalpha << "\n";
  std::cout << "Is Tiled  result consistent? " << std::boolalpha << (C1 == A) << std::noboolalpha << "\n";

  std::cout << "Base  time: " << t0 << "\n";
  std::cout << "Tiled time: " << t1 << "\n";

#ifdef _MKL

  auto t2= wtime();
  auto B2= mathcca::transpose(A,  mathcca::Trans::Mkl());
  auto C2= mathcca::transpose(B2, mathcca::Trans::Mkl());
  t2= wtime() - t2;

  std::cout << "Does Mkl result agree with Base  result? " << std::boolalpha << (B0 == B2) << std::noboolalpha << "\n";
  std::cout << "Does Mkl result agree with Tiled result? " << std::boolalpha << (B1 == B2) << std::noboolalpha << "\n";
  std::cout << "Is Mkl   result consistent? " << std::boolalpha << (C2 == A) << std::noboolalpha << "\n";
  std::cout << "Mkl time: " << t2 << "\n";

#endif

}


