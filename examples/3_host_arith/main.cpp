#include <mathcca/host_matrix.h>
#include <iostream>
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
  mathcca::host_matrix<value_type> B{r, c};
  mathcca::host_matrix<value_type> D{r, c};

  mathcca::fill_const(A.begin(), A.end(), static_cast<value_type>(1.));
  mathcca::fill_const(B.begin(), B.end(), static_cast<value_type>(2.));
  mathcca::fill_const(D.begin(), D.end(), static_cast<value_type>(10.));
 
  auto t0= wtime();  
  auto C= (2. * A + B) + (B + A) * 2.;
  t0= wtime() - t0;


  std::cout << "Does the resulting matrix corresponds to the expected output? " << std::boolalpha << (C == D) << std::noboolalpha << "\n";
  std::cout << "Time: " << t0 << "\n";

}


