/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include <mathcca/host_matrix.h>
#include <iostream>
#include <mathcca/fill_const.h>
#include <mathcca/copy.h>
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

  mathcca::device_matrix<value_type> A{r, c};
  mathcca::device_matrix<value_type> B{r, c};
  mathcca::device_matrix<value_type> D{r, c};
  mathcca::host_matrix<value_type> H{r, c};

  mathcca::fill_const(A.begin(), A.end(), static_cast<value_type>(1.));
  mathcca::fill_const(B.begin(), B.end(), static_cast<value_type>(2.));
  mathcca::fill_const(D.begin(), D.end(), static_cast<value_type>(10.));
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
 
  auto C= (static_cast<value_type>(2) * A + B) + (B + A) * static_cast<value_type>(2);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);


  float t0_ms;
  cudaEventElapsedTime(&t0_ms, start, stop);
  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  std::cout << "Does the resulting matrix corresponds to the expected output? " << std::boolalpha << (C == D) << std::noboolalpha << "\n";
  std::cout << "Time: " << t0_ms << "\n";

  mathcca::copy(C.cbegin(), C.cend(), H.begin());
  cudaDeviceSynchronize();
  std::cout << H[0] << "\n";

}


