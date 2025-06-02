/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include <mathcca/host_matrix.h>
#include <mathcca/device_matrix.h>
#include <mathcca/transpose.h>
#include <iostream>
#include <mathcca/fill_rand.h>
#include <mathcca/fill_const.h>
#include <mathcca/copy.h>

#ifdef _OPENMP
 #include <omp.h>
#endif

int main(int argc, char **argv)  {
  
  constexpr std::size_t l{33501};
  constexpr std::size_t m{35333};

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

  mathcca::device_matrix<value_type> dA{l, m};

  mathcca::fill_rand(dA.begin(), dA.end());

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  auto dB0= mathcca::transpose(dA,  mathcca::Trans::Base());
  auto dB1= mathcca::transpose(dB0, mathcca::Trans::Tiled());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float tB_ms;
  cudaEventElapsedTime(&tB_ms, start, stop);

  cudaEventRecord(start);
  auto dT0= mathcca::transpose(dA,  mathcca::Trans::Tiled());
  auto dT1= mathcca::transpose(dT0, mathcca::Trans::Tiled());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float tT_ms;
  cudaEventElapsedTime(&tT_ms, start, stop);

  std::cout << "Does Base result agree with Tiled result? "  << std::boolalpha << (dB0 == dT0) << std::noboolalpha << "\n";
  std::cout << "Is Base   result consistent? "  << std::boolalpha << (dA == dB1) << std::noboolalpha << "\n";
  std::cout << "Is Tiled  result consistent? "  << std::boolalpha << (dA == dT1) << std::noboolalpha << "\n";
  std::cout << "Base  time: " << tB_ms << " ms\n";
  std::cout << "Tiled time: " << tT_ms << " ms\n";

#ifdef _CUBLAS
  cudaEventRecord(start);
  auto dC0= mathcca::transpose(dA,  mathcca::Trans::Cublas());
  auto dC1= mathcca::transpose(dC0, mathcca::Trans::Cublas());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float tC_ms;
  cudaEventElapsedTime(&tC_ms, start, stop);
  
  std::cout << "Does Cublas result agree with Base  result? "  << std::boolalpha << (dC0 == dB0) << std::noboolalpha << "\n";
  std::cout << "Does Cublas result agree with Tiled result? "  << std::boolalpha << (dC0 == dT0) << std::noboolalpha << "\n";
  std::cout << "Is Cublas   result consistent? "  << std::boolalpha << (dA == dC1) << std::noboolalpha << "\n";
  std::cout << "Cublas time: " << tC_ms << " ms\n";

#endif

  // destroy events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

}


