#include <mathcca/host_matrix.h>
#include <mathcca/device_matrix.h>
#include <mathcca/matmul.h>
#include <iostream>
#include <mathcca/fill_rand.h>
#include <mathcca/fill_const.h>
#include <mathcca/copy.h>

int main(int argc, char **argv)  {

  constexpr std::size_t l{22501};
  constexpr std::size_t m{23003};
  constexpr std::size_t n{24002};
#ifdef _USE_DOUBLE_PRECISION
 using value_type= double;
#else
 using value_type= float;
#endif
  constexpr value_type gops = l * n * (2. * m - 1.) * 1.e-9;
  
  mathcca::host_matrix<value_type>   hA{l, m};
  mathcca::host_matrix<value_type>   hB{m, n};
  mathcca::device_matrix<value_type> dA{l, m};
  mathcca::device_matrix<value_type> dB{m, n};

  // For example, define random matrix on host and const matrix on device
  mathcca::fill_rand( hA.begin(), hA.end());
  mathcca::fill_const(dB.begin(), dB.end(), static_cast<value_type>(0.1));
  
  // Copy HtoD and DtoH
  mathcca::copy(hA.cbegin(), hA.cend(), dA.begin());
  mathcca::copy(dB.cbegin(), dB.cend(), hB.begin());
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  cudaEventRecord(start);
  // Device Matrix multiplication
  auto dC0= mathcca::matmul(dA, dB, mathcca::MM::Base());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float t0_ms;
  cudaEventElapsedTime(&t0_ms, start, stop);

  cudaEventRecord(start);
  auto dC1= mathcca::matmul(dA, dB, mathcca::MM::Tiled());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float t1_ms;
  cudaEventElapsedTime(&t1_ms, start, stop);

  // Check device consistency
  std::cout << "Does Base result agree with Tiled result? "   << std::boolalpha << (dC0 == dC1) << std::noboolalpha << "\n";
  std::cout << "Base  time: " << t0_ms << " ms ; GFLOPSs == " << gops/t0_ms * 1.e+3 << "\n";
  std::cout << "Tiled time: " << t1_ms << " ms ; GFLOPSs == " << gops/t1_ms * 1.e+3 << "\n";

#ifdef _CUBLAS
  // Eventually, use cublas
  cudaEventRecord(start);
  auto dC2 = mathcca::matmul(dA, dB, mathcca::MM::Cublas());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float t2_ms = 0;
  cudaEventElapsedTime(&t2_ms, start, stop);

  // Check device consistency
  std::cout << "Does Cublas result agree with Base  result? "  << std::boolalpha << (dC0 == dC2) << std::noboolalpha << "\n";
  std::cout << "Does Cublas result agree with Tiled result? "  << std::boolalpha << (dC1 == dC2) << std::noboolalpha << "\n";
  std::cout << "Cublas time: " << t2_ms << " ms ; GFLOPSs == " << gops/t2_ms * 1.e+3 << "\n";
#endif

  // destroy events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

}


