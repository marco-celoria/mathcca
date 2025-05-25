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
  mathcca::fill_rand(hA.begin(), hA.end());
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
  float t0_milliseconds;
  cudaEventElapsedTime(&t0_milliseconds, start, stop);

  cudaEventRecord(start);
  auto dC1= mathcca::matmul(dA, dB, mathcca::MM::Tiled());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float t1_milliseconds;
  cudaEventElapsedTime(&t1_milliseconds, start, stop);

  // Check device consistency
  std::cout << std::boolalpha << (dC0 == dC1) << std::noboolalpha << "\n";
  std::cout << "t0 == " << t0_milliseconds << " ms ; GFLOPSs == " << gops/t0_milliseconds * 1.e+3 << "\n";
  std::cout << "t1 == " << t1_milliseconds << " ms ; GFLOPSs == " << gops/t1_milliseconds * 1.e+3 << "\n";

#ifdef _CUBLAS
  // Eventually, use cublas
  cudaEventRecord(start);
  auto dC2 = mathcca::matmul(dA, dB, mathcca::MM::Cublas());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float t2_milliseconds = 0;
  cudaEventElapsedTime(&t2_milliseconds, start, stop);

  // Check device consistency
  std::cout << std::boolalpha << (dC0 == dC2) << std::noboolalpha << "\n";
  std::cout << std::boolalpha << (dC1 == dC2) << std::noboolalpha << "\n";
  std::cout << "t2 == " << t2_milliseconds << " ms ; GFLOPSs == " << gops/t2_milliseconds * 1.e+3 << "\n";
#endif

  // destroy events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

#ifdef _HOST_CHECK  
  mathcca::host_matrix<value_type> hC0{l,n};
  mathcca::host_matrix<value_type> hC1{l,n};
  mathcca::copy(dC0.cbegin(), dC0.cend(), hC0.begin());
  mathcca::copy(dC1.cbegin(), dC1.cend(), hC1.begin());
  cudaDeviceSynchronize();

  auto hD0= mathcca::matmul(hA, hB, mathcca::MM::Tiled());
  auto hD1= mathcca::matmul(hA, hB, mathcca::MM::Tiled());
  std::cout << std::boolalpha << (hD0 == hC0) << std::noboolalpha << "\n";
  std::cout << std::boolalpha << (hD1 == hC1) << std::noboolalpha << "\n";
  std::cout << std::boolalpha << (hD0 == hD1) << std::noboolalpha << "\n";
#endif

}


