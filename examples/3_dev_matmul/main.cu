#include <mathcca/host_matrix.h>
#include <mathcca/device_matrix.h>
#include <mathcca/matmul.h>
#include <iostream>
#include <mathcca/fill_rand.h>
#include <mathcca/fill_const.h>
#include <mathcca/copy.h>

int main(int argc, char **argv)  {
  std::size_t l{2501};
  std::size_t m{3003};
  std::size_t n{4002};
#ifdef _USE_DOUBLE_PRECISION
  mathcca::host_matrix<double>   hA{l, m};
  mathcca::host_matrix<double>   hB{m, n};
  mathcca::device_matrix<double> dA{l, m};
  mathcca::device_matrix<double> dB{m, n};
#else
  mathcca::host_matrix<float>   hA{l, m};
  mathcca::host_matrix<float>   hB{m, n};
  mathcca::device_matrix<float> dA{l, m};
  mathcca::device_matrix<float> dB{m, n};
#endif
  using value_type= typename decltype(hA)::value_type;
  // For example, define random matrix on host and const matrix on device
  mathcca::fill_rand(hA.begin(), hA.end());
  mathcca::fill_const(dB.begin(), dB.end(), static_cast<value_type>(0.1));
  // Copy HtoD and DtoH
  mathcca::copy(hA.cbegin(), hA.cend(), dA.begin());
  mathcca::copy(dB.cbegin(), dB.cend(), hB.begin());
  cudaDeviceSynchronize();

  // Device Matrix multiplication
  auto dC0= mathcca::matmul(dA, dB, mathcca::MM::Base());
  auto dC1= mathcca::matmul(dA, dB, mathcca::MM::Tiled());
  // Check device consistency
  std::cout << std::boolalpha << (dC0 == dC1) << std::noboolalpha << "\n";

#ifdef _CUBLAS
  // Eventually, use cublas
  auto dC2 = mathcca::matmul(dA, dB, mathcca::MM::Cublas());
  // Check device consistency
  std::cout << std::boolalpha << (dC0 == dC2) << std::noboolalpha << "\n";
  std::cout << std::boolalpha << (dC1 == dC2) << std::noboolalpha << "\n";
#endif

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


