#include <mathcca/host_matrix.h>
#include <mathcca/device_matrix.h>
#include <mathcca/transpose.h>
#include <iostream>
#include <mathcca/fill_rand.h>
#include <mathcca/fill_const.h>
#include <mathcca/copy.h>

int main(int argc, char **argv)  {
  std::size_t l{3504};
  std::size_t m{4333};
#ifdef _USE_DOUBLE_PRECISION
  mathcca::device_matrix<double> dA{l, m};
#else
  mathcca::device_matrix<float> dA{l, m};
#endif
  using value_type= typename decltype(dA)::value_type;

  mathcca::fill_rand(dA.begin(), dA.end());
  auto dB0= mathcca::transpose(dA, mathcca::Trans::Base());
  auto dT0= mathcca::transpose(dA, mathcca::Trans::Tiled());
 
  auto dB1= mathcca::transpose(dB0, mathcca::Trans::Tiled());
  auto dT1= mathcca::transpose(dT0, mathcca::Trans::Tiled());
  std::cout << std::boolalpha << (dA  == dB1) << std::noboolalpha << "\n";
  std::cout << std::boolalpha << (dA  == dT1) << std::noboolalpha << "\n";
  std::cout << std::boolalpha << (dB0 == dT0) << std::noboolalpha << "\n";

#ifdef _CUBLAS
  auto dC0= mathcca::transpose(dA,  mathcca::Trans::Cublas());
  auto dC1= mathcca::transpose(dC0, mathcca::Trans::Cublas());
  std::cout << std::boolalpha << (dA  == dC1) << std::noboolalpha << "\n";
  std::cout << std::boolalpha << (dC0 == dB0) << std::noboolalpha << "\n";
  std::cout << std::boolalpha << (dC0 == dT0) << std::noboolalpha << "\n";
#endif

#ifdef _HOST_CHECK
  mathcca::host_matrix<value_type> hA{l,m};
  mathcca::host_matrix<value_type> hB0{m,l};
  mathcca::host_matrix<value_type> hT0{m,l};
  mathcca::copy(dA.cbegin(),  dA.cend(),  hA.begin());
  mathcca::copy(dB0.cbegin(), dB0.cend(), hB0.begin());
  mathcca::copy(dT0.cbegin(), dT0.cend(), hT0.begin());
  cudaDeviceSynchronize();

  auto hD0= mathcca::transpose(hA,  mathcca::Trans::Tiled());
  auto hD1= mathcca::transpose(hD0, mathcca::Trans::Tiled());

  std::cout << std::boolalpha << (hA  == hD1) << std::noboolalpha << "\n";
  std::cout << std::boolalpha << (hD0 == hB0) << std::noboolalpha << "\n";
  std::cout << std::boolalpha << (hD0 == hT0) << std::noboolalpha << "\n";
#endif

}


