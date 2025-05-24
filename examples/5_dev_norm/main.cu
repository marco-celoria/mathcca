#include <mathcca/host_matrix.h>
#include <mathcca/device_matrix.h>
#include <mathcca/norm.h>
#include <iostream>
#include <mathcca/fill_rand.h>
#include <mathcca/fill_const.h>
#include <mathcca/copy.h>

int main(int argc, char **argv)  {
  std::size_t l{6504};
  std::size_t m{8333};
  std::size_t n{l * m};
#ifdef _USE_DOUBLE_PRECISION
  mathcca::device_matrix<double> dA{l, m};
#else
  mathcca::device_matrix<float> dA{l, m};
#endif
  using value_type= typename decltype(dA)::value_type;

  mathcca::fill_rand(dA.begin(), dA.end());
  auto resB= mathcca::frobenius_norm(dA, mathcca::Norm::Base());
  //https://en.wikipedia.org/wiki/Continuous_uniform_distribution
  auto res= std::sqrt(static_cast<value_type>(n/3.));
  std::cout << resB << " " << res << "\n";
#ifdef _CUBLAS
  auto resC= mathcca::frobenius_norm(dA, mathcca::Norm::Cublas());
  std::cout << resB << " " << resC << "\n";
#endif


#ifdef _HOST_CHECK
  mathcca::host_matrix<value_type> hA{l,m};
  mathcca::copy(dA.cbegin(),  dA.cend(),  hA.begin());
  cudaDeviceSynchronize();
  auto resH= mathcca::frobenius_norm(hA,  mathcca::Norm::Base());
  std::cout << std::boolalpha << resH << " " << resB << std::noboolalpha << "\n";
#endif

}


