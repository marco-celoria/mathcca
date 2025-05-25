#include <mathcca/host_matrix.h>
#include <mathcca/device_matrix.h>
#include <mathcca/norm.h>
#include <mathcca/detail/norm_impl.h>
#include <iostream>
#include <mathcca/fill_rand.h>
#include <mathcca/fill_const.h>
#include <mathcca/copy.h>

int main(int argc, char **argv)  {
  constexpr std::size_t l{36504};
  constexpr std::size_t m{28333};
  constexpr std::size_t n{l * m};
#ifdef _USE_DOUBLE_PRECISION
  using value_type= double;
#else
  using value_type= float;
#endif
  mathcca::device_matrix<value_type> dA{l, m};

  mathcca::fill_rand(dA.begin(), dA.end());

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  auto resB= mathcca::frobenius_norm(dA, mathcca::Norm::Base());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float tB_milliseconds;
  cudaEventElapsedTime(&tB_milliseconds, start, stop);
  std::cout << "tB == " << tB_milliseconds << " ms\n";

  //https://en.wikipedia.org/wiki/Continuous_uniform_distribution
  
  auto res= std::sqrt(static_cast<value_type>(n/3.));
  std::cout << resB << " " << res << "\n";

#ifdef _CUBLAS
  cudaEventRecord(start);
  auto resC= mathcca::frobenius_norm(dA, mathcca::Norm::Cublas());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float tC_milliseconds;
  cudaEventElapsedTime(&tC_milliseconds, start, stop);

  std::cout << resB << " " << resC << "\n";
  std::cout << "tC == " << tC_milliseconds << " ms\n";

#endif

#ifdef _THRUST
  cudaEventRecord(start);
  auto resT= mathcca::detail::frobenius_norm(mathcca::Thrust(), dA.cbegin().get(), dA.cend().get(), mathcca::Norm::Base());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float tT_milliseconds;
  cudaEventElapsedTime(&tT_milliseconds, start, stop);
  std::cout << resB << " " << resT << "\n";
  std::cout << "tT == " << tT_milliseconds << " ms\n";
#endif

    // destroy events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

#ifdef _HOST_CHECK
  mathcca::host_matrix<value_type> hA{l,m};
  mathcca::copy(dA.cbegin(),  dA.cend(),  hA.begin());
  cudaDeviceSynchronize();
  auto resH= mathcca::frobenius_norm(hA,  mathcca::Norm::Base());
  std::cout << std::boolalpha << resH << " " << resB << std::noboolalpha << "\n";
#endif

}


