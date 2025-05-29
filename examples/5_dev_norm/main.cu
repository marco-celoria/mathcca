#include <mathcca/host_matrix.h>
#include <mathcca/device_matrix.h>
#include <mathcca/norm.h>
#include <mathcca/detail/norm_impl.h>
#include <iostream>
#include <mathcca/fill_rand.h>
#include <mathcca/fill_const.h>
#include <mathcca/copy.h>

int main(int argc, char **argv)  {
  constexpr std::size_t l{46504};
  constexpr std::size_t m{38333};
  constexpr std::size_t n{l * m};
#ifdef _USE_DOUBLE_PRECISION
  using value_type= double;
#else
  using value_type= float;
#endif
  mathcca::device_matrix<value_type> dA{l, m};

  mathcca::fill_rand(dA.begin(), dA.end());
  
  auto res= std::sqrt(static_cast<value_type>(n/3.));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  auto resB= mathcca::frobenius_norm(dA, mathcca::Norm::Base());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float tB_ms;
  cudaEventElapsedTime(&tB_ms, start, stop);

  std::cout << "\n" << "Does Base result agree with real result? " << resB << " -:- " << res << "\tError: " << std::abs(resB - res) << "\n";
  std::cout         << "Base  time: " << tB_ms << "\n";

#ifdef _CUBLAS
  cudaEventRecord(start);
  auto resC= mathcca::frobenius_norm(dA, mathcca::Norm::Cublas());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float tC_ms;
  cudaEventElapsedTime(&tC_ms, start, stop);

  std::cout << "\n" << "Does Cublas result agree with real result? " << resC << " -:- " << res  << " \tError: " << std::abs(resC - res ) << "\n";
  std::cout         << "Does Cublas result agree with Base result? " << resC << " -:- " << resB << " \tError: " << std::abs(resC - resB) << "\n";
  std::cout << "Cublas  time: " << tC_ms << "\n";

#endif

#ifdef _THRUST
  cudaEventRecord(start);
  auto resT= mathcca::detail::frobenius_norm(mathcca::Thrust(), dA.cbegin().get(), dA.cend().get(), mathcca::Norm::Base());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float tT_ms;
  cudaEventElapsedTime(&tT_ms, start, stop);

  std::cout << "\n" << "Does Thrust result agree with real result? " << resT << " -:- " << res  << " \tError: " << std::abs(resT - res ) << "\n";
  std::cout         << "Does Thrust result agree with Base result? " << resT << " -:- " << resB << " \tError: " << std::abs(resT - resB) << "\n";
  std::cout << "Thrust  time: " << tT_ms << "\n";

#endif

  // destroy events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

}


