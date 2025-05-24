#include <mathcca/host_matrix.h>
#include <mathcca/matmul.h>
#include <iostream>
#include <mathcca/fill_rand.h>
#include <mathcca/fill_const.h>

int main(int argc, char **argv)  {
  std::size_t l{1501};
  std::size_t m{3003};
  std::size_t n{2002};
#ifdef _USE_DOUBLE_PRECISION
  mathcca::host_matrix<double> A{l, m};
  mathcca::host_matrix<double> B{m, n};
#else
  mathcca::host_matrix<float> A{l, m};
  mathcca::host_matrix<float> B{m, n};
#endif
  using value_type= typename decltype(A)::value_type;

  mathcca::fill_rand(A.begin(), A.end());
  mathcca::fill_const(B.begin(), B.end(), static_cast<value_type>(0.5));
      
  auto C0= mathcca::matmul(A, B, mathcca::MM::Base());
      
  auto C1= mathcca::matmul(A, B, mathcca::MM::Tiled());
      
  std::cout << std::boolalpha << (C0 == C1) << std::noboolalpha << "\n";

#ifdef _MKL
  auto C2 = mathcca::matmul(A, B, mathcca::MM::Mkl());
  std::cout << std::boolalpha << (C0 == C2) << std::noboolalpha << "\n";
  std::cout << std::boolalpha << (C1 == C2) << std::noboolalpha << "\n";
#endif

}


