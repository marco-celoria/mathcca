#include <mathcca/host_matrix.h>
#include <mathcca/transpose.h>
#include <iostream>
#include <mathcca/fill_rand.h>

int main(int argc, char **argv)  {
  std::size_t r{1501};
  std::size_t c{2002};
#ifdef _USE_DOUBLE_PRECISION
  mathcca::host_matrix<double> A{r, c};
#else
  mathcca::host_matrix<float> A{r, c};
#endif

  mathcca::fill_rand(A.begin(), A.end());
      
  auto B0= mathcca::transpose(A, mathcca::Trans::Base());
  auto B1= mathcca::transpose(A, mathcca::Trans::Tiled());
  std::cout << std::boolalpha << (B0 == B1) << std::noboolalpha << "\n";
  auto C0= mathcca::transpose(B0, mathcca::Trans::Base());
  auto C1= mathcca::transpose(B1, mathcca::Trans::Tiled());
  std::cout << std::boolalpha << (C0 == A) << std::noboolalpha << "\n";
  std::cout << std::boolalpha << (C1 == A) << std::noboolalpha << "\n";

#ifdef _MKL
  auto B2= mathcca::transpose(A, mathcca::Trans::Mkl());
  std::cout << std::boolalpha << (B0 == B2) << std::noboolalpha << "\n";
  std::cout << std::boolalpha << (B1 == B2) << std::noboolalpha << "\n";
  auto C2= mathcca::transpose(B2, mathcca::Trans::Mkl());
  std::cout << std::boolalpha << (C2 == A) << std::noboolalpha << "\n";
#endif

}


