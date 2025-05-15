#include <mathcca/device_matrix.h>
#include <mathcca/copy.h>
#include <mathcca/fill_const.h>
#include <mathcca/fill_rand.h>
#include <mathcca/fill_iota.h>
#include <mathcca/reduce_sum.h>

#include <mathcca/matmul.h>
#include <mathcca/norm.h>
#include <mathcca/transpose.h>

//#include "randev.h"


int main(int argc, char **argv)  {
  std::cout << "Test Matrix constructors" << std::endl;
  {
    std::size_t r{2};
    std::size_t c{5};
#ifdef _USE_DOUBLE_PRECISION
    for (auto n= 1; n < 9; ++n) {
      mathcca::device_matrix<double> a{r, c, static_cast<double>(n)};
      mathcca::device_matrix<double> cpy1{r, c};
      mathcca::device_matrix<double> cpy2{1, 2};
      mathcca::device_matrix<double> mve1{r, c};
      mathcca::device_matrix<double> mve2{2, 1};
#else
    for (auto n= 1; n < 8; ++n) {
      mathcca::device_matrix<float> a{r, c, static_cast<float>(n)};
      mathcca::device_matrix<float> cpy1{r, c};
      mathcca::device_matrix<float> cpy2{1, 2};
      mathcca::device_matrix<float> mve1{r, c};
      mathcca::device_matrix<float> mve2{2, 1};
#endif
      std::cout << "r = " << r << " c = " << c << "\n";
      std::cout << "--------------------------------------------------------\n";
      auto cpy0{a};
      cpy1= cpy0;
      cpy2= cpy0;
      std::cout << "--------------------------------------------------------\n";
      std::cout << std::boolalpha << (cpy0 == a) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (cpy1 == a) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (cpy1 == a) << std::noboolalpha << "\n";
      std::cout << "--------------------------------------------------------\n";
      auto mve0{std::move(cpy0)};
      mve1= std::move(cpy1);
      mve2= std::move(cpy2);
      std::cout << "--------------------------------------------------------\n";
      std::cout << std::boolalpha << (mve0 != cpy0) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (mve1 != cpy1) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (mve2 != cpy2) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (mve0 == a) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (mve1 == a) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (mve2 == a) << std::noboolalpha << "\n";
      std::cout << "--------------------------------------------------------\n";
      std::swap(r,c);
      r *= 5;
      c *= 2;
    }
  }
  std::cout << "Test Matrix algorithms\n";
  {
    std::size_t r{2};
    std::size_t c{5};
#ifdef _USE_DOUBLE_PRECISION
    for (auto n= 1; n < 9; ++n) {
      mathcca::device_matrix<double> A{r, c, static_cast<double>(n)};
      mathcca::device_matrix<double> B{r, c, static_cast<double>(n+1)};
      mathcca::device_matrix<double> C{r, c};
      mathcca::device_matrix<double> D{r, c, static_cast<double>(n)};
      mathcca::device_matrix<double> E{r, c, static_cast<double>(n+1)};
#else
    for (auto n= 1; n < 9; ++n) {
      mathcca::device_matrix<float> A{r, c, static_cast<float>(n)};
      mathcca::device_matrix<float> B{r, c, static_cast<float>(n+1)};
      mathcca::device_matrix<float> C{r, c};
      mathcca::device_matrix<float> D{r, c, static_cast<float>(n)};
      mathcca::device_matrix<float> E{r, c, static_cast<float>(n+1)};
#endif
      using value_type= typename decltype(A)::value_type;
      std::cout << "r = " << r << " c = " << c << "\n";
      std::cout << "---------------------------------------------------------\n";
      std::cout << std::boolalpha << (A != B) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (A != C) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (B != C) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (A == D) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (B == E) << std::noboolalpha << "\n";
      std::cout << "---------------------------------------------------------\n";
      ///////////////////////////////////
      cudaStream_t s_1;
      cudaStreamCreate(&s_1);
      mathcca::copy(A.cbegin(), A.cend(), B.begin(), s_1);
      cudaStreamSynchronize(s_1);
      cudaStreamDestroy(s_1);
      mathcca::copy(A.cbegin(), A.cend(),  C.begin());
      std::cout << "---------------------------------------------------------\n";
      std::cout << std::boolalpha << (A == B) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (A == C) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (B == C) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (A == D) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (B != E) << std::noboolalpha << "\n";
      std::cout << "--------------------------------------------------------\n" << "tol = " << decltype(A)::tol() << "\n"; 
      mathcca::fill_const(A.begin(), A.end(), static_cast<value_type>(n + decltype(A)::tol() / 2.));
      mathcca::fill_const(B.begin(), B.end(), static_cast<value_type>(n + decltype(A)::tol() * 2.));
      mathcca::fill_const(C.begin(), C.end(), static_cast<value_type>(n + 1));
      std::cout << "--------------------------------------------------------\n";
      std::cout << std::boolalpha << (A != B) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (A != C) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (A == D) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (A != E) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (C == E) << std::noboolalpha << "\n";
      std::cout << "--------------------------------------------------------\n";
    }
  }
  std::cout << "Test reduce and rand\n";
  {
    std::size_t r{2};
    std::size_t c{5};
#ifdef _USE_DOUBLE_PRECISION
    for (auto n= 1; n < 9; ++n) {
      mathcca::device_matrix<double> X{r, c};
      mathcca::device_matrix<double> Y{r, c, static_cast<double>(n)};
#else
    for (auto n= 1; n < 6; ++n) {
      mathcca::device_matrix<float> X{r, c};
      mathcca::device_matrix<float> Y{r, c, static_cast<float>(n)};
#endif
      using value_type= typename decltype(X)::value_type;
      mathcca::fill_const(X.begin(), X.end(), static_cast<value_type>(n));
      mathcca::fill_iota(Y.begin(),  Y.end(), static_cast<value_type>(1));
      const value_type sumX= mathcca::reduce_sum(X.cbegin(), X.cend(), static_cast<value_type>(0));
      const value_type sumY= mathcca::reduce_sum(Y.cbegin(), Y.cend(),  static_cast<value_type>(0));
      const auto sX= static_cast<value_type>(X.size());
      const auto sY= static_cast<value_type>(Y.size());
      const auto resX= static_cast<value_type>(n) * sX; 
      const auto resY= sY * (static_cast<value_type>(0.5)) * (sY + static_cast<value_type>(1)); 
      std::cout << std::boolalpha << ((sumX - resX) < decltype(X)::tol()) << "\n";
      std::cout << std::boolalpha << ((sumY - resY) < decltype(X)::tol()) << "\n";
      std::cout << "--------------------------------------------------------\n";
      if (n==1) {
        print_matrix(X);
      }
      mathcca::fill_rand(X.begin(), X.end());
      auto Z{X};
      std::cout << std::boolalpha << (X == Z) << std::noboolalpha << "\n";
      if (n==1) {
        print_matrix(X);
      }
      mathcca::fill_rand(X.begin(), X.end());
      std::cout << std::boolalpha << (X != Z) << std::noboolalpha << "\n";
      if (n==1) {
        print_matrix(X);
      }
      mathcca::fill_rand(X.begin(), X.end());
      if (n==1) {
        print_matrix(X);
      }
      std::cout << std::boolalpha << (X != Z) << std::noboolalpha << "\n";
      if (n==1) {
        print_matrix(Y);
      }
      mathcca::fill_rand(Y.begin(), Y.end());
      if (n==1) {
        print_matrix(Y);
      }
      std::cout << std::boolalpha << (Y != Z) << std::noboolalpha << "\n";
      std::swap(r,c);
      r *= 5;
      c *= 2;
    }
  }
  std::cout << "Test Matrix arithmetics\n";
  {
    std::size_t r{2};
    std::size_t c{5};
#ifdef _USE_DOUBLE_PRECISION
    for (auto n= 1; n < 9; ++n) {
      mathcca::device_matrix<double> A0{r, c, static_cast<double>(2)};
      mathcca::device_matrix<double> B0{r, c, static_cast<double>(3)};
      mathcca::device_matrix<double> CHECK{r, c};
      mathcca::device_matrix<double> ERR{r, r};
#else
    for (auto n= 1; n < 9; ++n) {
      mathcca::device_matrix<float> A0{r, c, static_cast<float>(2)};
      mathcca::device_matrix<float> B0{r, c, static_cast<float>(3)};
      mathcca::device_matrix<float> CHECK{r, c};
      mathcca::device_matrix<float> ERR{r, r};
#endif
      using value_type= typename decltype(CHECK)::value_type;
      std::cout << "r = " << r << " c = " << c << "\n";
      try {
        auto C0 = A0 + ERR;
      }
      catch(std::exception &e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
      }
      
      std::cout << "--------------------------------------------------------\n";
      auto C0 = A0 + B0;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(5));
      std::cout << std::boolalpha << (C0 == CHECK) << std::noboolalpha << "\n";
      std::cout << "--------------------------------------------------------\n";
      C0+= B0;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(8));
      std::cout << std::boolalpha << (C0 == CHECK) << std::noboolalpha << "\n";
      std::cout << "--------------------------------------------------------\n";
      auto C1 = A0 + B0 + C0;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(13));
      std::cout << std::boolalpha << (C1 == CHECK) << std::noboolalpha << "\n";
      std::cout << "--------------------------------------------------------\n";
      C1 = A0 + B0 + C0 + C1;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(26));
      std::cout << std::boolalpha << (C1 == CHECK) << std::noboolalpha << "\n";
      std::cout << "--------------------------------------------------------\n";
      C1-= C0;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(18));
      std::cout << std::boolalpha << (C1 == CHECK) << std::noboolalpha << "\n";
      std::cout << "--------------------------------------------------------\n";
      auto C2 = C1 - B0;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(15));
      std::cout << std::boolalpha << (C2 == CHECK) << std::noboolalpha << "\n";
      std::cout << "--------------------------------------------------------\n";
      C2 = C1 - A0 - B0 - C0;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(5));
      std::cout << std::boolalpha << (C2 == CHECK) << std::noboolalpha << "\n";
      std::cout << "--------------------------------------------------------\n";
      auto C3 = C2 * A0;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(10));
      std::cout << std::boolalpha << (C3 == CHECK) << std::noboolalpha << "\n";
      std::cout << "--------------------------------------------------------\n";
      C3 *= B0;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(30));
      std::cout << std::boolalpha << (C3 == CHECK) << std::noboolalpha << "\n";
      std::cout << "--------------------------------------------------------\n";
      C3 = C3 * (A0 + B0 + C0) * C2;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(1950));
      std::cout << std::boolalpha << (C3 == CHECK) << std::noboolalpha << "\n";
      std::cout << "--------------------------------------------------------\n";
      auto C4 = A0 * A0 * A0 * A0 * A0 * A0 * A0 * A0;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(256));
      std::cout << std::boolalpha << (C4 == CHECK) << std::noboolalpha << "\n";
      
      std::cout << "--------------------------------------------------------\n";
      C4 = C4 * static_cast<value_type>(2) * static_cast<value_type>(4) * A0;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(4096));
      std::cout << std::boolalpha << (C4 == CHECK) << std::noboolalpha << "\n";
      std::cout << "--------------------------------------------------------\n";

      std::swap(r,c);
      r *= 5;
      c *= 2;
    }
  }
  std::cout << "Test Matrix multiplication\n";
  {
    std::size_t l{5};
    std::size_t m{3};
    std::size_t n{2};
#ifdef _USE_DOUBLE_PRECISION
    for (auto i= 1; i < 8; ++i) {
      mathcca::device_matrix<double> A0{l, m};
      mathcca::device_matrix<double> B0{m, n};
      mathcca::device_matrix<double> C0{l, n};
      mathcca::device_matrix<double> ERR{99, 99};
#else
    for (auto i= 1; i < 8; ++i) {
      mathcca::device_matrix<float> A0{l, m};
      mathcca::device_matrix<float> B0{m, n};
      mathcca::device_matrix<float> C0{l, n};
      mathcca::device_matrix<float> ERR{99, 99};
#endif
      std::cout << "l = " << l << " m = " << m << " n = " << n << "\n";
      using value_type= typename decltype(A0)::value_type;
      try {
        mathcca::matmul<value_type, mathcca::DevMM::Base, 32>(A0,ERR);
      }
      catch(std::exception &e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
      }

      mathcca::fill_rand(A0.begin(), A0.end());
      mathcca::fill_rand(B0.begin(), B0.end());
      std::cout << "--------------------------------------------------------\n";
      std::cout << std::boolalpha << (A0 != B0) << std::noboolalpha << "\n";
      std::cout << "--------------------------------------------------------\n";
      mathcca::matmul<value_type, mathcca::DevMM::Base, 32>(A0, B0, C0);
      auto C1 = mathcca::matmul<value_type, mathcca::DevMM::Tiled, 32>(A0, B0);
#ifdef _CUBLAS
      auto C2 = mathcca::matmul<value_type, mathcca::DevMM::Cublas>(A0, B0);
#endif
      std::cout << "--------------------------------------------------------\n";
      std::cout << std::boolalpha << (C0 == C1) << std::noboolalpha << "\n";
#ifdef _CUBLAS
      std::cout << std::boolalpha << (C0 == C2) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (C1 == C2) << std::noboolalpha << "\n";
#endif
      std::cout << "--------------------------------------------------------\n";
      std::swap(l,n);
      l *= 5;
      m *= 3;
      n *= 2;
    }
  }
  std::cout << "Test Matrix transposition\n";
  {
    std::size_t r{5};
    std::size_t c{2};
#ifdef _USE_DOUBLE_PRECISION
    for (auto i= 1; i < 9; ++i) {
      mathcca::device_matrix<double> A{r, c};
      mathcca::device_matrix<double> B0{c, r};
      mathcca::device_matrix<double> C0{r, c};
      mathcca::device_matrix<double> ERR{99, 99};
#else
    for (auto i= 1; i < 8; ++i) {
      mathcca::device_matrix<float> A{r, c};
      mathcca::device_matrix<float> B0{c,r};
      mathcca::device_matrix<float> C0{r, c};
      mathcca::device_matrix<float> ERR{99, 99};
#endif
      std::cout << "r = " << r << " c = " << c << "\n";
      using value_type= typename decltype(A)::value_type;
      try {
        mathcca::transpose<value_type, mathcca::DevT::Base, 32>(A,ERR);
      }
      catch(std::exception &e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
      }

      mathcca::fill_rand(A.begin(), A.end());
      std::cout << "--------------------------------------------------------\n";
      mathcca::transpose<value_type, mathcca::DevT::Base, 32>(A, B0);
      mathcca::transpose<value_type, mathcca::DevT::Base, 32>(B0, C0);

      auto B1 = mathcca::transpose<value_type, mathcca::DevT::Tiled, 32>(A);
      auto C1 = mathcca::transpose<value_type, mathcca::DevT::Tiled, 32>(B1);
#ifdef _CUBLAS
      auto B2 = mathcca::transpose<value_type, mathcca::DevT::Cublas, 32>(A);
      auto C2 = mathcca::transpose<value_type, mathcca::DevT::Cublas, 32>(B2);
#endif
      std::cout << "--------------------------------------------------------\n";
      std::cout << std::boolalpha << (A  == C0) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (A  == C1) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (B0 == B1) << std::noboolalpha << "\n";
#ifdef _CUBLAS
      std::cout << std::boolalpha << (A  == C2) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (B0 == B2) << std::noboolalpha << "\n";
      std::cout << std::boolalpha << (B1 == B2) << std::noboolalpha << "\n";
#endif
      std::cout << "--------------------------------------------------------\n";
      std::swap(r,c);
      r *= 5;
      c *= 2;
    }
  }
  std::cout << "Test Matrix Frobenius Norm\n";
  {
    std::size_t r{5};
    std::size_t c{2};
#ifdef _USE_DOUBLE_PRECISION
    for (auto i= 1; i < 9 ; ++i) {
      mathcca::device_matrix<double> A{r, c};
#else
    for (auto i= 1; i < 7; ++i) {
      mathcca::device_matrix<float> A{r, c};
#endif
      std::cout << "r = " << r << " c = " << c << "\n";
      using value_type= typename decltype(A)::value_type;
      mathcca::fill_rand(A.begin(), A.end());
      auto res_base= mathcca::frobenius_norm<value_type, mathcca::DevFN::Base, 256>(A); 
#ifdef _CUBLAS
      auto res_cublas= mathcca::frobenius_norm<value_type, mathcca::DevFN::Cublas>(A); 
#endif
      std::cout << "--------------------------------------------------------\n";
#ifdef _CUBLAS      
      std::cout << std::boolalpha << (fabs(res_base   - res_cublas) < decltype(A)::tol()) << " " << res_base   << " " << res_cublas << std::noboolalpha << "\n";
#endif
      std::cout << "--------------------------------------------------------\n";
      
      mathcca::fill_const(A.begin(), A.end(), static_cast<value_type>(3));

      res_base= mathcca::frobenius_norm<value_type, mathcca::DevFN::Base, 128>(A);
#ifdef _CUBLAS
      res_cublas= mathcca::frobenius_norm<value_type, mathcca::DevFN::Cublas>(A);
#endif        
      value_type res= std::sqrt(static_cast<value_type>(3. * 3. * r * c));
      std::cout << std::boolalpha << (fabs(res_base   - res) < decltype(A)::tol()) << " " << res_base   << " " << res << std::noboolalpha << "\n";
#ifdef _CUBLAS      
      std::cout << std::boolalpha << (fabs(res_cublas - res) < decltype(A)::tol()) << " " << res_cublas << " " << res << std::noboolalpha << "\n";
#endif
      std::cout << "--------------------------------------------------------\n";
      // https://en.wikipedia.org/wiki/Square_pyramidal_number
      if (i < 6) {
        value_type n{static_cast<value_type>(r * c)};
        value_type n2{static_cast<value_type>(r * r * c * c)};
        value_type n3{static_cast<value_type>(r * r * r * c * c * c)};
        value_type res= std::sqrt(n3/static_cast<value_type>(3) + n2/static_cast<value_type>(2) + n/static_cast<value_type>(6));
        mathcca::fill_iota(A.begin(), A.end(), static_cast<value_type>(1));
        res_base= mathcca::frobenius_norm<value_type, mathcca::DevFN::Base>(A);
#ifdef _CUBLAS
        res_cublas= mathcca::frobenius_norm<value_type, mathcca::DevFN::Cublas>(A);
#endif
        std::cout << std::boolalpha << (fabs(res_base   - res) < decltype(A)::tol()) << " " << res_base   << " " << res << std::noboolalpha << "\n";
#ifdef _CUBLAS
        std::cout << std::boolalpha << (fabs(res_cublas - res) < decltype(A)::tol()) << " " << res_cublas << " " << res << std::noboolalpha << "\n";
#endif
        std::cout << "--------------------------------------------------------\n";
      }
      std::swap(r,c);
      r *= 5;
      c *= 2;
    }
    std::cout << "--------------------------------------------------------\n";
  }
    
}
