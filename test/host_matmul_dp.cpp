#include <mathcca.hpp>
#include <gtest/gtest.h>

TEST(MatmulDp, BasicAssertions)
{
    std::size_t l{5};
    std::size_t m{3};
    std::size_t n{2};
    for (auto i= 1; i < 8; ++i) {
      mathcca::host_matrix<double> X0{l, m};
      mathcca::host_matrix<double> Y0{m, n};
      mathcca::host_matrix<double> B0{l, n};
      mathcca::host_matrix<double> T0{l, n};
      mathcca::host_matrix<double> C0{l, n};
      mathcca::host_matrix<double> ERR{99, 99};
      
      mathcca::fill_rand(X0.begin(), X0.end());
      mathcca::fill_rand(Y0.begin(), Y0.end());
      EXPECT_TRUE(X0 != Y0);
      
      using value_type= typename decltype(X0)::value_type;
      
      EXPECT_THROW({mathcca::matmul(X0, ERR, mathcca::MM::Base());},  std::length_error);
      EXPECT_THROW({mathcca::matmul(X0, ERR, mathcca::MM::Tiled());}, std::length_error);
      
      mathcca::matmul<value_type, mathcca::MM::Base, 8>(X0, Y0, B0, mathcca::MM::Base());
      auto B1= mathcca::matmul<value_type, mathcca::MM::Base, 16>(X0, Y0, mathcca::MM::Base());
      auto B2= mathcca::matmul<value_type, mathcca::MM::Base, 32 >(X0, Y0, mathcca::MM::Base());
      
      EXPECT_TRUE(B0 == B1);
      EXPECT_TRUE(B1 == B2);
      
      mathcca::matmul<value_type, mathcca::MM::Tiled, 8>(X0, Y0, T0, mathcca::MM::Tiled());
      auto T1 = mathcca::matmul<value_type, mathcca::MM::Tiled, 16>(X0, Y0, mathcca::MM::Tiled());
      auto T2 = mathcca::matmul<value_type, mathcca::MM::Tiled, 32>(X0, Y0, mathcca::MM::Tiled());
      
      EXPECT_TRUE(T0 == T1);
      EXPECT_TRUE(T1 == T2);
      
      EXPECT_TRUE(T0 == B0);
      EXPECT_TRUE(T1 == B1);
      EXPECT_TRUE(T2 == B2);

#ifdef _MKL

      mathcca::matmul<value_type, mathcca::MM::Mkl, 8>(X0, Y0, C0, mathcca::MM::Mkl());
      auto C1 = mathcca::matmul<value_type, mathcca::MM::Mkl, 16>(X0, Y0, mathcca::MM::Mkl());
      auto C2 = mathcca::matmul<value_type, mathcca::MM::Mkl, 32>(X0, Y0, mathcca::MM::Mkl());
      
      EXPECT_TRUE(C0 == C1);
      EXPECT_TRUE(C1 == C2);

      EXPECT_TRUE(C0 == B0);
      EXPECT_TRUE(C1 == B1);
      EXPECT_TRUE(C2 == B2);

      EXPECT_TRUE(C0 == T0);
      EXPECT_TRUE(C1 == T1);
      EXPECT_TRUE(C2 == T2);      

#endif

      std::swap(l,n);
      l *= 5;
      m *= 3;
      n *= 2;
  }
}


