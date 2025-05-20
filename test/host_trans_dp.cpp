#include <mathcca.hpp>
#include <gtest/gtest.h>

TEST(TransDp, BasicAssertions)
{
    std::size_t r{5};
    std::size_t c{2};
    for (auto i= 1; i < 9; ++i) {
      mathcca::host_matrix<double> X{r, c};
      mathcca::host_matrix<double> Y{r, c};
      mathcca::host_matrix<double> Z{r, c};
      mathcca::host_matrix<double> X0{r, c};
      mathcca::host_matrix<double> Y0{r, c};
      mathcca::host_matrix<double> Z0{r, c};
      mathcca::host_matrix<double> B0{c, r};
      mathcca::host_matrix<double> T0{c, r};
      mathcca::host_matrix<double> C0{c, r};
      mathcca::host_matrix<double> ERR{99, 99};
      
      mathcca::fill_rand(X.begin(), X.end());

      mathcca::copy(X.begin(), X.end(), Y.begin());
      mathcca::copy(X.begin(), X.end(), Z.begin());
      
      using value_type= typename decltype(X)::value_type;
      
      EXPECT_TRUE(X == Y);
      EXPECT_TRUE(X == Z);

      EXPECT_THROW({mathcca::transpose(X, ERR, mathcca::Trans::Base());},  std::length_error);
      EXPECT_THROW({mathcca::transpose(Y, ERR, mathcca::Trans::Tiled());}, std::length_error);

      mathcca::transpose<value_type, mathcca::Trans::Base, 8>(X,  B0, mathcca::Trans::Base());
      mathcca::transpose<value_type, mathcca::Trans::Base, 8>(B0, X0, mathcca::Trans::Base());
      auto B1 = mathcca::transpose<value_type, mathcca::Trans::Base, 16>(X,  mathcca::Trans::Base());
      auto X1 = mathcca::transpose<value_type, mathcca::Trans::Base, 16>(B1, mathcca::Trans::Base());
      auto B2 = mathcca::transpose<value_type, mathcca::Trans::Base, 32>(X,  mathcca::Trans::Base());
      auto X2 = mathcca::transpose<value_type, mathcca::Trans::Base, 32>(B2, mathcca::Trans::Base());
      
      mathcca::transpose<value_type, mathcca::Trans::Tiled, 8>(Y,  T0, mathcca::Trans::Tiled());
      mathcca::transpose<value_type, mathcca::Trans::Tiled, 8>(T0, Y0, mathcca::Trans::Tiled());
      auto T1 = mathcca::transpose<value_type, mathcca::Trans::Tiled, 16>(Y,  mathcca::Trans::Tiled());
      auto Y1 = mathcca::transpose<value_type, mathcca::Trans::Tiled, 16>(T1, mathcca::Trans::Tiled());
      auto T2 = mathcca::transpose<value_type, mathcca::Trans::Tiled, 32>(Y,  mathcca::Trans::Tiled());
      auto Y2 = mathcca::transpose<value_type, mathcca::Trans::Tiled, 32>(T2, mathcca::Trans::Tiled());

      EXPECT_TRUE(X == X1);
      EXPECT_TRUE(X == X2);

      EXPECT_TRUE(Y == Y1);
      EXPECT_TRUE(Y == Y2);

      EXPECT_TRUE(B0 == T0);
      EXPECT_TRUE(B1 == T1);
      EXPECT_TRUE(B2 == T2);

      EXPECT_TRUE(B0 == B1);
      EXPECT_TRUE(B1 == B2); 
      EXPECT_TRUE(T0 == T1);
      EXPECT_TRUE(T1 == T2);           

#ifdef _MKL
      mathcca::transpose<value_type, mathcca::Trans::Mkl, 8>(Z,  C0, mathcca::Trans::Mkl());
      mathcca::transpose<value_type, mathcca::Trans::Mkl, 8>(C0, Z0, mathcca::Trans::Mkl());
      auto C1 = mathcca::transpose<value_type, mathcca::Trans::Mkl, 16>(Z,  mathcca::Trans::Mkl());
      auto Z1 = mathcca::transpose<value_type, mathcca::Trans::Mkl, 16>(C1, mathcca::Trans::Mkl());
      auto C2 = mathcca::transpose<value_type, mathcca::Trans::Mkl, 32>(Z,  mathcca::Trans::Mkl());
      auto Z2 = mathcca::transpose<value_type, mathcca::Trans::Mkl, 32>(C2, mathcca::Trans::Mkl());

      EXPECT_TRUE(Z == Z1);
      EXPECT_TRUE(Z == Z2);

      EXPECT_TRUE(B0 == C0);
      EXPECT_TRUE(B1 == C1);
      EXPECT_TRUE(B2 == C2);

      EXPECT_TRUE(C0 == C1);
      EXPECT_TRUE(C1 == C2);
#endif
      std::swap(r,c);
      r *= 5;
      c *= 2;
    }

}


