#include <mathcca.hpp>
#include <gtest/gtest.h>

TEST(CopySp, BasicAssertions)
{
    std::size_t r{2};
    std::size_t c{5};
    for (auto n= 1; n < 9; ++n) {
      mathcca::host_matrix<float> dA{r, c, static_cast<float>(n)};
      mathcca::host_matrix<float> dB{r, c, static_cast<float>(n+1)};
      mathcca::host_matrix<float> dC{r, c};
      mathcca::host_matrix<float> dD{r, c, static_cast<float>(n)};
      mathcca::host_matrix<float> dE{r, c, static_cast<float>(n+1)};
      using value_type= typename decltype(dA)::value_type;
      
      EXPECT_TRUE(dA != dB);
      EXPECT_TRUE(dA != dC);
      EXPECT_TRUE(dB != dC);
      EXPECT_TRUE(dA == dD);
      EXPECT_TRUE(dB == dE);
      
      mathcca::copy(dA.begin(),  dA.end(),  dB.begin());
      mathcca::copy(dA.cbegin(), dA.cend(), dC.begin());
      EXPECT_TRUE(dA == dB);
      EXPECT_TRUE(dA == dC);
      EXPECT_TRUE(dB == dC);
      EXPECT_TRUE(dA == dD);
      EXPECT_TRUE(dB != dE);
      
      mathcca::fill_const(dA.begin(), dA.end(), static_cast<value_type>(n + decltype(dA)::tol() / 10.));
      mathcca::fill_const(dB.begin(), dB.end(), static_cast<value_type>(n + decltype(dA)::tol() * 2.));
      mathcca::fill_const(dC.begin(), dC.end(), static_cast<value_type>(n + 1));
      EXPECT_TRUE(dA != dB);
      EXPECT_TRUE(dA != dC);
      EXPECT_TRUE(dA == dD);
      EXPECT_TRUE(dA != dE);
      EXPECT_TRUE(dC == dE);
      
      
      std::swap(r,c);
      r*= 5;
      c*= 2;
    }
}


