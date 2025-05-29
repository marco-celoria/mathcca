#include <mathcca.hpp>
#include <gtest/gtest.h>

TEST(MatArithSp, BasicAssertions)
{
    std::size_t r{2};
    std::size_t c{5};
    for (auto n= 1; n < 9; ++n) {

      using value_type= float;

      mathcca::device_matrix<value_type> dA0{r, c, static_cast<value_type>(2)};
      mathcca::host_matrix<value_type>   hA0{r, c, static_cast<value_type>(2)};
      mathcca::device_matrix<value_type> dB0{r, c, static_cast<value_type>(3)};
      mathcca::host_matrix<value_type>   hB0{r, c, static_cast<value_type>(3)};
      mathcca::device_matrix<value_type> dCHECK{r, c};
      mathcca::host_matrix<value_type>   hCHECK{r, c};
      mathcca::device_matrix<value_type> dERR{r, r};
      mathcca::host_matrix<value_type>   hERR{r, r};
      
      EXPECT_THROW({dA0 + dERR;},  std::length_error);
      EXPECT_THROW({dA0 - dERR;},  std::length_error);
      EXPECT_THROW({dA0 * dERR;},  std::length_error);
      
      EXPECT_THROW({hA0 + hERR;},  std::length_error);
      EXPECT_THROW({hA0 - hERR;},  std::length_error);
      EXPECT_THROW({hA0 * hERR;},  std::length_error);

      auto dC0 = dA0 + dB0;
      mathcca::fill_const(dCHECK.begin(), dCHECK.end(), static_cast<value_type>(5));
      EXPECT_TRUE(dC0 == dCHECK);
      
      auto hC0 = hA0 + hB0;
      mathcca::fill_const(hCHECK.begin(), hCHECK.end(), static_cast<value_type>(5));
      EXPECT_TRUE(hC0 == hCHECK);

      dC0+= dB0;
      mathcca::fill_const(dCHECK.begin(), dCHECK.end(), static_cast<value_type>(8));
      EXPECT_TRUE(dC0 == dCHECK);
      
      hC0+= hB0;
      mathcca::fill_const(hCHECK.begin(), hCHECK.end(), static_cast<value_type>(8));
      EXPECT_TRUE(hC0 == hCHECK);

      auto dC1 = dA0 + dB0 + dC0;
      mathcca::fill_const(dCHECK.begin(), dCHECK.end(), static_cast<value_type>(13));
      EXPECT_TRUE(dC1 == dCHECK);
      
      auto hC1 = hA0 + hB0 + hC0;
      mathcca::fill_const(hCHECK.begin(), hCHECK.end(), static_cast<value_type>(13));
      EXPECT_TRUE(hC1 == hCHECK);
      
      dC1 = dA0 + dB0 + dC0 + dC1;
      mathcca::fill_const(dCHECK.begin(), dCHECK.end(), static_cast<value_type>(26));
      EXPECT_TRUE(dC1 == dCHECK);
      
      hC1 = hA0 + hB0 + hC0 + hC1;
      mathcca::fill_const(hCHECK.begin(), hCHECK.end(), static_cast<value_type>(26));
      EXPECT_TRUE(hC1 == hCHECK);
      
      dC1-= dC0;
      mathcca::fill_const(dCHECK.begin(), dCHECK.end(), static_cast<value_type>(18));
      EXPECT_TRUE(dC1 == dCHECK);
      
      hC1-= hC0;
      mathcca::fill_const(hCHECK.begin(), hCHECK.end(), static_cast<value_type>(18));
      EXPECT_TRUE(hC1 == hCHECK);
      
      auto dC2 = dC1 - dB0;
      mathcca::fill_const(dCHECK.begin(), dCHECK.end(), static_cast<value_type>(15));
      EXPECT_TRUE(dC2 == dCHECK);
      
      auto hC2 = hC1 - hB0;
      mathcca::fill_const(hCHECK.begin(), hCHECK.end(), static_cast<value_type>(15));
      EXPECT_TRUE(hC2 == hCHECK);
      
      dC2 = dC1 - dA0 - dB0 - dC0;
      mathcca::fill_const(dCHECK.begin(), dCHECK.end(), static_cast<value_type>(5));
      EXPECT_TRUE(dC2 == dCHECK);
      
      hC2 = hC1 - hA0 - hB0 - hC0;
      mathcca::fill_const(hCHECK.begin(), hCHECK.end(), static_cast<value_type>(5));
      EXPECT_TRUE(hC2 == hCHECK);

      auto dC3 = dC2 * dA0;
      mathcca::fill_const(dCHECK.begin(), dCHECK.end(), static_cast<value_type>(10));
      EXPECT_TRUE(dC3 == dCHECK);
      
      auto hC3 = hC2 * hA0;
      mathcca::fill_const(hCHECK.begin(), hCHECK.end(), static_cast<value_type>(10));
      EXPECT_TRUE(hC3 == hCHECK);
      
      dC3 *= dB0;
      mathcca::fill_const(dCHECK.begin(), dCHECK.end(), static_cast<value_type>(30));
      EXPECT_TRUE(dC3 == dCHECK);
      
      hC3 *= hB0;
      mathcca::fill_const(hCHECK.begin(), hCHECK.end(), static_cast<value_type>(30));
      EXPECT_TRUE(hC3 == hCHECK);
      
      dC3 = dC3 * (dA0 + dB0 + dC0) * dC2;
      mathcca::fill_const(dCHECK.begin(), dCHECK.end(), static_cast<value_type>(1950));
      EXPECT_TRUE(dC3 == dCHECK);
      
      hC3 = hC3 * (hA0 + hB0 + hC0) * hC2;
      mathcca::fill_const(hCHECK.begin(), hCHECK.end(), static_cast<value_type>(1950));
      EXPECT_TRUE(hC3 == hCHECK);

      auto dC4 = dA0 * dA0 * dA0 * dA0 * dA0 * dA0 * dA0 * dA0;
      mathcca::fill_const(dCHECK.begin(), dCHECK.end(), static_cast<value_type>(256));
      EXPECT_TRUE(dC4 == dCHECK);
      
      auto hC4 = hA0 * hA0 * hA0 * hA0 * hA0 * hA0 * hA0 * hA0;
      mathcca::fill_const(hCHECK.begin(), hCHECK.end(), static_cast<value_type>(256));
      EXPECT_TRUE(hC4 == hCHECK);
      
      dC4 = dC4 * static_cast<value_type>(2) * static_cast<value_type>(4) * dA0;
      mathcca::fill_const(dCHECK.begin(), dCHECK.end(), static_cast<value_type>(4096));
      EXPECT_TRUE(dC4 == dCHECK);
      
      hC4 = hC4 * static_cast<value_type>(2) * static_cast<value_type>(4) * hA0;
      mathcca::fill_const(hCHECK.begin(), hCHECK.end(), static_cast<value_type>(4096));
      EXPECT_TRUE(hC4 == hCHECK);
      
      std::swap(r,c);
      r *= 5;
      c *= 2;
    }
}




