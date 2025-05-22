#include <mathcca.hpp>
#include <gtest/gtest.h>

TEST(TransSp, BasicAssertions)
{
    std::size_t r{5};
    std::size_t c{2};
    for (auto i= 1; i < 9; ++i) {
      mathcca::device_matrix<float> dX{r, c};
      mathcca::host_matrix<float>   hX{r, c};
      mathcca::device_matrix<float> dY{r, c};
      mathcca::host_matrix<float>   hY{r, c};
      mathcca::device_matrix<float> dZ{r, c};
      mathcca::device_matrix<float> dX0{r, c};
      mathcca::host_matrix<float>   hX0{r, c};
      mathcca::device_matrix<float> dY0{r, c};
      mathcca::host_matrix<float>   hY0{r, c};
      mathcca::device_matrix<float> dZ0{r, c};
      mathcca::device_matrix<float> dB0{c, r};
      mathcca::host_matrix<float>   hB0{c, r};
      mathcca::device_matrix<float> dT0{c, r};
      mathcca::host_matrix<float>   hT0{c, r};
      mathcca::device_matrix<float> dC0{c, r};
      mathcca::device_matrix<float> dERR{99, 99};
      mathcca::host_matrix<float>   hERR{99, 99};

      mathcca::fill_rand(dX.begin(), dX.end());

      mathcca::copy(dX.begin(), dX.end(), dY.begin());
      mathcca::copy(dX.begin(), dX.end(), dZ.begin());
      
      mathcca::copy(dX.cbegin(), dX.cend(), hX.begin());
      cudaDeviceSynchronize();
      
      mathcca::copy(hX.cbegin(), hX.cend(), hY.begin());

      using value_type= typename decltype(dX)::value_type;

      EXPECT_TRUE(dX == dY);
      EXPECT_TRUE(dX == dZ);
      
      EXPECT_TRUE(hX == hY);

      EXPECT_THROW({mathcca::transpose(dX, dERR, mathcca::Trans::Base());},  std::length_error);
      EXPECT_THROW({mathcca::transpose(dY, dERR, mathcca::Trans::Tiled());}, std::length_error);
      
      EXPECT_THROW({mathcca::transpose(hX, hERR, mathcca::Trans::Base());},  std::length_error);
      EXPECT_THROW({mathcca::transpose(hY, hERR, mathcca::Trans::Tiled());}, std::length_error);

      mathcca::transpose<value_type, mathcca::Trans::Base, 8>(dX,  dB0, mathcca::Trans::Base());
      mathcca::transpose<value_type, mathcca::Trans::Base, 8>(dB0, dX0, mathcca::Trans::Base());
      
      mathcca::transpose<value_type, mathcca::Trans::Base>(hX,  hB0, mathcca::Trans::Base());
      mathcca::transpose<value_type, mathcca::Trans::Base>(hB0, hX0, mathcca::Trans::Base());

      auto dB1 = mathcca::transpose<value_type, mathcca::Trans::Base, 16>(dX,  mathcca::Trans::Base());
      auto dX1 = mathcca::transpose<value_type, mathcca::Trans::Base, 16>(dB1, mathcca::Trans::Base());
      auto dB2 = mathcca::transpose<value_type, mathcca::Trans::Base, 32>(dX,  mathcca::Trans::Base());
      auto dX2 = mathcca::transpose<value_type, mathcca::Trans::Base, 32>(dB2, mathcca::Trans::Base());

      mathcca::transpose<value_type, mathcca::Trans::Tiled, 8>(dY,  dT0, mathcca::Trans::Tiled());
      mathcca::transpose<value_type, mathcca::Trans::Tiled, 8>(dT0, dY0, mathcca::Trans::Tiled());
      
      mathcca::transpose<value_type, mathcca::Trans::Tiled>(hY,  hT0, mathcca::Trans::Tiled());
      mathcca::transpose<value_type, mathcca::Trans::Tiled>(hT0, hY0, mathcca::Trans::Tiled());
      
      auto dT1 = mathcca::transpose<value_type, mathcca::Trans::Tiled, 16>(dY,  mathcca::Trans::Tiled());
      auto dY1 = mathcca::transpose<value_type, mathcca::Trans::Tiled, 16>(dT1, mathcca::Trans::Tiled());
      auto dT2 = mathcca::transpose<value_type, mathcca::Trans::Tiled, 32>(dY,  mathcca::Trans::Tiled());
      auto dY2 = mathcca::transpose<value_type, mathcca::Trans::Tiled, 32>(dT2, mathcca::Trans::Tiled());

      EXPECT_TRUE(hX  == hX0);
      EXPECT_TRUE(hY  == hY0);
      EXPECT_TRUE(hB0 == hT0);

      EXPECT_TRUE(dX == dX0);
      EXPECT_TRUE(dX == dX1);
      EXPECT_TRUE(dX == dX2);

      EXPECT_TRUE(dY == dY0);
      EXPECT_TRUE(dY == dY1);
      EXPECT_TRUE(dY == dY2);

      EXPECT_TRUE(dB0 == dT0);
      EXPECT_TRUE(dB1 == dT1);
      EXPECT_TRUE(dB2 == dT2);

      EXPECT_TRUE(dB0 == dB1);
      EXPECT_TRUE(dB1 == dB2);
      EXPECT_TRUE(dT0 == dT1);
      EXPECT_TRUE(dT1 == dT2);

      mathcca::device_matrix<value_type> dRB{c,r};
      mathcca::device_matrix<value_type> dRT{c,r};
      
      mathcca::copy(hB0.cbegin(), hB0.cend(), dRB.begin());
      
      mathcca::copy(hT0.begin(),  hT0.end(),  dRT.begin());
      
      EXPECT_TRUE(dB0 == dRB);
      EXPECT_TRUE(dT0 == dRT);

#ifdef _CUBLAS
      mathcca::transpose<value_type, mathcca::Trans::Cublas, 8>(dZ,  dC0, mathcca::Trans::Cublas());
      mathcca::transpose<value_type, mathcca::Trans::Cublas, 8>(dC0, dZ0, mathcca::Trans::Cublas());
      auto dC1 = mathcca::transpose<value_type, mathcca::Trans::Cublas, 16>(dZ,  mathcca::Trans::Cublas());
      auto dZ1 = mathcca::transpose<value_type, mathcca::Trans::Cublas, 16>(dC1, mathcca::Trans::Cublas());
      auto dC2 = mathcca::transpose<value_type, mathcca::Trans::Cublas, 32>(dZ,  mathcca::Trans::Cublas());
      auto dZ2 = mathcca::transpose<value_type, mathcca::Trans::Cublas, 32>(dC2, mathcca::Trans::Cublas());

      EXPECT_TRUE(dZ == dZ1);
      EXPECT_TRUE(dZ == dZ2);

      EXPECT_TRUE(dB0 == dC0);
      EXPECT_TRUE(dB1 == dC1);
      EXPECT_TRUE(dB2 == dC2);

      EXPECT_TRUE(dC0 == dC1);
      EXPECT_TRUE(dC1 == dC2);
#endif
      std::swap(r,c);
      r *= 5;
      c *= 2;
    }

}


