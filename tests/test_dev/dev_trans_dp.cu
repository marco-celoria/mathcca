/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include <mathcca.h>
#include <gtest/gtest.h>

TEST(TransDp, BasicAssertions)
{
    std::size_t r{5};
    std::size_t c{2};
    for (auto i= 1; i < 9; ++i) {

      using value_type= double;

      mathcca::device_matrix<value_type> dX{r, c};
      mathcca::host_matrix<value_type>   hX{r, c};
      mathcca::device_matrix<value_type> dY{r, c};
      mathcca::host_matrix<value_type>   hY{r, c};
      mathcca::device_matrix<value_type> dZ{r, c};
      mathcca::device_matrix<value_type> dX0{r, c};
      mathcca::host_matrix<value_type>   hX0{r, c};
      mathcca::device_matrix<value_type> dY0{r, c};
      mathcca::host_matrix<value_type>   hY0{r, c};
      mathcca::device_matrix<value_type> dZ0{r, c};
      mathcca::device_matrix<value_type> dB0{c, r};
      mathcca::host_matrix<value_type>   hB0{c, r};
      mathcca::device_matrix<value_type> dT0{c, r};
      mathcca::host_matrix<value_type>   hT0{c, r};
      mathcca::device_matrix<value_type> dC0{c, r};
      mathcca::device_matrix<value_type> dERR{99, 99};
      mathcca::host_matrix<value_type>   hERR{99, 99};

      mathcca::fill_rand(dX.begin(), dX.end());

      mathcca::copy(dX.begin(), dX.end(), dY.begin());
      mathcca::copy(dX.begin(), dX.end(), dZ.begin());
      
      mathcca::copy(dX.cbegin(), dX.cend(), hX.begin());
      cudaDeviceSynchronize();
      
      mathcca::copy(hX.cbegin(), hX.cend(), hY.begin());

      EXPECT_TRUE(dX == dY);
      EXPECT_TRUE(dX == dZ);
      
      EXPECT_TRUE(hX == hY);

      EXPECT_THROW({mathcca::transpose(dX, dERR, mathcca::Trans::Base());},  std::length_error);
      EXPECT_THROW({mathcca::transpose(dY, dERR, mathcca::Trans::Tiled());}, std::length_error);
      
      EXPECT_THROW({mathcca::transpose(hX, hERR, mathcca::Trans::Base());},  std::length_error);
      EXPECT_THROW({mathcca::transpose(hY, hERR, mathcca::Trans::Tiled());}, std::length_error);

      mathcca::transpose<decltype(dX),  mathcca::Trans::Base, 8>(dX,  dB0, mathcca::Trans::Base());
      mathcca::transpose<decltype(dB0), mathcca::Trans::Base, 8>(dB0, dX0, mathcca::Trans::Base());
      
      mathcca::transpose<decltype(hX),  mathcca::Trans::Base>(hX,  hB0, mathcca::Trans::Base());
      mathcca::transpose<decltype(hB0), mathcca::Trans::Base>(hB0, hX0, mathcca::Trans::Base());

      auto dB1 = mathcca::transpose<decltype(dX),  mathcca::Trans::Base, 16>(dX,  mathcca::Trans::Base());
      auto dX1 = mathcca::transpose<decltype(dB1), mathcca::Trans::Base, 16>(dB1, mathcca::Trans::Base());
      auto dB2 = mathcca::transpose<decltype(dX),  mathcca::Trans::Base, 32>(dX,  mathcca::Trans::Base());
      auto dX2 = mathcca::transpose<decltype(dB2), mathcca::Trans::Base, 32>(dB2, mathcca::Trans::Base());

      mathcca::transpose<decltype(dY),  mathcca::Trans::Tiled, 8>(dY,  dT0, mathcca::Trans::Tiled());
      mathcca::transpose<decltype(dT0), mathcca::Trans::Tiled, 8>(dT0, dY0, mathcca::Trans::Tiled());
      
      mathcca::transpose<decltype(hY),  mathcca::Trans::Tiled>(hY,  hT0, mathcca::Trans::Tiled());
      mathcca::transpose<decltype(hT0), mathcca::Trans::Tiled>(hT0, hY0, mathcca::Trans::Tiled());
      
      auto dT1 = mathcca::transpose<decltype(dY),  mathcca::Trans::Tiled, 16>(dY,  mathcca::Trans::Tiled());
      auto dY1 = mathcca::transpose<decltype(dT1), mathcca::Trans::Tiled, 16>(dT1, mathcca::Trans::Tiled());
      auto dT2 = mathcca::transpose<decltype(dY),  mathcca::Trans::Tiled, 32>(dY,  mathcca::Trans::Tiled());
      auto dY2 = mathcca::transpose<decltype(dT2), mathcca::Trans::Tiled, 32>(dT2, mathcca::Trans::Tiled());

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

      mathcca::transpose<decltype(dZ),  mathcca::Trans::Cublas, 8>(dZ,  dC0, mathcca::Trans::Cublas());
      mathcca::transpose<decltype(dC0), mathcca::Trans::Cublas, 8>(dC0, dZ0, mathcca::Trans::Cublas());
      auto dC1 = mathcca::transpose<decltype(dZ),  mathcca::Trans::Cublas, 16>(dZ,  mathcca::Trans::Cublas());
      auto dZ1 = mathcca::transpose<decltype(dC1), mathcca::Trans::Cublas, 16>(dC1, mathcca::Trans::Cublas());
      auto dC2 = mathcca::transpose<decltype(dZ),  mathcca::Trans::Cublas, 32>(dZ,  mathcca::Trans::Cublas());
      auto dZ2 = mathcca::transpose<decltype(dC2), mathcca::Trans::Cublas, 32>(dC2, mathcca::Trans::Cublas());

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


