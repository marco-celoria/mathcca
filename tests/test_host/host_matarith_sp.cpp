/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include <mathcca.h>
#include <gtest/gtest.h>

TEST(MatArithSp, BasicAssertions)
{
    std::size_t r{2};
    std::size_t c{5};
    for (auto n= 1; n < 9; ++n) {

      using value_type= float;

      mathcca::host_matrix<value_type> A0{r, c, static_cast<value_type>(2)};
      mathcca::host_matrix<value_type> B0{r, c, static_cast<value_type>(3)};
      mathcca::host_matrix<value_type> CHECK{r, c};
      mathcca::host_matrix<value_type> ERR{r, r};
      
      EXPECT_THROW({A0 + ERR;},  std::length_error);
      EXPECT_THROW({A0 - ERR;},  std::length_error);
      EXPECT_THROW({A0 * ERR;},  std::length_error);

      auto C0 = A0 + B0;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(5));
      EXPECT_TRUE(C0 == CHECK);

      C0+= B0;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(8));
      EXPECT_TRUE(C0 == CHECK);

      auto C1 = A0 + B0 + C0;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(13));
      EXPECT_TRUE(C1 == CHECK);
      
      C1 = A0 + B0 + C0 + C1;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(26));
      EXPECT_TRUE(C1 == CHECK);
      
      C1-= C0;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(18));
      EXPECT_TRUE(C1 == CHECK);
      
      
      auto C2 = C1 - B0;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(15));
      EXPECT_TRUE(C2 == CHECK);
      
      C2 = C1 - A0 - B0 - C0;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(5));
      EXPECT_TRUE(C2 == CHECK);

      auto C3 = C2 * A0;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(10));
      EXPECT_TRUE(C3 == CHECK);
      
      C3 *= B0;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(30));
      EXPECT_TRUE(C3 == CHECK);
      
      C3 = C3 * (A0 + B0 + C0) * C2;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(1950));
      EXPECT_TRUE(C3 == CHECK);

      auto C4 = A0 * A0 * A0 * A0 * A0 * A0 * A0 * A0;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(256));
      EXPECT_TRUE(C4 == CHECK);
      
      C4 = C4 * static_cast<value_type>(2) * static_cast<value_type>(4) * A0;
      mathcca::fill_const(CHECK.begin(), CHECK.end(), static_cast<value_type>(4096));
      EXPECT_TRUE(C4 == CHECK);
      
      std::swap(r,c);
      r *= 5;
      c *= 2;
    }
}




