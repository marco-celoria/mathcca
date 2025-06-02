/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include <mathcca.h>
#include <gtest/gtest.h>

TEST(CopyCtorDp, BasicAssertions)
{
  std::size_t row{2};
  std::size_t col{5};
  for (auto n= 1; n < 9; ++n) {

    using value_type= double;

    mathcca::host_matrix<value_type> a1{row, col};
    mathcca::host_matrix<value_type> a2{row, col, static_cast<value_type>(n)};
    mathcca::host_matrix<value_type> d1{row, col, static_cast<value_type>(n+1)};
    mathcca::host_matrix<value_type> d2{1, 2};
    
    EXPECT_TRUE(a1 != a2);
    EXPECT_TRUE(a1 != d1);
    EXPECT_TRUE(d1 != d2);
    EXPECT_TRUE(a2 != d2);
    
    auto b1{a1};
    auto b2{a2};
    
    EXPECT_TRUE(a1 == b1);
    EXPECT_TRUE(a2 == b2);
    EXPECT_TRUE(b1 != b2);
    
    auto c1{b1};
    auto c2{b2};
    
    EXPECT_TRUE(c1 == a1);
    EXPECT_TRUE(c2 == a2);
    EXPECT_TRUE(c1 != c2);
    
    d1= c1;
    d2= c2;
    
    EXPECT_TRUE(d1 == a1);
    EXPECT_TRUE(d2 == a2);
    EXPECT_TRUE(d1 != d2);

    std::swap(row,col);
    row*= 5;
    col*= 2;
  }
}


