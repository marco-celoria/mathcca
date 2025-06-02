/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include <mathcca.h>
#include <gtest/gtest.h>

TEST(ReduceSp, BasicAssertions)
{
    std::size_t r{2};
    std::size_t c{5};
    for (auto n= 1; n < 5; ++n) {
      
      using value_type= float;    
	    
      mathcca::host_matrix<value_type> X{r, c};
      mathcca::host_matrix<value_type> Y{r, c, static_cast<value_type>(n)};
      
      mathcca::fill_const(X.begin(), X.end(), static_cast<value_type>(n));
      mathcca::fill_iota(Y.begin(),  Y.end(), static_cast<value_type>(1));

      const value_type sumX= mathcca::reduce_sum(X.begin(),  X.end(),  static_cast<value_type>(0));
      const value_type sumY= mathcca::reduce_sum(Y.cbegin(), Y.cend(), static_cast<value_type>(0));

      const auto sX= static_cast<value_type>(X.size());
      const auto sY= static_cast<value_type>(Y.size());

      const auto resX= static_cast<value_type>(n) * sX;
      const auto resY= sY / (static_cast<value_type>(2)) * (sY + static_cast<value_type>(1));

      EXPECT_FLOAT_EQ(sumX, resX);
      EXPECT_FLOAT_EQ(sumY, resY);
      
      std::swap(r,c);
      r*= 5;
      c*= 2;
    }
}


