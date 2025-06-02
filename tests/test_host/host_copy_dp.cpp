/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include <mathcca.h>
#include <gtest/gtest.h>

TEST(CopyDp, BasicAssertions)
{
    std::size_t r{2};
    std::size_t c{5};
    for (auto n= 1; n < 9; ++n) {

      using value_type= double;
        
      mathcca::host_matrix<value_type> hA{r, c, static_cast<value_type>(n)};
      mathcca::host_matrix<value_type> hB{r, c, static_cast<value_type>(n+1)};
      mathcca::host_matrix<value_type> hC{r, c};
      mathcca::host_matrix<value_type> hD{r, c, static_cast<value_type>(n)};
      mathcca::host_matrix<value_type> hE{r, c, static_cast<value_type>(n+1)};
      
      EXPECT_TRUE(hA != hB);
      EXPECT_TRUE(hA != hC);
      EXPECT_TRUE(hB != hC);
      EXPECT_TRUE(hA == hD);
      EXPECT_TRUE(hB == hE);
      
      mathcca::copy(hA.begin(),  hA.end(),  hB.begin());
      mathcca::copy(hA.cbegin(), hA.cend(), hC.begin());
      EXPECT_TRUE(hA == hB);
      EXPECT_TRUE(hA == hC);
      EXPECT_TRUE(hB == hC);
      EXPECT_TRUE(hA == hD);
      EXPECT_TRUE(hB != hE);
      
      mathcca::fill_const(hA.begin(), hA.end(), static_cast<value_type>(n + decltype(hA)::tol() / 10.));
      mathcca::fill_const(hB.begin(), hB.end(), static_cast<value_type>(n + decltype(hA)::tol() * 2.));
      mathcca::fill_const(hC.begin(), hC.end(), static_cast<value_type>(n + 1));
      EXPECT_TRUE(hA != hB);
      EXPECT_TRUE(hA != hC);
      EXPECT_TRUE(hA == hD);
      EXPECT_TRUE(hA != hE);
      EXPECT_TRUE(hC == hE);

      mathcca::host_matrix<value_type> hOMP{hA.num_rows(), hA.num_cols()};
      mathcca::detail::copy(mathcca::Omp(), hA.begin().get(), hA.end().get(), hOMP.begin().get());
      EXPECT_TRUE(hA == hOMP);

#ifdef _STDPAR
      mathcca::host_matrix<value_type> hSTDPAR{hA.num_rows(), hA.num_cols()};
      mathcca::detail::copy(mathcca::StdPar(), hA.cbegin().get(), hA.cend().get(), hSTDPAR.begin().get());
      EXPECT_TRUE(hA == hSTDPAR);
#endif
      
      std::swap(r,c);
      r*= 5;
      c*= 2;
    }
}


