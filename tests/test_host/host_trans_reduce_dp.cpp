/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include <mathcca.h>
#include <gtest/gtest.h>
#include <numbers>

TEST(TransReduceDp, BasicAssertions)
{

    std::size_t r{5};
    std::size_t c{2};
    for (auto i= 1; i < 9; ++i) {
     
      using value_type= double;
         
      mathcca::host_matrix<value_type> A{r, c};
      mathcca::fill_const(A.begin(), A.end(), static_cast<value_type>(3));
       
      auto init= static_cast<value_type>(0);
       
      using const_iter_type= decltype(A.cbegin());
      using iter_type= decltype(A.begin());
      using square_type= mathcca::detail::Square<value_type>;
        
      auto res_1= mathcca::transform_reduce_sum<const_iter_type, value_type, square_type>(A.cbegin(), A.cend(), mathcca::detail::Square<value_type>(), init);
      auto res_2= mathcca::transform_reduce_sum<      iter_type, value_type, square_type>(A.begin(),  A.end(),  mathcca::detail::Square<value_type>(), init);
       
      value_type res= static_cast<value_type>(3. * 3. * r * c);
      
      EXPECT_FLOAT_EQ(res, res_1);
      EXPECT_FLOAT_EQ(res, res_2);
          
      mathcca::fill_iota(A.begin(), A.end(), static_cast<value_type>(1));
      // https://en.wikipedia.org/wiki/Square_pyramidal_number
      if (i <= 6) {
        
	value_type n1{static_cast<value_type>(r * c)};
        value_type n2{static_cast<value_type>(r * r * c * c)};
        value_type n3{static_cast<value_type>(r * r * r * c * c * c)};
        
	res= n3/static_cast<value_type>(3) + n2/static_cast<value_type>(2) + n1/static_cast<value_type>(6);
         
	res_1= mathcca::transform_reduce_sum<const_iter_type, value_type, square_type>(A.cbegin(), A.cend(), mathcca::detail::Square<value_type>(), init);
        res_2= mathcca::transform_reduce_sum<      iter_type, value_type, square_type>(A.begin(),  A.end(),  mathcca::detail::Square<value_type>(), init);
         
	EXPECT_FLOAT_EQ(res, res_1);
        EXPECT_FLOAT_EQ(res, res_2);
	
      }
      else {
        // https://en.wikipedia.org/wiki/Basel_problem
        using invsquare_type= mathcca::detail::InverseSquare<value_type>;
        res_1= mathcca::transform_reduce_sum<const_iter_type, value_type, invsquare_type>(A.cbegin(), A.cend(), mathcca::detail::InverseSquare<value_type>(), init);
        res_2= mathcca::transform_reduce_sum<      iter_type, value_type, invsquare_type>(A.begin(),  A.end(),  mathcca::detail::InverseSquare<value_type>(), init);
           
        res= static_cast<value_type>(std::numbers::pi * std::numbers::pi / 6);
          
        EXPECT_FLOAT_EQ(res, res_1);
        EXPECT_FLOAT_EQ(res, res_2);
      }
      std::swap(r,c);
      r *= 5;
      c *= 2;
    }
}


