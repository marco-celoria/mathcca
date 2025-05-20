#include <mathcca.hpp>
#include <gtest/gtest.h>


TEST(TransReduceSp, BasicAssertions)
{

    std::size_t r{5};
    std::size_t c{2};
    for (auto i= 1; i < 9; ++i) {
      std::cout << "A i= " << i << "\n";      
      mathcca::device_matrix<float> A{r, c};
      using value_type= typename decltype(A)::value_type;
      mathcca::fill_const(A.begin(), A.end(), static_cast<value_type>(3));
      auto init= static_cast<value_type>(0);
      using const_iter_type= decltype(A.cbegin());
      using iter_type= decltype(A.begin());
      using square_type= mathcca::detail::Square<value_type>;

      auto res_1= mathcca::transform_reduce_sum<const_iter_type, value_type, square_type, 32  >(A.cbegin(), A.cend(), mathcca::detail::Square<value_type>(), init);
      auto res_2= mathcca::transform_reduce_sum<      iter_type, value_type, square_type, 64  >(A.begin(),  A.end(),  mathcca::detail::Square<value_type>(), init);
      auto res_3= mathcca::transform_reduce_sum<const_iter_type, value_type, square_type, 128 >(A.cbegin(), A.cend(), mathcca::detail::Square<value_type>(), init);
      auto res_4= mathcca::transform_reduce_sum<      iter_type, value_type, square_type, 256 >(A.begin(),  A.end(),  mathcca::detail::Square<value_type>(), init);
      auto res_5= mathcca::transform_reduce_sum<const_iter_type, value_type, square_type, 512 >(A.cbegin(), A.cend(), mathcca::detail::Square<value_type>(), init);
      auto res_6= mathcca::transform_reduce_sum<      iter_type, value_type, square_type, 1024>(A.begin(),  A.end(),  mathcca::detail::Square<value_type>(), init);

      value_type res= static_cast<value_type>(3. * 3. * r * c);
      
      EXPECT_FLOAT_EQ(res, res_1);
      EXPECT_FLOAT_EQ(res, res_2);
      EXPECT_FLOAT_EQ(res, res_3);
      EXPECT_FLOAT_EQ(res, res_4);
      EXPECT_FLOAT_EQ(res, res_5);
      EXPECT_FLOAT_EQ(res, res_6);

      mathcca::fill_iota(A.begin(), A.end(), static_cast<value_type>(1));
      // https://en.wikipedia.org/wiki/Square_pyramidal_number
      if (i <= 6) {
        std::cout << "B i= " << i << "\n";
	value_type n1{static_cast<value_type>(r * c)};
        value_type n2{static_cast<value_type>(r * r * c * c)};
        value_type n3{static_cast<value_type>(r * r * r * c * c * c)};
        
	res= n3/static_cast<value_type>(3) + n2/static_cast<value_type>(2) + n1/static_cast<value_type>(6);
      
	res_1= mathcca::transform_reduce_sum<const_iter_type, value_type, square_type, 32  >(A.cbegin(), A.cend(), mathcca::detail::Square<value_type>(), init);
        res_2= mathcca::transform_reduce_sum<      iter_type, value_type, square_type, 64  >(A.begin(),  A.end(),  mathcca::detail::Square<value_type>(), init);
        res_3= mathcca::transform_reduce_sum<const_iter_type, value_type, square_type, 128 >(A.cbegin(), A.cend(), mathcca::detail::Square<value_type>(), init);
        res_4= mathcca::transform_reduce_sum<      iter_type, value_type, square_type, 256 >(A.begin(),  A.end(),  mathcca::detail::Square<value_type>(), init);
        res_5= mathcca::transform_reduce_sum<const_iter_type, value_type, square_type, 512 >(A.cbegin(), A.cend(), mathcca::detail::Square<value_type>(), init);
        res_6= mathcca::transform_reduce_sum<      iter_type, value_type, square_type, 1024>(A.begin(),  A.end(),  mathcca::detail::Square<value_type>(), init);
        
	EXPECT_FLOAT_EQ(res, res_1);
        EXPECT_FLOAT_EQ(res, res_2);
        EXPECT_FLOAT_EQ(res, res_3);
        EXPECT_FLOAT_EQ(res, res_4);
        EXPECT_FLOAT_EQ(res, res_5);
        EXPECT_FLOAT_EQ(res, res_6);
	
      }
      else {
        std::cout << "C i= " << i << "\n";      
        // https://en.wikipedia.org/wiki/Basel_problem
        using invsquare_type= mathcca::detail::InverseSquare<value_type>;
        res_1= mathcca::transform_reduce_sum<const_iter_type, value_type, invsquare_type, 32  >(A.cbegin(), A.cend(), mathcca::detail::InverseSquare<value_type>(), init);
        res_2= mathcca::transform_reduce_sum<      iter_type, value_type, invsquare_type, 64  >(A.begin(),  A.end(),  mathcca::detail::InverseSquare<value_type>(), init);
        res_3= mathcca::transform_reduce_sum<const_iter_type, value_type, invsquare_type, 128 >(A.cbegin(), A.cend(), mathcca::detail::InverseSquare<value_type>(), init);
        res_4= mathcca::transform_reduce_sum<      iter_type, value_type, invsquare_type, 256 >(A.begin(),  A.end(),  mathcca::detail::InverseSquare<value_type>(), init);
        res_5= mathcca::transform_reduce_sum<const_iter_type, value_type, invsquare_type, 512 >(A.cbegin(), A.cend(), mathcca::detail::InverseSquare<value_type>(), init);
        res_6= mathcca::transform_reduce_sum<      iter_type, value_type, invsquare_type, 1024>(A.begin(),  A.end(),  mathcca::detail::InverseSquare<value_type>(), init);

        res= static_cast<value_type>(3.14159265359 * 3.14159265359 / 6);
      
        EXPECT_FLOAT_EQ(res, res_1);
        EXPECT_FLOAT_EQ(res, res_2);
        EXPECT_FLOAT_EQ(res, res_3);
        EXPECT_FLOAT_EQ(res, res_4);
        EXPECT_FLOAT_EQ(res, res_5);
        EXPECT_FLOAT_EQ(res, res_6);
      }
      std::swap(r,c);
      r *= 5;
      c *= 2;
    }
}


