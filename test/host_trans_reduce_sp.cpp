#include <mathcca.hpp>
#include <gtest/gtest.h>


TEST(TransReduceSp, BasicAssertions)
{

    std::size_t r{5};
    std::size_t c{2};
    for (auto i= 1; i < 7; ++i) {
      std::cout << "A i= " << i << "\n";
      mathcca::host_matrix<float> A{r, c};
      using value_type= typename decltype(A)::value_type;
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
      if (i < 3) {
        std::cout << "B i= " << i << " " << r * c <<"\n";
	value_type n1{static_cast<value_type>(r * c)};
        value_type n2{static_cast<value_type>(r * r * c * c)};
        value_type n3{static_cast<value_type>(r * r * r * c * c * c)};
        
	res= n3/static_cast<value_type>(3) + n2/static_cast<value_type>(2) + n1/static_cast<value_type>(6);
        
	int true_res=0;
        for (auto i=1; i <= r * c; ++i) {
	   true_res+=(i*i);
	}
	
        res_1= mathcca::transform_reduce_sum<const_iter_type, value_type, square_type>(A.cbegin(), A.cend(), mathcca::detail::Square<value_type>(), init);
        res_2= mathcca::transform_reduce_sum<      iter_type, value_type, square_type>(A.begin(),  A.end(),  mathcca::detail::Square<value_type>(), init);
	std::cout << std::fixed;
	std::cout << res << " " << res_1 << " " << res_2 << " " << true_res << "\n"; 
	std::cout << std::scientific;
	EXPECT_FLOAT_EQ(res, res_1);
        EXPECT_FLOAT_EQ(res, res_2);
	
      }
      else {
        std::cout << "C i= " << i << "\n";
        // https://en.wikipedia.org/wiki/Basel_problem
        using invsquare_type= mathcca::detail::InverseSquare<value_type>;
        res_1= mathcca::transform_reduce_sum<const_iter_type, value_type, invsquare_type>(A.cbegin(), A.cend(), mathcca::detail::InverseSquare<value_type>(), init);
        res_2= mathcca::transform_reduce_sum<      iter_type, value_type, invsquare_type>(A.begin(),  A.end(),  mathcca::detail::InverseSquare<value_type>(), init);

        res= static_cast<value_type>(3.14159265359 * 3.14159265359 / 6);
      
        EXPECT_NEAR(res, res_1, decltype(A)::tol());
        EXPECT_NEAR(res, res_2, decltype(A)::tol());
      }
      std::swap(r,c);
      r *= 5;
      c *= 2;
    }
}


