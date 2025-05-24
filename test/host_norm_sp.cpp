#include <mathcca.hpp>
#include <gtest/gtest.h>

TEST(NormSp, BasicAssertions)
{
    std::size_t r{5};
    std::size_t c{2};
    for (auto i= 1; i < 7; ++i) {
    //for (auto i= 1; i < 8; ++i) {
      
      mathcca::host_matrix<float> A{r, c};
      mathcca::fill_rand(A.begin(), A.end());
      using value_type= typename decltype(A)::value_type; 

      // https://en.wikipedia.org/wiki/Continuous_uniform_distribution
      auto res= std::sqrt(static_cast<value_type>(r * c / 3.) );

      auto res_base= mathcca::frobenius_norm<decltype(A), mathcca::Norm::Base>(A, mathcca::Norm::Base());
      EXPECT_NEAR(res_base, res, 0.99);

#ifdef _MKL
      auto res_mkl= mathcca::frobenius_norm<decltype(A), mathcca::Norm::Mkl> (A, mathcca::Norm::Mkl());
      EXPECT_NEAR(res_base, res_mkl, 0.99);
      EXPECT_NEAR(res_mkl, res, 0.99);
#endif

      mathcca::fill_const(A.begin(), A.end(), static_cast<value_type>(3));
      
      res= std::sqrt(static_cast<value_type>(3 * 3 * r * c));
      
      res_base= mathcca::frobenius_norm<decltype(A), mathcca::Norm::Base>(A, mathcca::Norm::Base());
      EXPECT_FLOAT_EQ(res, res_base);
#ifdef _MKL
      res_mkl= mathcca::frobenius_norm<decltype(A), mathcca::Norm::Mkl> (A, mathcca::Norm::Mkl());
      EXPECT_FLOAT_EQ(res, res_mkl);
      EXPECT_FLOAT_EQ(res_mkl, res_base);
#endif

      // https://en.wikipedia.org/wiki/Square_pyramidal_number
      if (i < 6) {
	mathcca::fill_iota(A.begin(), A.end(), static_cast<value_type>(1));
        
	value_type n1{static_cast<value_type>(r * c)};
        value_type n2{static_cast<value_type>(r * r * c * c)};
        value_type n3{static_cast<value_type>(r * r * r * c * c * c)};
        
	res= std::sqrt(n3/static_cast<value_type>(3) + n2/static_cast<value_type>(2) + n1/static_cast<value_type>(6));
        
	res_base= mathcca::frobenius_norm<decltype(A), mathcca::Norm::Base>(A, mathcca::Norm::Base());

	EXPECT_FLOAT_EQ(res, res_base);

#ifdef _MKL
        res_mkl= mathcca::frobenius_norm<decltype(A), mathcca::Norm::Mkl> (A, mathcca::Norm::Mkl());
        EXPECT_FLOAT_EQ(res, res_mkl);
        EXPECT_FLOAT_EQ(res_mkl, res_base);
#endif
      }
      std::swap(r,c);
      r *= 5;
      c *= 2;
    }
}


