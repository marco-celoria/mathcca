#include <mathcca.hpp>
#include <gtest/gtest.h>

TEST(NormDp, BasicAssertions)
{
    std::size_t r{5};
    std::size_t c{2};
    for (auto i= 1; i < 9; ++i) {
      
      mathcca::host_matrix<double> A{r, c};
      using value_type= typename decltype(A)::value_type;
      mathcca::fill_rand(A.begin(), A.end());
      
      auto res_base= mathcca::frobenius_norm<decltype(A), mathcca::Norm::Base>(A, mathcca::Norm::Base());

#ifdef _MKL
      auto res_mkl= mathcca::frobenius_norm<decltype(A), mathcca::Norm::Mkl> (A, mathcca::Norm::Mkl());
      EXPECT_FLOAT_EQ(res_base, res_mkl);
#endif

      mathcca::fill_const(A.begin(), A.end(), static_cast<value_type>(3));
      
      value_type res= std::sqrt(static_cast<value_type>(3. * 3. * r * c));
      
      res_base= mathcca::frobenius_norm<decltype(A), mathcca::Norm::Base>(A, mathcca::Norm::Base());
      EXPECT_FLOAT_EQ(res, res_base);
#ifdef _MKL
      res_mkl= mathcca::frobenius_norm<decltype(A), mathcca::Norm::Mkl> (A, mathcca::Norm::Mkl());
      EXPECT_FLOAT_EQ(res, res_mkl);
      EXPECT_FLOAT_EQ(res_base, res_mkl);
#endif

      // https://en.wikipedia.org/wiki/Square_pyramidal_number
      if (i < 7) {
	mathcca::fill_iota(A.begin(), A.end(), static_cast<value_type>(1));
        
	value_type n1{static_cast<value_type>(r * c)};
        value_type n2{static_cast<value_type>(r * r * c * c)};
        value_type n3{static_cast<value_type>(r * r * r * c * c * c)};
        
	res= std::sqrt(n3/static_cast<value_type>(3) + n2/static_cast<value_type>(2) + n1/static_cast<value_type>(6));
        
	res_base= mathcca::frobenius_norm<decltype(A), mathcca::Norm::Base>(A, mathcca::Norm::Base());
	EXPECT_FLOAT_EQ(res, res_base);

#ifdef _MKL
        res_mkl= mathcca::frobenius_norm<decltype(A), mathcca::Norm::Mkl> (A, mathcca::Norm::Mkl());
        EXPECT_FLOAT_EQ(res, res_base);
	EXPECT_FLOAT_EQ(res_base, res_mkl);
#endif
      }
      std::swap(r,c);
      r *= 5;
      c *= 2;
    }
}


