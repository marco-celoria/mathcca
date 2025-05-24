#include <mathcca.hpp>
#include <gtest/gtest.h>

TEST(NormSp, BasicAssertions)
{
    std::size_t r{5};
    std::size_t c{2};
    for (auto i= 1; i < 7; ++i) {
      std::cout << "A i= " << i << "\n"; 
      mathcca::device_matrix<float> dA{r, c};
      mathcca::host_matrix<float>   hA{r, c};
      using value_type= typename decltype(dA)::value_type;
      
      mathcca::fill_rand(dA.begin(), dA.end());
      mathcca::copy(dA.cbegin(), dA.cend(), hA.begin());
      cudaDeviceSynchronize();
      // https://en.wikipedia.org/wiki/Continuous_uniform_distribution      
      auto res= std::sqrt(static_cast<value_type>(r * c / 3.) );
      auto res_host= mathcca::frobenius_norm<mathcca::host_matrix<float>, mathcca::Norm::Base >(hA, mathcca::Norm::Base());

      auto res_base_1= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Base, 32  >(dA, mathcca::Norm::Base());
      auto res_base_2= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Base, 64  >(dA, mathcca::Norm::Base());
      auto res_base_3= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Base, 128 >(dA, mathcca::Norm::Base());
      auto res_base_4= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Base, 256 >(dA, mathcca::Norm::Base());
      auto res_base_5= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Base, 512 >(dA, mathcca::Norm::Base());
      auto res_base_6= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Base, 1024>(dA, mathcca::Norm::Base());
      EXPECT_NEAR(res_host, res_base_1, 0.99);
      EXPECT_FLOAT_EQ(res_base_1, res_base_2);
      EXPECT_FLOAT_EQ(res_base_2, res_base_3);
      EXPECT_FLOAT_EQ(res_base_3, res_base_4);
      EXPECT_FLOAT_EQ(res_base_4, res_base_5);
      EXPECT_FLOAT_EQ(res_base_5, res_base_6);

      EXPECT_NEAR(res_host,   res, 0.99);
      EXPECT_NEAR(res_base_1, res, 0.99);
      EXPECT_NEAR(res_base_2, res, 0.99);
      EXPECT_NEAR(res_base_3, res, 0.99);
      EXPECT_NEAR(res_base_4, res, 0.99);
      EXPECT_NEAR(res_base_5, res, 0.99);
      EXPECT_NEAR(res_base_6, res, 0.99);

#ifdef _CUBLAS
      auto res_cublas_1= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Cublas, 32  > (dA, mathcca::Norm::Cublas());
      auto res_cublas_2= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Cublas, 64  > (dA, mathcca::Norm::Cublas());
      auto res_cublas_3= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Cublas, 128 > (dA, mathcca::Norm::Cublas());
      auto res_cublas_4= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Cublas, 256 > (dA, mathcca::Norm::Cublas());
      auto res_cublas_5= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Cublas, 512 > (dA, mathcca::Norm::Cublas());
      auto res_cublas_6= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Cublas, 1024> (dA, mathcca::Norm::Cublas());
      EXPECT_FLOAT_EQ(res_base_1, res_cublas_1);
      EXPECT_FLOAT_EQ(res_base_2, res_cublas_2);
      EXPECT_FLOAT_EQ(res_base_3, res_cublas_3);
      EXPECT_FLOAT_EQ(res_base_4, res_cublas_4);
      EXPECT_FLOAT_EQ(res_base_5, res_cublas_5);
      EXPECT_FLOAT_EQ(res_base_6, res_cublas_6);
      EXPECT_NEAR(res_cublas_1, res, 0.99);
      EXPECT_NEAR(res_cublas_2, res, 0.99);
      EXPECT_NEAR(res_cublas_3, res, 0.99);
      EXPECT_NEAR(res_cublas_4, res, 0.99);
      EXPECT_NEAR(res_cublas_5, res, 0.99);
      EXPECT_NEAR(res_cublas_6, res, 0.99);

#endif
      std::cout << "B i= " << i << "\n";
      mathcca::fill_const(dA.begin(), dA.end(), static_cast<value_type>(3));
      mathcca::copy(dA.cbegin(), dA.cend(), hA.begin());
      cudaDeviceSynchronize();
      
      res= std::sqrt(static_cast<value_type>(3. * 3. * r * c));
      
      res_host= mathcca::frobenius_norm<mathcca::host_matrix<float>, mathcca::Norm::Base >(hA, mathcca::Norm::Base());

      res_base_1= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Base, 32  >(dA, mathcca::Norm::Base());
      res_base_2= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Base, 64  >(dA, mathcca::Norm::Base());
      res_base_3= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Base, 128 >(dA, mathcca::Norm::Base());
      res_base_4= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Base, 256 >(dA, mathcca::Norm::Base());
      res_base_5= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Base, 512 >(dA, mathcca::Norm::Base());
      res_base_6= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Base, 1024>(dA, mathcca::Norm::Base());
      EXPECT_FLOAT_EQ(res, res_host);
      EXPECT_FLOAT_EQ(res, res_base_1);
      EXPECT_FLOAT_EQ(res, res_base_2);
      EXPECT_FLOAT_EQ(res, res_base_3);
      EXPECT_FLOAT_EQ(res, res_base_4);
      EXPECT_FLOAT_EQ(res, res_base_5);
      EXPECT_FLOAT_EQ(res, res_base_6);
#ifdef _CUBLAS
      res_cublas_1= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Cublas, 32  > (dA, mathcca::Norm::Cublas());
      res_cublas_2= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Cublas, 64  > (dA, mathcca::Norm::Cublas());
      res_cublas_3= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Cublas, 128 > (dA, mathcca::Norm::Cublas());
      res_cublas_4= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Cublas, 256 > (dA, mathcca::Norm::Cublas());
      res_cublas_5= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Cublas, 512 > (dA, mathcca::Norm::Cublas());
      res_cublas_6= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Cublas, 1024> (dA, mathcca::Norm::Cublas());
      EXPECT_FLOAT_EQ(res, res_cublas_1);
      EXPECT_FLOAT_EQ(res, res_cublas_2);
      EXPECT_FLOAT_EQ(res, res_cublas_3);
      EXPECT_FLOAT_EQ(res, res_cublas_4);
      EXPECT_FLOAT_EQ(res, res_cublas_5);
      EXPECT_FLOAT_EQ(res, res_cublas_6);
#endif

      // https://en.wikipedia.org/wiki/Square_pyramidal_number
      if (i < 5) {
        std::cout << "C i= " << i << "\n";
        
	mathcca::fill_iota(dA.begin(), dA.end(), static_cast<value_type>(1));
        mathcca::copy(dA.cbegin(), dA.cend(), hA.begin());
        cudaDeviceSynchronize();
        
	value_type n1{static_cast<value_type>(r * c)};
        value_type n2{static_cast<value_type>(r * r * c * c)};
        value_type n3{static_cast<value_type>(r * r * r * c * c * c)};
        
	res= std::sqrt(n3/static_cast<value_type>(3) + n2/static_cast<value_type>(2) + n1/static_cast<value_type>(6));
        
	res_host= mathcca::frobenius_norm<mathcca::host_matrix<float>, mathcca::Norm::Base >(hA, mathcca::Norm::Base());
        
	res_base_1= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Base, 32  >(dA, mathcca::Norm::Base());
	res_base_2= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Base, 64  >(dA, mathcca::Norm::Base());
	res_base_3= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Base, 128 >(dA, mathcca::Norm::Base());
	res_base_4= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Base, 256 >(dA, mathcca::Norm::Base());
	res_base_5= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Base, 512 >(dA, mathcca::Norm::Base());
	res_base_6= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Base, 1024>(dA, mathcca::Norm::Base());
	EXPECT_NEAR(res, res_host, 0.2);
	EXPECT_FLOAT_EQ(res, res_base_1);
        EXPECT_FLOAT_EQ(res, res_base_2);
        EXPECT_FLOAT_EQ(res, res_base_3);
        EXPECT_FLOAT_EQ(res, res_base_4);
        EXPECT_FLOAT_EQ(res, res_base_5);
        EXPECT_FLOAT_EQ(res, res_base_6);

#ifdef _CUBLAS
        res_cublas_1= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Cublas, 32  > (dA, mathcca::Norm::Cublas());
        res_cublas_2= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Cublas, 64  > (dA, mathcca::Norm::Cublas());
        res_cublas_3= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Cublas, 128 > (dA, mathcca::Norm::Cublas());
        res_cublas_4= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Cublas, 256 > (dA, mathcca::Norm::Cublas());
        res_cublas_5= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Cublas, 512 > (dA, mathcca::Norm::Cublas());
        res_cublas_6= mathcca::frobenius_norm<mathcca::device_matrix<float>, mathcca::Norm::Cublas, 1024> (dA, mathcca::Norm::Cublas());
        EXPECT_FLOAT_EQ(res, res_cublas_1);
        EXPECT_FLOAT_EQ(res, res_cublas_2);
        EXPECT_FLOAT_EQ(res, res_cublas_3);
        EXPECT_FLOAT_EQ(res, res_cublas_4);
        EXPECT_FLOAT_EQ(res, res_cublas_5);
        EXPECT_FLOAT_EQ(res, res_cublas_6);
#endif
      }
      std::swap(r,c);
      r *= 5;
      c *= 2;
    }
}


