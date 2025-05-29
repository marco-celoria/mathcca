#include <mathcca.hpp>
#include <gtest/gtest.h>

TEST(MatmulDp, BasicAssertions)
{
    std::size_t l{5};
    std::size_t m{3};
    std::size_t n{2};
    for (auto i= 1; i < 8; ++i) {

      using value_type= double;

      mathcca::device_matrix<value_type> dX0{l, m};
      mathcca::host_matrix<value_type>   hX0{l, m};
      mathcca::device_matrix<value_type> dY0{m, n};
      mathcca::host_matrix<value_type>   hY0{m, n};
      mathcca::device_matrix<value_type> dB0{l, n};
      mathcca::host_matrix<value_type>   hB0{l, n};
      mathcca::device_matrix<value_type> dT0{l, n};
      mathcca::host_matrix<value_type>   hT0{l, n};
      mathcca::device_matrix<value_type> dC0{l, n};
      mathcca::device_matrix<value_type> dERR{99, 99};
      mathcca::host_matrix<value_type>   hERR{99, 99};

      mathcca::fill_rand(dX0.begin(),  dX0.end());
      mathcca::copy(     dX0.cbegin(), dX0.cend(), hX0.begin());
      cudaDeviceSynchronize();
      
      mathcca::fill_rand(dY0.begin(),  dY0.end());
      mathcca::copy(     dY0.cbegin(), dY0.cend(), hY0.begin());
      cudaDeviceSynchronize();
      
      EXPECT_TRUE(dX0 != dY0);
      EXPECT_TRUE(hX0 != hY0);
      
      EXPECT_THROW({mathcca::matmul(dX0, dERR, mathcca::MM::Base());},  std::length_error);
      EXPECT_THROW({mathcca::matmul(hX0, hERR, mathcca::MM::Base());},  std::length_error);
      EXPECT_THROW({mathcca::matmul(dX0, dERR, mathcca::MM::Tiled());}, std::length_error);
      EXPECT_THROW({mathcca::matmul(hX0, hERR, mathcca::MM::Tiled());}, std::length_error);
      
      mathcca::matmul<mathcca::device_matrix<value_type>, mathcca::MM::Base, 8>(dX0, dY0, dB0, mathcca::MM::Base());
      mathcca::matmul<mathcca::host_matrix<value_type>, mathcca::MM::Base>   (hX0, hY0, hB0,  mathcca::MM::Base());
      auto dB1= mathcca::matmul<mathcca::device_matrix<value_type>, mathcca::MM::Base, 16>(dX0, dY0, mathcca::MM::Base());
      auto dB2= mathcca::matmul<mathcca::device_matrix<value_type>, mathcca::MM::Base, 32>(dX0, dY0, mathcca::MM::Base());
      auto hB1= mathcca::matmul<mathcca::host_matrix<value_type>, mathcca::MM::Base>    (hX0, hY0, mathcca::MM::Base());
      
      EXPECT_TRUE(dB0 == dB1);
      EXPECT_TRUE(dB1 == dB2);
      
      EXPECT_TRUE(hB0 == hB1);
      
      mathcca::matmul<mathcca::device_matrix<value_type>, mathcca::MM::Tiled, 8>(dX0, dY0, dT0, mathcca::MM::Tiled());
      mathcca::matmul<mathcca::host_matrix<value_type>, mathcca::MM::Tiled>   (hX0, hY0, hT0, mathcca::MM::Tiled());
      auto dT1= mathcca::matmul<mathcca::device_matrix<value_type>, mathcca::MM::Tiled, 16>(dX0, dY0, mathcca::MM::Tiled());
      auto dT2= mathcca::matmul<mathcca::device_matrix<value_type>, mathcca::MM::Tiled, 32>(dX0, dY0, mathcca::MM::Tiled());
      auto hT1= mathcca::matmul<mathcca::host_matrix<value_type>, mathcca::MM::Tiled>    (hX0, hY0, mathcca::MM::Tiled());
      
      EXPECT_TRUE(dT0 == dT1);
      EXPECT_TRUE(dT1 == dT2);
      
      EXPECT_TRUE(dT0 == dB0);
      EXPECT_TRUE(dT1 == dB1);
      EXPECT_TRUE(dT2 == dB2);
      
      EXPECT_TRUE(hT0 == hT1);
      EXPECT_TRUE(hT0 == hB0);
      EXPECT_TRUE(hT1 == hB1);

      auto dRB{dB0};
      auto dRT{dT0};

      mathcca::copy(hB0.cbegin(), hB0.cend(), dRB.begin());
     
      mathcca::copy(hT0.cbegin(), hT0.cend(), dRT.begin());

      EXPECT_TRUE(dB0 == dRB);
      EXPECT_TRUE(dT0 == dRT);

#ifdef _CUBLAS

      mathcca::matmul<mathcca::device_matrix<value_type>, mathcca::MM::Cublas, 8>(dX0, dY0, dC0, mathcca::MM::Cublas());
      auto dC1 = mathcca::matmul<mathcca::device_matrix<value_type>, mathcca::MM::Cublas, 16>(dX0, dY0, mathcca::MM::Cublas());
      auto dC2 = mathcca::matmul<mathcca::device_matrix<value_type>, mathcca::MM::Cublas, 32>(dX0, dY0, mathcca::MM::Cublas());
      
      EXPECT_TRUE(dC0 == dC1);
      EXPECT_TRUE(dC1 == dC2);

      EXPECT_TRUE(dC0 == dB0);
      EXPECT_TRUE(dC1 == dB1);
      EXPECT_TRUE(dC2 == dB2);

      EXPECT_TRUE(dC0 == dT0);
      EXPECT_TRUE(dC1 == dT1);
      EXPECT_TRUE(dC2 == dT2);      

#endif

      std::swap(l,n);
      l *= 5;
      m *= 3;
      n *= 2;
  }
}


