#include <mathcca.hpp>
#include <gtest/gtest.h>

TEST(FillDp, BasicAssertions)
{
    std::size_t r{2};
    std::size_t c{5};
    for (auto n= 1; n < 9; ++n) {
      mathcca::device_matrix<double> dA{r, c};
      mathcca::host_matrix<double> hA{r, c};
      mathcca::device_matrix<double> dX0{r, c};
      mathcca::device_matrix<double> dX1{r, c};
      mathcca::device_matrix<double> dX2{r, c};
      mathcca::device_matrix<double> dX3{r, c};
      mathcca::device_matrix<double> dX4{r, c};
      mathcca::device_matrix<double> dX5{r, c};
      mathcca::device_matrix<double> dX6{r, c};
      mathcca::device_matrix<double> dX7{r, c};
      mathcca::device_matrix<double> dX8{r, c};
      mathcca::device_matrix<double> dX9{r, c};
      using value_type= typename decltype(dA)::value_type;
      mathcca::fill_const(dA.begin(), dA.end(), static_cast<value_type>(n));
      cudaStream_t s_A;
      cudaStreamCreate(&s_A);
      
      mathcca::copy(dA.begin(),  dA.end() , hA.begin(), s_A);
      cudaStreamSynchronize(s_A);
      
      for (std::size_t i=0; i < hA.size(); ++i) {
        EXPECT_FLOAT_EQ(hA[i], static_cast<value_type>(n));
      }

      mathcca::fill_iota(dA.begin(), dA.end(), static_cast<value_type>(10));
      
      mathcca::copy(dA.begin(),  dA.end() , hA.begin(), s_A);
      cudaStreamSynchronize(s_A);
      
      cudaStreamDestroy(s_A);

      for (std::size_t i=0; i < hA.size(); ++i) {
        EXPECT_FLOAT_EQ(hA[i], static_cast<value_type>(10 + i));
      }
      
      mathcca::fill_rand(dX0.begin(), dX0.end()); 
      mathcca::fill_rand(dX1.begin(), dX1.end()); 
      mathcca::fill_rand(dX2.begin(), dX2.end()); 
      mathcca::fill_rand(dX3.begin(), dX3.end()); 
      mathcca::fill_rand(dX4.begin(), dX4.end()); 
      mathcca::fill_rand(dX5.begin(), dX5.end()); 
      mathcca::fill_rand(dX6.begin(), dX6.end()); 
      mathcca::fill_rand(dX7.begin(), dX7.end()); 
      mathcca::fill_rand(dX8.begin(), dX8.end()); 
      mathcca::fill_rand(dX9.begin(), dX9.end()); 

      EXPECT_TRUE(dX0 != dX1);
      EXPECT_TRUE(dX0 != dX2);
      EXPECT_TRUE(dX0 != dX3);
      EXPECT_TRUE(dX0 != dX4);
      EXPECT_TRUE(dX0 != dX5);
      EXPECT_TRUE(dX0 != dX6);
      EXPECT_TRUE(dX0 != dX7);
      EXPECT_TRUE(dX0 != dX8);
      EXPECT_TRUE(dX0 != dX9);
      
      EXPECT_TRUE(dX1 != dX2);
      EXPECT_TRUE(dX1 != dX3);
      EXPECT_TRUE(dX1 != dX4);
      EXPECT_TRUE(dX1 != dX5);
      EXPECT_TRUE(dX1 != dX6);
      EXPECT_TRUE(dX1 != dX7);
      EXPECT_TRUE(dX1 != dX8);
      EXPECT_TRUE(dX1 != dX9);
      
      EXPECT_TRUE(dX2 != dX3);
      EXPECT_TRUE(dX2 != dX4);
      EXPECT_TRUE(dX2 != dX5);
      EXPECT_TRUE(dX2 != dX6);
      EXPECT_TRUE(dX2 != dX7);
      EXPECT_TRUE(dX2 != dX8);
      EXPECT_TRUE(dX2 != dX9);
     
      EXPECT_TRUE(dX3 != dX4);
      EXPECT_TRUE(dX3 != dX5);
      EXPECT_TRUE(dX3 != dX6);
      EXPECT_TRUE(dX3 != dX7);
      EXPECT_TRUE(dX3 != dX8);
      EXPECT_TRUE(dX3 != dX9);

      EXPECT_TRUE(dX4 != dX5);
      EXPECT_TRUE(dX4 != dX6);
      EXPECT_TRUE(dX4 != dX7);
      EXPECT_TRUE(dX4 != dX8);
      EXPECT_TRUE(dX4 != dX9);

      EXPECT_TRUE(dX5 != dX6);
      EXPECT_TRUE(dX5 != dX7);
      EXPECT_TRUE(dX5 != dX8);
      EXPECT_TRUE(dX5 != dX9);

      EXPECT_TRUE(dX6 != dX7);
      EXPECT_TRUE(dX6 != dX8);
      EXPECT_TRUE(dX6 != dX9);
      
      EXPECT_TRUE(dX7 != dX8);
      EXPECT_TRUE(dX7 != dX9);

      EXPECT_TRUE(dX8 != dX9);
      
      std::swap(r,c);
      r*= 5;
      c*= 2;
    }
}


