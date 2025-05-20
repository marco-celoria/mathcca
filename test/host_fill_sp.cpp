#include <mathcca.hpp>
#include <gtest/gtest.h>

TEST(FillSp, BasicAssertions)
{
    std::size_t r{2};
    std::size_t c{5};
    for (auto n= 1; n < 9; ++n) {
      mathcca::host_matrix<float> A{r, c};
      mathcca::host_matrix<float> X0{r, c};
      mathcca::host_matrix<float> X1{r, c};
      mathcca::host_matrix<float> X2{r, c};
      mathcca::host_matrix<float> X3{r, c};
      mathcca::host_matrix<float> X4{r, c};
      mathcca::host_matrix<float> X5{r, c};
      mathcca::host_matrix<float> X6{r, c};
      mathcca::host_matrix<float> X7{r, c};
      mathcca::host_matrix<float> X8{r, c};
      mathcca::host_matrix<float> X9{r, c};
      using value_type= typename decltype(A)::value_type;
      mathcca::fill_const(A.begin(), A.end(), static_cast<value_type>(n));
      mathcca::copy(A.begin(), A.end(), A.begin());
      for (std::size_t i= 0; i < A.size(); ++i) {
        EXPECT_FLOAT_EQ(A[i], static_cast<value_type>(n));
      }

      mathcca::fill_iota(A.begin(), A.end(), static_cast<value_type>(10));

      for (std::size_t i= 0; i < A.size(); ++i) {
        EXPECT_FLOAT_EQ(A[i], static_cast<value_type>(10 + i));
      }
      
      mathcca::fill_rand(X0.begin(), X0.end()); 
      mathcca::fill_rand(X1.begin(), X1.end());
      mathcca::fill_rand(X2.begin(), X2.end());
      mathcca::fill_rand(X3.begin(), X3.end());
      mathcca::fill_rand(X4.begin(), X4.end());
      mathcca::fill_rand(X5.begin(), X5.end());
      mathcca::fill_rand(X6.begin(), X6.end());
      mathcca::fill_rand(X7.begin(), X7.end()); 
      mathcca::fill_rand(X8.begin(), X8.end());
      mathcca::fill_rand(X9.begin(), X9.end());

      EXPECT_TRUE(X0 != X1);
      EXPECT_TRUE(X0 != X2);
      EXPECT_TRUE(X0 != X3);
      EXPECT_TRUE(X0 != X4);
      EXPECT_TRUE(X0 != X5);
      EXPECT_TRUE(X0 != X6);
      EXPECT_TRUE(X0 != X7);
      EXPECT_TRUE(X0 != X8);
      EXPECT_TRUE(X0 != X9);
      
      EXPECT_TRUE(X1 != X2);
      EXPECT_TRUE(X1 != X3);
      EXPECT_TRUE(X1 != X4);
      EXPECT_TRUE(X1 != X5);
      EXPECT_TRUE(X1 != X6);
      EXPECT_TRUE(X1 != X7);
      EXPECT_TRUE(X1 != X8);
      EXPECT_TRUE(X1 != X9);
      
      EXPECT_TRUE(X2 != X3);
      EXPECT_TRUE(X2 != X4);
      EXPECT_TRUE(X2 != X5);
      EXPECT_TRUE(X2 != X6);
      EXPECT_TRUE(X2 != X7);
      EXPECT_TRUE(X2 != X8);
      EXPECT_TRUE(X2 != X9);
     
      EXPECT_TRUE(X3 != X4);
      EXPECT_TRUE(X3 != X5);
      EXPECT_TRUE(X3 != X6);
      EXPECT_TRUE(X3 != X7);
      EXPECT_TRUE(X3 != X8);
      EXPECT_TRUE(X3 != X9);

      EXPECT_TRUE(X4 != X5);
      EXPECT_TRUE(X4 != X6);
      EXPECT_TRUE(X4 != X7);
      EXPECT_TRUE(X4 != X8);
      EXPECT_TRUE(X4 != X9);

      EXPECT_TRUE(X5 != X6);
      EXPECT_TRUE(X5 != X7);
      EXPECT_TRUE(X5 != X8);
      EXPECT_TRUE(X5 != X9);

      EXPECT_TRUE(X6 != X7);
      EXPECT_TRUE(X6 != X8);
      EXPECT_TRUE(X6 != X9);
      
      EXPECT_TRUE(X7 != X8);
      EXPECT_TRUE(X7 != X9);

      EXPECT_TRUE(X8 != X9);
      
      std::swap(r,c);
      r*= 5;
      c*= 2;
    }
}


