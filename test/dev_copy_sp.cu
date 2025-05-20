#include <mathcca.hpp>
#include <gtest/gtest.h>

TEST(CopySp, BasicAssertions)
{
    std::size_t r{2};
    std::size_t c{5};
    for (auto n= 1; n < 9; ++n) {
      mathcca::device_matrix<float> dA{r, c, static_cast<float>(n)};
      mathcca::device_matrix<float> dB{r, c, static_cast<float>(n+1)};
      mathcca::device_matrix<float> dC{r, c};
      mathcca::device_matrix<float> dD{r, c, static_cast<float>(n)};
      mathcca::device_matrix<float> dE{r, c, static_cast<float>(n+1)};
      using value_type= typename decltype(dA)::value_type;
      
      EXPECT_TRUE(dA != dB);
      EXPECT_TRUE(dA != dC);
      EXPECT_TRUE(dB != dC);
      EXPECT_TRUE(dA == dD);
      EXPECT_TRUE(dB == dE);
      
      mathcca::copy(dA.begin(),  dA.end(),  dB.begin());
      mathcca::copy(dA.cbegin(), dA.cend(), dC.begin());
      EXPECT_TRUE(dA == dB);
      EXPECT_TRUE(dA == dC);
      EXPECT_TRUE(dB == dC);
      EXPECT_TRUE(dA == dD);
      EXPECT_TRUE(dB != dE);
      
      mathcca::fill_const(dA.begin(), dA.end(), static_cast<value_type>(n + decltype(dA)::tol() / 10.));
      mathcca::fill_const(dB.begin(), dB.end(), static_cast<value_type>(n + decltype(dA)::tol() * 2.));
      mathcca::fill_const(dC.begin(), dC.end(), static_cast<value_type>(n + 1));
      EXPECT_TRUE(dA != dB);
      EXPECT_TRUE(dA != dC);
      EXPECT_TRUE(dA == dD);
      EXPECT_TRUE(dA != dE);
      EXPECT_TRUE(dC == dE);
      
      mathcca::host_matrix<float> hA{r, c};
      mathcca::host_matrix<float> hB{r, c};
      mathcca::host_matrix<float> hC{r, c};
      mathcca::host_matrix<float> hD{r, c};
      mathcca::host_matrix<float> hE{r, c};
      cudaStream_t s_A;
      cudaStream_t s_B;
      cudaStream_t s_C;
      cudaStream_t s_D;
      cudaStream_t s_E;
      cudaStreamCreate(&s_A);
      cudaStreamCreate(&s_B);
      cudaStreamCreate(&s_C);
      cudaStreamCreate(&s_D);
      cudaStreamCreate(&s_E);

      mathcca::copy(dA.begin(),  dA.end() , hA.begin(), s_A);
      mathcca::copy(dB.cbegin(), dB.cend(), hB.begin(), s_B);
      mathcca::copy(dC.begin(),  dC.end(),  hC.begin(), s_C);
      mathcca::copy(dD.cbegin(), dD.cend(), hD.begin(), s_D);
      mathcca::copy(dE.begin(),  dE.end(),  hE.begin(), s_E);
      cudaStreamSynchronize(s_A);
      cudaStreamSynchronize(s_B);
      cudaStreamSynchronize(s_C);
      cudaStreamSynchronize(s_D);
      cudaStreamSynchronize(s_E);
      cudaStreamDestroy(s_B);
      cudaStreamDestroy(s_C);
      cudaStreamDestroy(s_E);
      EXPECT_TRUE(hA != hB);
      EXPECT_TRUE(hA != hC);
      EXPECT_TRUE(hA == hD);
      EXPECT_TRUE(hA != hE);
      EXPECT_TRUE(hC == hE);
      
      dA*= static_cast<value_type>(2);
      EXPECT_TRUE(dA != dD);
      EXPECT_TRUE(hA == hD);
      mathcca::copy(dA.begin(),  dA.end() , hA.begin(), s_A);
      cudaStreamSynchronize(s_A);
      EXPECT_TRUE(hA != hD);
      hD*= static_cast<value_type>(2);
      EXPECT_TRUE(hA == hD);
      EXPECT_TRUE(dA != dD);
      mathcca::copy(hD.begin(),  hD.end() , dD.begin(), s_D);
      cudaStreamSynchronize(s_D);
      EXPECT_TRUE(dA == dD);
      cudaStreamDestroy(s_A);
      cudaStreamDestroy(s_D);
      
      std::swap(r,c);
      r*= 5;
      c*= 2;
    }
}


