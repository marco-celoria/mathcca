#include <mathcca.hpp>
#include <gtest/gtest.h>

TEST(CopyDp, BasicAssertions)
{
    std::size_t r{2};
    std::size_t c{5};
    for (auto n= 1; n < 9; ++n) {

      using value_type= double;

      mathcca::device_matrix<value_type> dA{r, c, static_cast<value_type>(n)};
      mathcca::device_matrix<value_type> dB{r, c, static_cast<value_type>(n+1)};
      mathcca::device_matrix<value_type> dC{r, c};
      mathcca::device_matrix<value_type> dD{r, c, static_cast<value_type>(n)};
      mathcca::device_matrix<value_type> dE{r, c, static_cast<value_type>(n+1)};
      
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
      
      mathcca::host_matrix<value_type> hA{r, c};
      mathcca::host_matrix<value_type> hB{r, c};
      mathcca::host_matrix<value_type> hC{r, c};
      mathcca::host_matrix<value_type> hD{r, c};
      mathcca::host_matrix<value_type> hE{r, c};
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
      cudaStreamSynchronize(s_A);
      
      mathcca::copy(dB.cbegin(), dB.cend(), hB.begin(), s_B);
      cudaStreamSynchronize(s_B);
      
      mathcca::copy(dC.begin(),  dC.end(),  hC.begin(), s_C);
      cudaStreamSynchronize(s_C);
      
      mathcca::copy(dD.cbegin(), dD.cend(), hD.begin(), s_D);
      cudaStreamSynchronize(s_D);
      
      mathcca::copy(dE.begin(),  dE.end(),  hE.begin(), s_E);
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
      EXPECT_TRUE(dA == dD);
      cudaStreamDestroy(s_A);
      cudaStreamDestroy(s_D);
   
      mathcca::copy(hA.begin(),  hA.end() , hE.begin());
      mathcca::copy(hA.begin(),  hA.end() , hC.begin());
      EXPECT_TRUE(hA == hE);
      EXPECT_TRUE(hA == hC);


      mathcca::device_matrix<value_type> dCUDA{dA.num_rows(), dA.num_cols()};
      mathcca::detail::copy(mathcca::Cuda(), dA.cbegin().get(), dA.cend().get(), dCUDA.begin().get());
      EXPECT_TRUE(dA == dCUDA);

#ifdef _THRUST
      mathcca::device_matrix<value_type> dTHRUST{dA.num_rows(), dA.num_cols()};
      mathcca::detail::copy(mathcca::Thrust(), dA.cbegin().get(), dA.cend().get(), dTHRUST.begin().get());
      EXPECT_TRUE(dA == dTHRUST);
#endif

      mathcca::host_matrix<value_type> hOMP{hA.num_rows(), hA.num_cols()};
      mathcca::detail::copy(mathcca::Omp(), hA.begin().get(), hA.end().get(), hOMP.begin().get());
      EXPECT_TRUE(hA == hOMP);

#ifdef _STDPAR
      mathcca::host_matrix<value_type> hSTDPAR{hA.num_rows(), hA.num_cols()};
      mathcca::detail::copy(mathcca::StdPar(), hA.cbegin().get(), hA.cend().get(), hSTDPAR.begin().get());
      EXPECT_TRUE(hA == hSTDPAR);      
#endif      

      std::swap(r,c);
      r*= 5;
      c*= 2;
    }
}


