#include <mathcca.hpp>
#include <gtest/gtest.h>

TEST(FillSp, BasicAssertions)
{
    std::size_t r{2};
    std::size_t c{5};
    for (auto n= 1; n < 9; ++n) {

      using value_type= float;
      
      mathcca::device_matrix<value_type> dA{r, c};
      mathcca::host_matrix<value_type>   hA{r, c};
      mathcca::device_matrix<value_type> dX0{r, c};
      mathcca::device_matrix<value_type> dX1{r, c};
      mathcca::device_matrix<value_type> dX2{r, c};
      mathcca::device_matrix<value_type> dX3{r, c};
      mathcca::device_matrix<value_type> dX4{r, c};
      mathcca::device_matrix<value_type> dX5{r, c};
      mathcca::device_matrix<value_type> dX6{r, c};
      mathcca::device_matrix<value_type> dX7{r, c};
      mathcca::device_matrix<value_type> dX8{r, c};
      mathcca::device_matrix<value_type> dX9{r, c};

      cudaStream_t s_A;
      cudaStreamCreate(&s_A);
      
      mathcca::fill_const(dA.begin(), dA.end(), static_cast<value_type>(n));
      mathcca::copy(dA.begin(), dA.end(), hA.begin(), s_A);
      cudaStreamSynchronize(s_A);
      
      for (std::size_t i= 0; i < hA.size(); ++i) {
        EXPECT_FLOAT_EQ(hA[i], static_cast<value_type>(n));
      }


      mathcca::detail::fill_const(mathcca::Cuda(), dA.begin().get(), dA.end().get(), static_cast<value_type>(n+1));
      mathcca::copy(dA.begin(), dA.end(), hA.begin(), s_A);
      cudaStreamSynchronize(s_A);
      
      for (std::size_t i= 0; i < hA.size(); ++i) {
        EXPECT_FLOAT_EQ(hA[i], static_cast<value_type>(n+1));
      }

#ifdef _THRUST

      mathcca::detail::fill_const(mathcca::Thrust(), dA.begin().get(), dA.end().get(), static_cast<value_type>(n+2));
      mathcca::copy(dA.begin(), dA.end(), hA.begin(), s_A);
      cudaStreamSynchronize(s_A);
      
      for (std::size_t i= 0; i < hA.size(); ++i) {
        EXPECT_FLOAT_EQ(hA[i], static_cast<value_type>(n+2));
      }

#endif

      mathcca::fill_iota(dA.begin(), dA.end(), static_cast<value_type>(10));
      mathcca::copy(dA.begin(), dA.end(), hA.begin(), s_A);
      cudaStreamSynchronize(s_A);

      for (std::size_t i= 0; i < hA.size(); ++i) {
        EXPECT_FLOAT_EQ(hA[i], static_cast<value_type>(10 + i));
      }
      
      mathcca::detail::fill_iota(mathcca::Cuda(), dA.begin().get(), dA.end().get(), static_cast<value_type>(11));
      mathcca::copy(dA.begin(), dA.end(), hA.begin(), s_A);
      cudaStreamSynchronize(s_A);

      for (std::size_t i= 0; i < hA.size(); ++i) {
        EXPECT_FLOAT_EQ(hA[i], static_cast<value_type>(11 + i));
      }
      
#ifdef _THRUST

      mathcca::detail::fill_iota(mathcca::Thrust(), dA.begin().get(), dA.end().get(), static_cast<value_type>(12));
      mathcca::copy(dA.begin(), dA.end(), hA.begin(), s_A);
      cudaStreamSynchronize(s_A);

      for (std::size_t i= 0; i < hA.size(); ++i) {
        EXPECT_FLOAT_EQ(hA[i], static_cast<value_type>(12 + i));
      }

#endif
      
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
      
      mathcca::copy(dX0.begin(), dX0.end(), hA.begin(), s_A);
      cudaStreamSynchronize(s_A);

      std::size_t same=0;
      for (std::size_t i= 0; i < hA.size() - 1; ++i) {
        if (hA[i] == hA[i+1]) {
          ++same;
	}
      }
      std::cout << "SAME= " << same << "\n";        
      EXPECT_TRUE(same < 10);


      mathcca::detail::fill_rand(mathcca::Cuda(), dX0.begin().get(), dX0.end().get()); 
      mathcca::detail::fill_rand(mathcca::Cuda(), dX1.begin().get(), dX1.end().get());
      mathcca::detail::fill_rand(mathcca::Cuda(), dX2.begin().get(), dX2.end().get());
      mathcca::detail::fill_rand(mathcca::Cuda(), dX3.begin().get(), dX3.end().get());
      mathcca::detail::fill_rand(mathcca::Cuda(), dX4.begin().get(), dX4.end().get());
      mathcca::detail::fill_rand(mathcca::Cuda(), dX5.begin().get(), dX5.end().get());
      mathcca::detail::fill_rand(mathcca::Cuda(), dX6.begin().get(), dX6.end().get());
      mathcca::detail::fill_rand(mathcca::Cuda(), dX7.begin().get(), dX7.end().get()); 
      mathcca::detail::fill_rand(mathcca::Cuda(), dX8.begin().get(), dX8.end().get());
      mathcca::detail::fill_rand(mathcca::Cuda(), dX9.begin().get(), dX9.end().get());

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

      mathcca::copy(dX0.begin(), dX0.end(), hA.begin(), s_A);
      cudaStreamSynchronize(s_A);

      same=0;
      for (std::size_t i= 0; i < hA.size() - 1; ++i) {
        if (hA[i] == hA[i+1]) {
          ++same;
        }
      }

      std::cout << "SAME= " << same << "\n";        
      EXPECT_TRUE(same < 10);



#ifdef _THRUST

      mathcca::detail::fill_rand(mathcca::Thrust(), dX0.begin().get(), dX0.end().get());
      mathcca::detail::fill_rand(mathcca::Thrust(), dX1.begin().get(), dX1.end().get());
      mathcca::detail::fill_rand(mathcca::Thrust(), dX2.begin().get(), dX2.end().get());
      mathcca::detail::fill_rand(mathcca::Thrust(), dX3.begin().get(), dX3.end().get());
      mathcca::detail::fill_rand(mathcca::Thrust(), dX4.begin().get(), dX4.end().get());
      mathcca::detail::fill_rand(mathcca::Thrust(), dX5.begin().get(), dX5.end().get());
      mathcca::detail::fill_rand(mathcca::Thrust(), dX6.begin().get(), dX6.end().get());
      mathcca::detail::fill_rand(mathcca::Thrust(), dX7.begin().get(), dX7.end().get());
      mathcca::detail::fill_rand(mathcca::Thrust(), dX8.begin().get(), dX8.end().get());
      mathcca::detail::fill_rand(mathcca::Thrust(), dX9.begin().get(), dX9.end().get());

      EXPECT_TRUE(dX0 != dX1);
      EXPECT_TRUE(dX0 != dX2);
      EXPECT_TRUE(dX0 != dX3);
      EXPECT_TRUE(dX0 != dX4);
      EXPECT_TRUE(dX0 != dX5);
      EXPECT_TRUE(dX0 != dX6);
      EXPECT_TRUE(dX0 != dX7);
      EXPECT_TRUE(dX0 != dX8);
      EXPECT_TRUE(dX0 != dX9);

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

      mathcca::copy(dX0.begin(), dX0.end(), hA.begin(), s_A);
      cudaStreamSynchronize(s_A);

      same=0;
      for (std::size_t i= 0; i < hA.size() - 1; ++i) {
        if (hA[i] == hA[i+1]) {
          ++same;
        } 
      } 
      std::cout << "SAME= " << same << "\n";        
      EXPECT_TRUE(same < 10);
      
#endif

      cudaStreamDestroy(s_A);
      std::swap(r,c);
      r*= 5;
      c*= 2;
    }
}


