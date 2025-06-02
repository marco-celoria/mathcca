/*
 * SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include <mathcca.h>
#include <gtest/gtest.h>

TEST(ReduceSp, BasicAssertions)
{
    std::size_t r{2};
    std::size_t c{5};

    for (auto n= 1; n < 6; ++n) {

      using value_type= float;

      mathcca::device_matrix<value_type> dX{r, c};
      mathcca::host_matrix<value_type>   hX{r, c};
      
      mathcca::fill_rand(dX.begin(), dX.end());
      mathcca::copy(dX.cbegin(), dX.cend(), hX.begin());
      cudaDeviceSynchronize();

      value_type sumdX_0=  mathcca::reduce_sum(dX.begin(),  dX.end(),  static_cast<value_type>(1));
      value_type sumdX_C1= mathcca::detail::reduce_sum<value_type, 32  >(mathcca::Cuda(), dX.cbegin().get(), dX.cend().get(), static_cast<value_type>(1));
      value_type sumdX_C2= mathcca::detail::reduce_sum<value_type, 64  >(mathcca::Cuda(), dX.begin().get(),  dX.end().get(),  static_cast<value_type>(1));
      value_type sumdX_C3= mathcca::detail::reduce_sum<value_type, 128 >(mathcca::Cuda(), dX.cbegin().get(), dX.cend().get(), static_cast<value_type>(1));
      value_type sumdX_C4= mathcca::detail::reduce_sum<value_type, 256 >(mathcca::Cuda(), dX.begin().get(),  dX.end().get(),  static_cast<value_type>(1));
      value_type sumdX_C5= mathcca::detail::reduce_sum<value_type, 512 >(mathcca::Cuda(), dX.cbegin().get(), dX.cend().get(), static_cast<value_type>(1));
      value_type sumdX_C6= mathcca::detail::reduce_sum<value_type, 1024>(mathcca::Cuda(), dX.begin().get(),  dX.end().get(),  static_cast<value_type>(1));

#ifdef _THRUST

      value_type sumdX_T= mathcca::detail::reduce_sum(mathcca::Thrust(), dX.cbegin().get(), dX.cend().get(), static_cast<value_type>(1));

#endif

      value_type sumhX_0= mathcca::reduce_sum(hX.begin(), hX.end(), static_cast<value_type>(1));
      value_type sumhX_O= mathcca::detail::reduce_sum(mathcca::Omp(), hX.cbegin().get(), hX.cend().get(), static_cast<value_type>(1));

#ifdef _STDPAR

      value_type sumhX_S= mathcca::detail::reduce_sum(mathcca::StdPar(), hX.cbegin().get(), hX.cend().get(), static_cast<value_type>(1));

#endif

      EXPECT_NEAR(sumdX_0,  sumhX_0, 0.99);
      EXPECT_FLOAT_EQ(sumdX_0,  sumdX_C1);
      EXPECT_FLOAT_EQ(sumdX_C1, sumdX_C2);
      EXPECT_FLOAT_EQ(sumdX_C2, sumdX_C3);
      EXPECT_FLOAT_EQ(sumdX_C3, sumdX_C4);
      EXPECT_FLOAT_EQ(sumdX_C4, sumdX_C5);
      EXPECT_FLOAT_EQ(sumdX_C5, sumdX_C6);

#ifdef _THRUST

      EXPECT_FLOAT_EQ(sumdX_C6, sumdX_T);

#endif

      EXPECT_NEAR(sumhX_0,  sumhX_O, 0.1);

#ifdef _STDPAR      

      EXPECT_NEAR(sumhX_O,  sumhX_S, 0.1);

#endif
      
      mathcca::fill_const(dX.begin(), dX.end(), static_cast<value_type>(n));
      mathcca::fill_const(hX.begin(), hX.end(), static_cast<value_type>(n));
      
      sumdX_0=  mathcca::reduce_sum(dX.begin(),  dX.end(),  static_cast<value_type>(0));
      sumdX_C1= mathcca::detail::reduce_sum<value_type, 32  >(mathcca::Cuda(), dX.cbegin().get(), dX.cend().get(), static_cast<value_type>(0));
      sumdX_C2= mathcca::detail::reduce_sum<value_type, 64  >(mathcca::Cuda(), dX.begin().get(),  dX.end().get(),  static_cast<value_type>(0));
      sumdX_C3= mathcca::detail::reduce_sum<value_type, 128 >(mathcca::Cuda(), dX.cbegin().get(), dX.cend().get(), static_cast<value_type>(0));
      sumdX_C4= mathcca::detail::reduce_sum<value_type, 256 >(mathcca::Cuda(), dX.begin().get(),  dX.end().get(),  static_cast<value_type>(0));
      sumdX_C5= mathcca::detail::reduce_sum<value_type, 512 >(mathcca::Cuda(), dX.cbegin().get(), dX.cend().get(), static_cast<value_type>(0));
      sumdX_C6= mathcca::detail::reduce_sum<value_type, 1024>(mathcca::Cuda(), dX.begin().get(),  dX.end().get(),  static_cast<value_type>(0));

#ifdef _THRUST

      sumdX_T= mathcca::detail::reduce_sum(mathcca::Thrust(), dX.cbegin().get(), dX.cend().get(), static_cast<value_type>(0));

#endif

      sumhX_0= mathcca::reduce_sum(hX.begin(), hX.end(), static_cast<value_type>(0));
      sumhX_O= mathcca::detail::reduce_sum(mathcca::Omp(), hX.cbegin().get(), hX.cend().get(), static_cast<value_type>(0));

#ifdef _STDPAR

      sumhX_S= mathcca::detail::reduce_sum(mathcca::StdPar(), hX.cbegin().get(), hX.cend().get(), static_cast<value_type>(0));

#endif

      auto sX= static_cast<value_type>(dX.size());
      auto resX= static_cast<value_type>(n) * sX;

      EXPECT_FLOAT_EQ(sumdX_0,  sumhX_0);
      EXPECT_FLOAT_EQ(sumdX_0,  sumdX_C1);
      EXPECT_FLOAT_EQ(sumdX_C1, sumdX_C2);
      EXPECT_FLOAT_EQ(sumdX_C2, sumdX_C3);
      EXPECT_FLOAT_EQ(sumdX_C3, sumdX_C4);
      EXPECT_FLOAT_EQ(sumdX_C5, sumdX_C6);

#ifdef _THRUST

      EXPECT_FLOAT_EQ(sumdX_C6, sumdX_T);

#endif

      EXPECT_FLOAT_EQ(sumhX_0,  sumhX_O);

#ifdef _STDPAR      

      EXPECT_FLOAT_EQ(sumhX_O,  sumhX_S);

#endif

      EXPECT_FLOAT_EQ(sumdX_0, resX);
      EXPECT_FLOAT_EQ(sumhX_0, resX);

      if (n<4) {
      
        mathcca::fill_iota(dX.begin(),  dX.end(), static_cast<value_type>(1));
        mathcca::fill_iota(hX.begin(),  hX.end(), static_cast<value_type>(1));
        
	sumdX_0=  mathcca::reduce_sum(dX.begin(),  dX.end(),  static_cast<value_type>(0));
        sumdX_C1= mathcca::detail::reduce_sum<value_type, 32  >(mathcca::Cuda(), dX.cbegin().get(), dX.cend().get(), static_cast<value_type>(0));
        sumdX_C2= mathcca::detail::reduce_sum<value_type, 64  >(mathcca::Cuda(), dX.begin().get(),  dX.end().get(),  static_cast<value_type>(0));
        sumdX_C3= mathcca::detail::reduce_sum<value_type, 128 >(mathcca::Cuda(), dX.cbegin().get(), dX.cend().get(), static_cast<value_type>(0));
        sumdX_C4= mathcca::detail::reduce_sum<value_type, 256 >(mathcca::Cuda(), dX.begin().get(),  dX.end().get(),  static_cast<value_type>(0));
        sumdX_C5= mathcca::detail::reduce_sum<value_type, 512 >(mathcca::Cuda(), dX.cbegin().get(), dX.cend().get(), static_cast<value_type>(0));
        sumdX_C6= mathcca::detail::reduce_sum<value_type, 1024>(mathcca::Cuda(), dX.begin().get(),  dX.end().get(),  static_cast<value_type>(0));

#ifdef _THRUST

  	sumdX_T= mathcca::detail::reduce_sum(mathcca::Thrust(), dX.cbegin().get(), dX.cend().get(), static_cast<value_type>(0));

#endif
        sumhX_0= mathcca::reduce_sum(hX.begin(), hX.end(), static_cast<value_type>(0));
        sumhX_O= mathcca::detail::reduce_sum(mathcca::Omp(), hX.cbegin().get(), hX.cend().get(), static_cast<value_type>(0));

#ifdef _STDPAR
        
	sumhX_S= mathcca::detail::reduce_sum(mathcca::StdPar(), hX.cbegin().get(), hX.cend().get(), static_cast<value_type>(0));

#endif  
        
        sX= static_cast<value_type>(dX.size());
        resX= sX / (static_cast<value_type>(2)) * (sX + static_cast<value_type>(1));
         
        EXPECT_FLOAT_EQ(sumdX_0,  sumhX_0);
        EXPECT_FLOAT_EQ(sumdX_0,  sumdX_C1);
        EXPECT_FLOAT_EQ(sumdX_C1, sumdX_C2);
        EXPECT_FLOAT_EQ(sumdX_C2, sumdX_C3);
        EXPECT_FLOAT_EQ(sumdX_C3, sumdX_C4);
        EXPECT_FLOAT_EQ(sumdX_C5, sumdX_C6);
         
#ifdef _THRUST
         
        EXPECT_FLOAT_EQ(sumdX_C6, sumdX_T);
         
#endif
         
        EXPECT_FLOAT_EQ(sumhX_0,  sumhX_O);
        
#ifdef _STDPAR
         
        EXPECT_FLOAT_EQ(sumhX_O,  sumhX_S);
         
#endif  
        
        EXPECT_FLOAT_EQ(sumdX_0, resX);
        EXPECT_FLOAT_EQ(sumhX_0, resX);

      }
      
      std::swap(r,c);
      r*= 5;
      c*= 2;
    
    }

}


