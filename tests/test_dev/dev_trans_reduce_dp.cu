#include <mathcca.hpp>
#include <gtest/gtest.h>
#include <numbers>

TEST(TransReduceDp, BasicAssertions)
{

    std::size_t r{5};
    std::size_t c{2};
    
    for (auto i= 1; i < 9; ++i) {
    
      using value_type= double;
      using square_type= mathcca::detail::Square<value_type>;
      using invsquare_type= mathcca::detail::InverseSquare<value_type>;
      
      mathcca::device_matrix<value_type> dX{r, c};
      mathcca::host_matrix<value_type>   hX{r, c};

      mathcca::fill_rand(dX.begin(), dX.end());
      mathcca::copy(dX.cbegin(), dX.cend(), hX.begin());
      cudaDeviceSynchronize();

      value_type sumdX_0= mathcca::transform_reduce_sum(dX.begin(),  dX.end(), mathcca::detail::Square<value_type>(),  static_cast<value_type>(1));
      value_type sumdX_C1= mathcca::detail::transform_reduce_sum<value_type, square_type, 32  >(mathcca::Cuda(), dX.cbegin().get(), dX.cend().get(), mathcca::detail::Square<value_type>(),  static_cast<value_type>(1));
      value_type sumdX_C2= mathcca::detail::transform_reduce_sum<value_type, square_type, 64  >(mathcca::Cuda(), dX.begin().get(),  dX.end().get(),  mathcca::detail::Square<value_type>(),  static_cast<value_type>(1));
      value_type sumdX_C3= mathcca::detail::transform_reduce_sum<value_type, square_type, 128 >(mathcca::Cuda(), dX.cbegin().get(), dX.cend().get(), mathcca::detail::Square<value_type>(),  static_cast<value_type>(1));
      value_type sumdX_C4= mathcca::detail::transform_reduce_sum<value_type, square_type, 256 >(mathcca::Cuda(), dX.begin().get(),  dX.end().get(),  mathcca::detail::Square<value_type>(),  static_cast<value_type>(1));
      value_type sumdX_C5= mathcca::detail::transform_reduce_sum<value_type, square_type, 512 >(mathcca::Cuda(), dX.cbegin().get(), dX.cend().get(), mathcca::detail::Square<value_type>(),  static_cast<value_type>(1));
      value_type sumdX_C6= mathcca::detail::transform_reduce_sum<value_type, square_type, 1024>(mathcca::Cuda(), dX.begin().get(),  dX.end().get(),  mathcca::detail::Square<value_type>(),  static_cast<value_type>(1));

#ifdef _THRUST

      value_type sumdX_T= mathcca::detail::transform_reduce_sum(mathcca::Thrust(), dX.cbegin().get(), dX.cend().get(), mathcca::detail::Square<value_type>(), static_cast<value_type>(1));

#endif

      value_type sumhX_0= mathcca::transform_reduce_sum(hX.begin(), hX.end(), mathcca::detail::Square<value_type>(), static_cast<value_type>(1));
      value_type sumhX_O= mathcca::detail::transform_reduce_sum(mathcca::Omp(), hX.cbegin().get(), hX.cend().get(), mathcca::detail::Square<value_type>(), static_cast<value_type>(1));

#ifdef _STDPAR

      value_type sumhX_S= mathcca::detail::transform_reduce_sum(mathcca::StdPar(), hX.cbegin().get(), hX.cend().get(), mathcca::detail::Square<value_type>(), static_cast<value_type>(1));

#endif

      EXPECT_NEAR(sumdX_0,  sumhX_0, 0.49);
      EXPECT_FLOAT_EQ(sumdX_0,  sumdX_C1);
      EXPECT_FLOAT_EQ(sumdX_C1, sumdX_C2);
      EXPECT_FLOAT_EQ(sumdX_C2, sumdX_C3);
      EXPECT_FLOAT_EQ(sumdX_C3, sumdX_C4);
      EXPECT_FLOAT_EQ(sumdX_C4, sumdX_C5);
      EXPECT_FLOAT_EQ(sumdX_C5, sumdX_C6);

#ifdef _THRUST

      EXPECT_FLOAT_EQ(sumdX_C6, sumdX_T);

#endif

      EXPECT_FLOAT_EQ(sumhX_0,  sumhX_O);

#ifdef _STDPAR      

      EXPECT_FLOAT_EQ(sumhX_O,  sumhX_S);

#endif


      mathcca::fill_const(dX.begin(), dX.end(), static_cast<value_type>(3));
      mathcca::fill_const(hX.begin(), hX.end(), static_cast<value_type>(3));

      auto init= static_cast<value_type>(0);

      using iter_type= decltype(dX.begin());
      using const_iter_type= decltype(dX.cbegin());

      sumdX_0=  mathcca::transform_reduce_sum(dX.begin(),  dX.end(), mathcca::detail::Square<value_type>(),  init);
      sumdX_C1= mathcca::detail::transform_reduce_sum<value_type, square_type, 32  >(mathcca::Cuda(), dX.cbegin().get(), dX.cend().get(), mathcca::detail::Square<value_type>(), init);
      sumdX_C2= mathcca::detail::transform_reduce_sum<value_type, square_type, 64  >(mathcca::Cuda(), dX.begin().get(),  dX.end().get(), mathcca::detail::Square<value_type>(),  init);
      sumdX_C3= mathcca::detail::transform_reduce_sum<value_type, square_type, 128 >(mathcca::Cuda(), dX.cbegin().get(), dX.cend().get(), mathcca::detail::Square<value_type>(), init);
      sumdX_C4= mathcca::detail::transform_reduce_sum<value_type, square_type, 256 >(mathcca::Cuda(), dX.begin().get(),  dX.end().get(), mathcca::detail::Square<value_type>(),  init);
      sumdX_C5= mathcca::detail::transform_reduce_sum<value_type, square_type, 512 >(mathcca::Cuda(), dX.cbegin().get(), dX.cend().get(), mathcca::detail::Square<value_type>(), init);
      sumdX_C6= mathcca::detail::transform_reduce_sum<value_type, square_type, 1024>(mathcca::Cuda(), dX.begin().get(),  dX.end().get(), mathcca::detail::Square<value_type>(),  init);

#ifdef _THRUST

      sumdX_T= mathcca::detail::transform_reduce_sum(mathcca::Thrust(), dX.cbegin().get(), dX.cend().get(), mathcca::detail::Square<value_type>(), init);

#endif

      sumhX_0= mathcca::transform_reduce_sum(hX.begin(), hX.end(), mathcca::detail::Square<value_type>(), init);
      sumhX_O= mathcca::detail::transform_reduce_sum(mathcca::Omp(), hX.cbegin().get(), hX.cend().get(), mathcca::detail::Square<value_type>(), init);

#ifdef _STDPAR

      sumhX_S= mathcca::detail::transform_reduce_sum(mathcca::StdPar(), hX.cbegin().get(), hX.cend().get(), mathcca::detail::Square<value_type>(), init);

#endif
      
      auto res= static_cast<value_type>(3. * 3. * r * c);

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

      EXPECT_FLOAT_EQ(sumdX_0, res);
      EXPECT_FLOAT_EQ(sumhX_0, res);

      // https://en.wikipedia.org/wiki/Square_pyramidal_number
      if (i < 7) {
           
        mathcca::fill_iota(dX.begin(), dX.end(), static_cast<value_type>(1));
        mathcca::fill_iota(hX.begin(), hX.end(), static_cast<value_type>(1));
	  
	value_type n1{static_cast<value_type>(r * c)};
        value_type n2{static_cast<value_type>(r * r * c * c)};
        value_type n3{static_cast<value_type>(r * r * r * c * c * c)};
          
	res= n3/static_cast<value_type>(3) + n2/static_cast<value_type>(2) + n1/static_cast<value_type>(6);
         
        sumdX_0=  mathcca::transform_reduce_sum(dX.begin(),  dX.end(), mathcca::detail::Square<value_type>(),  init);
        sumdX_C1= mathcca::detail::transform_reduce_sum<value_type, square_type, 32  >(mathcca::Cuda(), dX.cbegin().get(), dX.cend().get(), mathcca::detail::Square<value_type>(), init);
        sumdX_C2= mathcca::detail::transform_reduce_sum<value_type, square_type, 64  >(mathcca::Cuda(), dX.begin().get(),  dX.end().get(), mathcca::detail::Square<value_type>(),  init);
        sumdX_C3= mathcca::detail::transform_reduce_sum<value_type, square_type, 128 >(mathcca::Cuda(), dX.cbegin().get(), dX.cend().get(), mathcca::detail::Square<value_type>(), init);
        sumdX_C4= mathcca::detail::transform_reduce_sum<value_type, square_type, 256 >(mathcca::Cuda(), dX.begin().get(),  dX.end().get(), mathcca::detail::Square<value_type>(),  init);
        sumdX_C5= mathcca::detail::transform_reduce_sum<value_type, square_type, 512 >(mathcca::Cuda(), dX.cbegin().get(), dX.cend().get(), mathcca::detail::Square<value_type>(), init);
        sumdX_C6= mathcca::detail::transform_reduce_sum<value_type, square_type, 1024>(mathcca::Cuda(), dX.begin().get(),  dX.end().get(), mathcca::detail::Square<value_type>(),  init);

#ifdef _THRUST

  	sumdX_T= mathcca::detail::transform_reduce_sum(mathcca::Thrust(), dX.cbegin().get(), dX.cend().get(), mathcca::detail::Square<value_type>(), init);

#endif

  	sumhX_0= mathcca::transform_reduce_sum(hX.begin(), hX.end(), mathcca::detail::Square<value_type>(), init);
        sumhX_O= mathcca::detail::transform_reduce_sum(mathcca::Omp(), hX.cbegin().get(), hX.cend().get(), mathcca::detail::Square<value_type>(), init);

#ifdef _STDPAR

  	sumhX_S= mathcca::detail::transform_reduce_sum(mathcca::StdPar(), hX.cbegin().get(), hX.cend().get(), mathcca::detail::Square<value_type>(), init);

#endif

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

        EXPECT_FLOAT_EQ(sumdX_0, res);
        EXPECT_FLOAT_EQ(sumhX_0, res);
	
      }
      else {

        mathcca::fill_iota(dX.begin(), dX.end(), static_cast<value_type>(1));
        mathcca::fill_iota(hX.begin(), hX.end(), static_cast<value_type>(1));
        
	// https://en.wikipedia.org/wiki/Basel_problem
        sumdX_0=  mathcca::transform_reduce_sum(dX.begin(),  dX.end(), mathcca::detail::InverseSquare<value_type>(),  init);
        sumdX_C1= mathcca::detail::transform_reduce_sum<value_type, invsquare_type, 32  >(mathcca::Cuda(), dX.cbegin().get(), dX.cend().get(), mathcca::detail::InverseSquare<value_type>(), init);
        sumdX_C2= mathcca::detail::transform_reduce_sum<value_type, invsquare_type, 64  >(mathcca::Cuda(), dX.begin().get(),  dX.end().get(), mathcca::detail::InverseSquare<value_type>(),  init);
        sumdX_C3= mathcca::detail::transform_reduce_sum<value_type, invsquare_type, 128 >(mathcca::Cuda(), dX.cbegin().get(), dX.cend().get(), mathcca::detail::InverseSquare<value_type>(), init);
        sumdX_C4= mathcca::detail::transform_reduce_sum<value_type, invsquare_type, 256 >(mathcca::Cuda(), dX.begin().get(),  dX.end().get(), mathcca::detail::InverseSquare<value_type>(),  init);
        sumdX_C5= mathcca::detail::transform_reduce_sum<value_type, invsquare_type, 512 >(mathcca::Cuda(), dX.cbegin().get(), dX.cend().get(), mathcca::detail::InverseSquare<value_type>(), init);
        sumdX_C6= mathcca::detail::transform_reduce_sum<value_type, invsquare_type, 1024>(mathcca::Cuda(), dX.begin().get(),  dX.end().get(), mathcca::detail::InverseSquare<value_type>(),  init);

#ifdef _THRUST

  	sumdX_T= mathcca::detail::transform_reduce_sum(mathcca::Thrust(), dX.cbegin().get(), dX.cend().get(), mathcca::detail::InverseSquare<value_type>(), init);

#endif

  	sumhX_0= mathcca::transform_reduce_sum(hX.begin(), hX.end(), mathcca::detail::InverseSquare<value_type>(), init);
   	sumhX_O= mathcca::detail::transform_reduce_sum(mathcca::Omp(), hX.cbegin().get(), hX.cend().get(), mathcca::detail::InverseSquare<value_type>(), init);

#ifdef _STDPAR

  	sumhX_S= mathcca::detail::transform_reduce_sum(mathcca::StdPar(), hX.cbegin().get(), hX.cend().get(), mathcca::detail::InverseSquare<value_type>(), init);

#endif

        res= static_cast<value_type>(std::numbers::pi * std::numbers::pi / 6);
      
	EXPECT_NEAR(sumdX_0,  sumhX_0,  0.001);
        EXPECT_NEAR(sumdX_0,  sumdX_C1, 0.001);
        EXPECT_NEAR(sumdX_C1, sumdX_C2, 0.001);
        EXPECT_NEAR(sumdX_C2, sumdX_C3, 0.001);
        EXPECT_NEAR(sumdX_C3, sumdX_C4, 0.001);
        EXPECT_NEAR(sumdX_C5, sumdX_C6, 0.001);

#ifdef _THRUST

  	EXPECT_NEAR(sumdX_C6, sumdX_T, 0.001);

#endif

  	EXPECT_NEAR(sumhX_0,  sumhX_O, 0.001);

#ifdef _STDPAR

  	EXPECT_NEAR(sumhX_O,  sumhX_S, 0.001);

#endif

        EXPECT_NEAR(sumdX_0, res, 0.001);
        EXPECT_NEAR(sumhX_0, res, 0.001);
      
      }
    
      std::swap(r,c);
      r *= 5;
      c *= 2;
    
    }

}


