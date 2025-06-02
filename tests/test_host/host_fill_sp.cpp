#include <mathcca.hpp>
#include <gtest/gtest.h>

TEST(FillSp, BasicAssertions)
{
    std::size_t r{2};
    std::size_t c{5};
    for (auto n= 1; n < 8; ++n) {

      using value_type= float;

      mathcca::host_matrix<value_type> A{r, c};
      mathcca::host_matrix<value_type> X0{r, c};
      mathcca::host_matrix<value_type> X1{r, c};
      mathcca::host_matrix<value_type> X2{r, c};
      mathcca::host_matrix<value_type> X3{r, c};
      mathcca::host_matrix<value_type> X4{r, c};
      mathcca::host_matrix<value_type> X5{r, c};
      mathcca::host_matrix<value_type> X6{r, c};
      mathcca::host_matrix<value_type> X7{r, c};
      mathcca::host_matrix<value_type> X8{r, c};
      mathcca::host_matrix<value_type> X9{r, c};
     


      mathcca::fill_const(A.begin(), A.end(), static_cast<value_type>(n));
      
      for (std::size_t i= 0; i < A.size(); ++i) {
        EXPECT_FLOAT_EQ(A[i], static_cast<value_type>(n));
      }

      mathcca::detail::fill_const(mathcca::Omp(), A.begin().get(), A.end().get(), static_cast<value_type>(n+1));
      
      for (std::size_t i= 0; i < A.size(); ++i) {
        EXPECT_FLOAT_EQ(A[i], static_cast<value_type>(n+1));
      }

#ifdef _STDPAR
      
      mathcca::detail::fill_const(mathcca::StdPar(), A.begin().get(), A.end().get(), static_cast<value_type>(n+2));
      
      for (std::size_t i= 0; i < A.size(); ++i) {
        EXPECT_FLOAT_EQ(A[i], static_cast<value_type>(n+2));
      }

#endif
      
      mathcca::fill_iota(A.begin(), A.end(), static_cast<value_type>(10));

      for (std::size_t i= 0; i < A.size(); ++i) {
        EXPECT_FLOAT_EQ(A[i], static_cast<value_type>(10 + i));
      }
      
      mathcca::detail::fill_iota(mathcca::Omp(), A.begin().get(), A.end().get(), static_cast<value_type>(11));

      for (std::size_t i= 0; i < A.size(); ++i) {
        EXPECT_FLOAT_EQ(A[i], static_cast<value_type>(11 + i));
      }

#ifdef _STDPAR
      
      mathcca::detail::fill_iota(mathcca::StdPar(), A.begin().get(), A.end().get(), static_cast<value_type>(12));
      
      for (std::size_t i= 0; i < A.size(); ++i) {
        EXPECT_FLOAT_EQ(A[i], static_cast<value_type>(12+i));
      }
        
#endif
      
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

      std::size_t same= 0;
      for (std::size_t i= 0; i < X0.size() - 1; ++i) {
        if (X0[i] == X0[i+1]) {
          ++same;
        }
      }
      std::cout << "SAME= " << same << "\n";
      EXPECT_TRUE(same < 10);

      mathcca::detail::fill_rand(mathcca::Omp(), X0.begin().get(), X0.end().get()); 
      mathcca::detail::fill_rand(mathcca::Omp(), X1.begin().get(), X1.end().get());
      mathcca::detail::fill_rand(mathcca::Omp(), X2.begin().get(), X2.end().get());
      mathcca::detail::fill_rand(mathcca::Omp(), X3.begin().get(), X3.end().get());
      mathcca::detail::fill_rand(mathcca::Omp(), X4.begin().get(), X4.end().get());
      mathcca::detail::fill_rand(mathcca::Omp(), X5.begin().get(), X5.end().get());
      mathcca::detail::fill_rand(mathcca::Omp(), X6.begin().get(), X6.end().get());
      mathcca::detail::fill_rand(mathcca::Omp(), X7.begin().get(), X7.end().get()); 
      mathcca::detail::fill_rand(mathcca::Omp(), X8.begin().get(), X8.end().get());
      mathcca::detail::fill_rand(mathcca::Omp(), X9.begin().get(), X9.end().get());

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

      same= 0;
      for (std::size_t i= 0; i < X0.size() - 1; ++i) {
        if (X0[i] == X0[i+1]) {
          ++same;
        }
      }
      std::cout << "SAME= " << same << "\n";
      EXPECT_TRUE(same < 10);

#ifdef _STDPAR

      mathcca::detail::fill_rand(mathcca::StdPar(), X0.begin().get(), X0.end().get());
      mathcca::detail::fill_rand(mathcca::StdPar(), X1.begin().get(), X1.end().get());
      mathcca::detail::fill_rand(mathcca::StdPar(), X2.begin().get(), X2.end().get());
      mathcca::detail::fill_rand(mathcca::StdPar(), X3.begin().get(), X3.end().get());
      mathcca::detail::fill_rand(mathcca::StdPar(), X4.begin().get(), X4.end().get());
      mathcca::detail::fill_rand(mathcca::StdPar(), X5.begin().get(), X5.end().get());
      mathcca::detail::fill_rand(mathcca::StdPar(), X6.begin().get(), X6.end().get());
      mathcca::detail::fill_rand(mathcca::StdPar(), X7.begin().get(), X7.end().get());
      mathcca::detail::fill_rand(mathcca::StdPar(), X8.begin().get(), X8.end().get());
      mathcca::detail::fill_rand(mathcca::StdPar(), X9.begin().get(), X9.end().get());

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

      same= 0;
      for (std::size_t i= 0; i < X0.size() - 1; ++i) {
        if (X0[i] == X0[i+1]) {
          ++same;
        }
      }
      std::cout << "SAME= " << same << "\n";
      EXPECT_TRUE(same < 10);

#endif

      std::swap(r,c);
      r*= 5;
      c*= 2;
    }
}


