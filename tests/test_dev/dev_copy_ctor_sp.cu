#include <mathcca.hpp>
#include <gtest/gtest.h>

TEST(CopyCtorSp, BasicAssertions)
{
  std::size_t row{2};
  std::size_t col{5};
  for (auto n= 1; n < 9; ++n) {

    using value_type= float;
        
    mathcca::device_matrix<value_type> dA1{row, col};
    mathcca::host_matrix<value_type>   hA1{row, col};
    
    mathcca::device_matrix<value_type> dA2{row, col, static_cast<value_type>(n)};
    mathcca::host_matrix<value_type>   hA2{row, col, static_cast<value_type>(n)};
    
    mathcca::device_matrix<value_type> dD1{row, col, static_cast<value_type>(n+1)};
    mathcca::host_matrix<value_type>   hD1{row, col, static_cast<value_type>(n+1)};
    
    mathcca::device_matrix<value_type> dD2{1, 2};
    mathcca::host_matrix<value_type> hD2{1, 2};
    
    EXPECT_TRUE(dA1 != dA2);
    EXPECT_TRUE(dA1 != dD1);
    EXPECT_TRUE(dD1 != dD2);
    EXPECT_TRUE(dA2 != dD2);
    
    EXPECT_TRUE(hA1 != hA2);
    EXPECT_TRUE(hA1 != hD1);
    EXPECT_TRUE(hD1 != hD2);
    EXPECT_TRUE(hA2 != hD2);
    
    auto dB1{dA1};
    auto dB2{dA2};
    
    auto hB1{hA1};
    auto hB2{hA2};
    
    EXPECT_TRUE(dA1 == dB1);
    EXPECT_TRUE(dA2 == dB2);
    EXPECT_TRUE(dB1 != dB2);
    
    EXPECT_TRUE(hA1 == hB1);
    EXPECT_TRUE(hA2 == hB2);
    EXPECT_TRUE(hB1 != hB2);
    
    auto dC1{dB1};
    auto dC2{dB2};
    
    auto hC1{hB1};
    auto hC2{hB2};
    
    EXPECT_TRUE(dC1 == dA1);
    EXPECT_TRUE(dC2 == dA2);
    EXPECT_TRUE(dC1 != dC2);
    
    EXPECT_TRUE(hC1 == hA1);
    EXPECT_TRUE(hC2 == hA2);
    EXPECT_TRUE(hC1 != hC2);
    
    dD1= dC1;
    dD2= dC2;
    
    hD1= hC1;
    hD2= hC2;
    
    EXPECT_TRUE(dD1 == dA1);
    EXPECT_TRUE(dD2 == dA2);
    EXPECT_TRUE(dD1 != dD2);
    
    EXPECT_TRUE(hD1 == hA1);
    EXPECT_TRUE(hD2 == hA2);
    EXPECT_TRUE(hD1 != hD2);

    std::swap(row,col);
    row*= 5;
    col*= 2;

  }
}


