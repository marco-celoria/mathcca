#include <mathcca.hpp>
#include <gtest/gtest.h>

TEST(MoveCtorDp, BasicAssertions)
{
  std::size_t row{2};
  std::size_t col{5};
  for (auto n= 1; n < 9; ++n) {
    mathcca::device_matrix<double> a1{row,  col};
    mathcca::device_matrix<double> a2{row,  col, static_cast<double>(n)};
    mathcca::device_matrix<double> d1{row, col};
    mathcca::device_matrix<double> d2{1, 2};
    
    EXPECT_TRUE(a1 != a2);
    EXPECT_TRUE(d1 != d2);
    
    auto b1{a1};
    auto b2{a2};

    EXPECT_TRUE(a1 == b1);
    EXPECT_TRUE(a2 == b2);
    EXPECT_TRUE(b1 != b2);
    
    auto c1{std::move(b1)};
    auto c2{std::move(b2)};
    
    EXPECT_TRUE(c1 != b1);
    EXPECT_TRUE(c2 != b2);

    EXPECT_TRUE(c1 == a1);
    EXPECT_TRUE(c2 == a2);
    EXPECT_TRUE(c1 != c2);
    
    d1= std::move(c1);
    d2= std::move(c2);
    
    EXPECT_TRUE(a1 == d1);
    EXPECT_TRUE(a2 == d2);
    EXPECT_TRUE(d1 != c1);
    EXPECT_TRUE(d2 != c2);
    EXPECT_TRUE(d1 != d2);

    std::swap(row,col);
    row*= 5;
    col*= 2;
  }
}


