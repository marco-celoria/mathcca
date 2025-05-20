#include <hello.hpp>

#include <gtest/gtest.h>


TEST(GreetingTest1, BasicAssertions)
{
    // Expect equality.
    std::string value = hello::greeting();
    EXPECT_EQ(value,  std::string("Hello World!!"));
}

