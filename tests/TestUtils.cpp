#include <catch2/catch.hpp>


#include "simdhelpers/utils.hpp"
#include <iostream>

TEST_CASE("Test - ilog2") {
    REQUIRE(ilog2(1) == 0);
    REQUIRE(ilog2(8) == 3);
    REQUIRE(ilog2(6) == 2);
}
