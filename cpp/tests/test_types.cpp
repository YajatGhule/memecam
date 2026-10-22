#include <catch2/catch_test_macros.hpp>

#include "types.h"

using namespace memecam;

TEST_CASE("hand size and bbox") {
    Hand h;
    for (int i = 0; i < 21; ++i) h.pts[i] = {100.f + i, 200.f + 2 * i, 0.f};
    REQUIRE(h.size() > 0);
    auto b = h.bbox();
    REQUIRE(b.x == 100);
    REQUIRE(b.width == 20);
}
