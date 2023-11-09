#include <gtest/gtest.h>
#include <cmath>
#include "test_util.h"
#include "ray.h"
#include "vec3.h"

TEST(RayTest, AtTest) {
    vec3 o(0, 0, 0);
    vec3 d(1, 0, 0);
    ray r(o, d);
    float x = 5;
    vec3 y = r.at(5);
    EXPECT_FLOAT_EQ(y.x(), 5);
    EXPECT_FLOAT_EQ(y.y(), 0);
    EXPECT_FLOAT_EQ(y.z(), 0);
}
