#include <gtest/gtest.h>
#include <cmath>
#include "test_util.h"
#include "ray.h"
#include "vec3.h"
#include "shapes.h"

TEST(ShapeTest, SphereIntersectTest) {
    vec3 c(5, 5, 5);
    float r = 2;

    vec3 origin(0, 0, 0);
    vec3 direction(1, 1, 1);
    direction = direction.normalize();

    ray ray1(origin, direction);
    Sphere sphere1(c, r);

    Sphere sphere2(c + vec3(0, 10, 0), r);

    EXPECT_EQ(sphere1.intersects(ray1), true);
    EXPECT_EQ(sphere2.intersects(ray1), false);
}