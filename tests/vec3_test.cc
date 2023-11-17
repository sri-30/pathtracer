#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cmath>
#include "test_util.h"
#include "vec3.h"

using testing::FloatEq;

class Vec3Test : public testing::Test {
  protected:
    void SetUp() override {
      v1 = vec3(x1, y1, z1);
      v2 = vec3(x2, y2, z2);
    }

    float x1 = 0.1;
    float y1 = 0.2;
    float z1 = 0.3;

    float x2 = 0.5;
    float y2 = 0.6;
    float z2 = 0.7;

    float t = 0.8;
  
    vec3 v1;
    vec3 v2;
};

// Demonstrate some basic assertions.
TEST_F(Vec3Test, ConstructionTest) {
  // Expect equality.
  EXPECT_THAT(v1.x(), FloatEq(x1));
  EXPECT_THAT(v1.y(), FloatEq(y1));
  EXPECT_THAT(v1.z(), FloatEq(z1));
}

TEST_F(Vec3Test, DotTest) {

  EXPECT_THAT(v1.dot(v2), FloatEq((x1 * x2) + (y1 * y2) + (z1 * z2)));
  EXPECT_THAT(vec3(1, 3, 8).dot(vec3(5, 3.3, 2)), FloatEq(30.9));
}

TEST_F(Vec3Test, PlusTest) {
  vec3 v3 = v1 + v2;
  EXPECT_THAT(v3.x(), FloatEq(x1+x2));
  EXPECT_THAT(v3.y(), FloatEq(y1+y2));
  EXPECT_THAT(v3.z(), FloatEq(z1+z2));
}

TEST_F(Vec3Test, MulTest) {
  vec3 v3 = v1 * t;
  EXPECT_THAT(v3.x(), FloatEq(v1.x()*t));
  EXPECT_THAT(v3.y(), FloatEq(v1.y()*t));
  EXPECT_THAT(v3.z(), FloatEq(v1.z()*t));
}

TEST_F(Vec3Test, MagTest) {
  EXPECT_THAT(v1.magnitude(), FloatEq(sqrt(v1.x()*v1.x() + v1.y()*v1.y() + v1.z()*v1.z())));
  EXPECT_THAT(v1.magnitude_squared(), FloatEq(v1.x()*v1.x() + v1.y()*v1.y() + v1.z()*v1.z()));
}

TEST_F(Vec3Test, NormalizeTest) {
  vec3 v3 = v1.normalize();
  EXPECT_THAT(v3.magnitude(), FloatEq(1));
}
