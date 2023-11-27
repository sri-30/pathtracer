#include <gtest/gtest.h>
#include "test_util.h"
#include "../ray.h"
#include "../shapes_unit.h"

TEST(ShapeTest, SphereIntersectTest) {
    Eigen::Affine3f t1 = IDENTITY;
    Eigen::Affine3f t2 = IDENTITY;
    Eigen::Affine3f t3 = IDENTITY;
    t2.translation() = vec3(0, 10, 0);
    t3.translation() = vec3(5, 5, 5);
    Material m;

    vec3 origin(0, 0, 0);
    vec3 direction(1, 1, 1);
    direction.normalize();

    ray ray1(origin, direction);
    Sphere sphere1(m, t1);
    Sphere sphere2(m, t2);
    Sphere sphere3(m, t3);

    RayPath p1 = sphere1.getIntersections(ray1);
    RayPath p2 = sphere2.getIntersections(ray1);
    RayPath p3 = sphere3.getIntersections(ray1);

    EXPECT_EQ(p1.n_intersections, 1);
    EXPECT_EQ(p2.n_intersections, 0);
    EXPECT_EQ(p3.n_intersections, 2);
}

TEST(ShapeTest, PlaneIntersectTest) {
    Eigen::Affine3f t1 = IDENTITY;
    Material m;

    vec3 origin(0, 0, -5);
    vec3 direction(1, 1, 1);
    direction.normalize();

    ray ray1(origin, direction);
    Plane plane1(m, t1);
    Plane plane2(m, t1, 0, 0, 2, 2);

    RayPath p1 = plane1.getIntersections(ray1);
    RayPath p2 = plane2.getIntersections(ray1);

    EXPECT_EQ(p1.n_intersections, 1);
    EXPECT_EQ(p2.n_intersections, 0);
}

TEST(ShapeTest, CubeIntersectTest) {
    Eigen::Affine3f t1 = IDENTITY;
    Eigen::Affine3f t2 = IDENTITY;
    Eigen::Affine3f t3 = IDENTITY;
    t2.translation() = vec3(0, 10, 0);
    t3.translation() = vec3(5, 5, 5);
    Material m;

    vec3 origin(0, 0, 0);
    vec3 direction(1, 1, 1);
    direction.normalize();

    ray ray1(origin, direction);
    Cube Cube1(m, t1);
    Cube Cube2(m, t2);
    Cube Cube3(m, t3);

    RayPath p1 = Cube1.getIntersections(ray1);
    RayPath p2 = Cube2.getIntersections(ray1);
    RayPath p3 = Cube3.getIntersections(ray1);

    EXPECT_EQ(p1.n_intersections, 1);
    EXPECT_EQ(p2.n_intersections, 0);
    EXPECT_EQ(p3.n_intersections, 2);
}

TEST(ShapeTest, CylinderIntersectTest) {
    Eigen::Affine3f t1 = IDENTITY;
    Eigen::Affine3f t2 = IDENTITY;
    Eigen::Affine3f t3 = IDENTITY;
    t2.translation() = vec3(0, 10, 0);
    t3.translation() = vec3(0, 0, -5);
    Material m;

    ray ray1(vec3(0, 0, 0), vec3(0, 0, -1));
    ray ray2(vec3(0, 0, 0), vec3(0, 1, 0));


    Cylinder Cylinder1(m, t1);
    Cylinder Cylinder2(m, t2);
    Cylinder Cylinder3(m, t3);

    RayPath p1 = Cylinder1.getIntersections(ray1);
    RayPath p2 = Cylinder2.getIntersections(ray1);
    RayPath p3 = Cylinder3.getIntersections(ray1);

    EXPECT_EQ(p1.n_intersections, 1);
    EXPECT_EQ(p2.n_intersections, 0);
    EXPECT_EQ(p3.n_intersections, 2);

    p1 = Cylinder1.getIntersections(ray2);
    p2 = Cylinder2.getIntersections(ray2);
    p3 = Cylinder3.getIntersections(ray2);

    EXPECT_EQ(p1.n_intersections, 1);
    EXPECT_EQ(p2.n_intersections, 2);
    EXPECT_EQ(p3.n_intersections, 0);
}