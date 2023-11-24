#pragma once

#include <Eigen/Core>

#define PI 3.14159265358979323846
#define DEG_TO_RAD(X) (X*PI)/180
#define SMALL_NUMBER 0.00001
#define IDENTITY Eigen::Affine3f::Identity()

template <typename T> __device__ T lerp(T v0, T v1, float t) {
  return (1 - t) * v0 + t * v1;
}

using vec3 = Eigen::Vector3<float>;
using vec4 = Eigen::Vector4<float>;
using matrix4 = Eigen::Matrix4<float>;
using Q4 = Eigen::Quaternion<float>;

__device__ vec3 reflect(vec3 original, vec3 normal) {
    return original - 2 * (original.dot(normal) * normal);
};

__device__ vec3 refract(vec3 I, vec3 N, float eta)
{
    float k = 1.0 - eta * eta * (1.0 - (N.dot(I)) * (N.dot(I)));
    if (k < 0.0){
        return vec3(0, 0, 0); }
    else{
        return eta * I - (eta * (N.dot(I)) + sqrt(k)) * N;}
}
