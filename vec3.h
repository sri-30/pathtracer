#pragma once
#include <cmath>

class vec3 {
    private:
        float e[3];
    public:
        __host__ __device__ vec3() : e{0, 0, 0} {}
        __host__ __device__ vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2;}
        __host__ __device__ inline float x() const {return e[0];}
        __host__ __device__ inline float y() const {return e[1];}
        __host__ __device__ inline float z() const {return e[2];}
        __host__ __device__ inline float dot(const vec3& v) {return (e[0] * v.x() + e[1] * v.y() + e[2] * v.z());}
        __host__ __device__ vec3 operator+(const vec3& v) {return vec3(e[0] + v.x(), e[1] + v.y(), e[2] + v.z());}
        __host__ __device__ vec3& operator=(const vec3& v) {this -> e[0] = v.x(); this -> e[1] = v.y(); this -> e[2] = v.z(); return *this;}
        __host__ __device__ vec3 operator-() {return vec3(-e[0], -e[1], -e[2]);}
        __host__ __device__ vec3 operator-(const vec3& v) {return *this + -vec3(v.x(), v.y(), v.z());}
        __host__ __device__ float operator[](int i) {return e[i];}
        __host__ __device__ inline vec3 operator*(float t) { return vec3(t*e[0], t*e[1], t*e[2]);}
        __host__ __device__ inline vec3 operator/(float t) {return (*this) * (1/t);}
        __host__ __device__ vec3& operator*=(float t) {e[0]*=t; e[1]*=t; e[2]*=t; return *this;}
        __host__ __device__ float magnitude() {return sqrt(magnitude_squared());}
        __host__ __device__ float magnitude_squared() {return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];}
        __host__ __device__ inline vec3 normalize() {return (*this) * (1.0f/(this->magnitude()));}
        __host__ __device__ vec3 reflect(vec3 normal) {return (normal * (2 * this->dot(normal)) - (*this));
    }
};
