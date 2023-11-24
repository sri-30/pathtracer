#pragma once
#include "vec_math.h"

class ray {
    private:
        vec3 o;
        vec3 d;
    public:
    __device__ ray() {}
    __device__ ray(vec3 origin, vec3 direction) : o(origin), d(direction) {}
    __device__ vec3 origin() {return this->o;}
    __device__ vec3 direction() {return this->d;}
    __device__ vec3 at(float t) {return this->o + (this->d)*t;}
};