#pragma once

#include "vec_math.h"

class Shape;

class Material {
    public:
        __device__ Material() {}

        /* Diffuse colour of surface */
        vec3 albedo = vec3(0.0f, 0.0f, 0.0f);
        
        /* Emissitivity of the surface */
        vec3 emissive = vec3(0.0f, 0.0f, 0.0f);
        
        /* Base Reflectivity of Material when viewed at normal */
        float f0 = 0.0f;

        /* Roughness of Specular Reflections */
        float specularRoughness = 0.0f;

        /* Colour of Specular Reflections */
        vec3 specularColor = vec3(0.0f, 0.0f, 0.0f);

        /* Index of Refraction */
        float IOR = 1.0f;

        /* Probability of refraction */
        float transparency = 0.0f;

        /* Roughness of refraction */
        float refractionRoughness = 0.0f;

        /* Attenuation of light - colour absorbed by material */
        vec3 refractionColor = vec3(0.0f, 0.0f, 0.0f);

        int bsdfType = 0;
};