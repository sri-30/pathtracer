
/* Uniformly Distributed Random Directions in Hemisphere */
__device__ vec3 randomDirectionHemisphere(vec3 normal, curandState *s) {
    vec3 res(curand_normal(s), curand_normal(s), curand_normal(s));
    res.normalize();
    normal.normalize();
    res = res.dot(normal) < 0 ? -1 * res : res;
    return res.normalized();
}

/* Cosine-Weighted Distribution Sampling in Hemisphere */
__device__ vec3 sampleHemisphere(float u1, float u2) {
    float z = u1 * 2.0f - 1.0f;
    float a = u2 * 2*PI;
    float r = sqrt(1.0f - z * z);
    float x = r * cosf(a);
    float y = r * sinf(a);
    return vec3(x, y, z);
}

__device__ float FresnelSchlick(float n1, float n2, float cosTheta) {
    float r0 = pow((n1-n2)/(n1+n2), 2);
    return r0 + (1 - r0)*pow(1-cosTheta, 5);
}

__device__ float getFresnelRatio(vec3 N, vec3 I, float f0, float f90, float n1, float n2) {
        float cX = -N.dot(I);

        /* Check for TIR - Check if Sin2 critical angle > 1.0 */
        if (n1 > n2) {
                float n = n1/n2;
                float s2 = pow(n,2)*(1.0-pow(cX, 2));
                if (s2 > 1.0)
                    return f90;
                cX = sqrt(1.0-s2);
        }

        float fresnel_res = FresnelSchlick(n1, n2, cX);

        /* Adjust for material base reflectivity */
        return lerp(f0, f90, fresnel_res);
}

__device__ vec3 tracePath(Shape **scene, int n_objects, ray& r, curandState *s) {
    vec3 contribution(0.0f, 0.0f, 0.0f);
    int n_bounces = 30;
    vec3 coefficient = vec3(1.0f, 1.0f, 1.0f);

    for (int i = 0; i <= n_bounces; i++) {
        
        /* Get Nearest Intersection Point of Ray*/
        IntersectionPoint min_p = getNearestIntersection(r, scene, n_objects);
        
        /* Return if ray missed - add environmental light? */
        if (!min_p.intersects){
            break;
        }

        Material material = min_p.material;
        
        /* Attenuation of light - Beer's Law? Not sure how physically accurate this is */
        if (i > 0 && min_p.inside) {
            vec3 attenuation = -1 * min_p.material.refractionColor * min_p.distance;
            coefficient = coefficient.cwiseProduct(vec3(exp(attenuation[0]), exp(attenuation[1]), exp(attenuation[2])));
        }
        
        /* Reflectivity */
        float f0 = material.f0;
        
        /* Refractivity */
        float transparency = material.transparency;

        float p_specular = f0;
        float p_refract = transparency;

        if (f0 > 0.0f)
        {
            p_specular = getFresnelRatio(r.direction(), min_p.normal, f0, 1.0f, min_p.inside ? material.IOR : 1.0, !min_p.inside ? material.IOR : 1.0);
            p_refract = transparency * (1.0f - p_specular) / (1.0f - f0);
        }
        
        /* Determine whether we do specular reflection, diffuse reflection or refraction*/        
        int mode = 0;
        float u = curand_uniform(s);
        float p_ray;
		if (p_specular > 0.0f && u < p_specular) {
            mode = 1;
            p_ray = p_specular;
        } else if (p_refract > 0.0f && u < p_specular + p_refract) {
            mode = 2;
            p_ray = p_refract;
        } else {
            mode = 3;
            p_ray = 1.0f - (p_specular + p_refract);
        }
        
        /* Avoid numerical division by zero errors */
		p_ray = max(p_ray, 0.001f);

        /* Origin of new ray */
        vec3 newPosition = min_p.position;
        
        /* Diffuse Ray Direction - Cosine weighted random direction in hemisphere */
        vec3 diffuseRayDir = (min_p.normal + sampleHemisphere(curand_uniform(s), curand_uniform(s))).normalized();
        
        /* Specular Ray Direction - Perfect reflection */
        vec3 specularRayDir = reflect(r.direction(), min_p.normal).normalized();

        /* Account for roughness of specular material */
        specularRayDir = (lerp(specularRayDir, diffuseRayDir, material.specularRoughness*material.specularRoughness)).normalized();

        /* Refraction Ray Direction - according to Snell's Law */
        vec3 refractionRayDir = refract(r.direction(), min_p.normal, min_p.inside ? min_p.material.IOR : 1.0f / min_p.material.IOR).normalized();
        
        refractionRayDir = (lerp(refractionRayDir, (min_p.normal + sampleHemisphere(curand_uniform(s), curand_uniform(s))).normalized(), material.refractionRoughness*material.refractionRoughness).normalized());
        
        vec3 newDirection = (mode == 1) ? specularRayDir : (mode == 2) ? refractionRayDir : diffuseRayDir;
        
		// add in emissive lighting
        if (material.emissive.norm() > 0){
            contribution = contribution + material.emissive.cwiseProduct(coefficient);
        }
        
        /* Add colour to coefficient */
        coefficient = mode == 2 ? coefficient : coefficient.cwiseProduct(mode == 1 ? material.specularColor : material.albedo);
        
        coefficient = coefficient / p_ray;
        
        /* Russian Roulette */
        float q = max(coefficient.x(), max(coefficient.y(), coefficient.z()));
        if (curand_uniform(s) > q)
            break;

        coefficient = coefficient * 1.0f / q;    
        r = ray(newPosition, newDirection); 
    }
    return contribution;
}