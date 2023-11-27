/* Alternative Class based BSDF Implementation - TODO */

struct ONB {
    vec3 x;
    vec3 y;
    vec3 z;
};

__device__ ONB constructONB(vec3 N) {
    float sign = (N.z() < 0 ? -1.0f : 1.0f );
    float a = -1 / (sign + N.z());
    float b = N.x() * N.y() * a;
    ONB res;
    res.x = vec3(1 + sign * pow(N.x(), 2) * a, sign * b, -sign * N.x());
    res.y = vec3(b, sign + pow(N.y(), 2) * a, -N.y());
    res.z = N;
    return res;
}

__device__ vec3 toLocalONB(ONB onb, vec3 v) {
    return vec3(v.dot(onb.x), v.dot(onb.y), v.dot(onb.z));
}

__device__ vec3 inverseONB(ONB onb, vec3 v) {
    return v.x() * onb.x + v.y() * onb.y + v.z() * onb.z;
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


/* Uniformly Distributed Random Directions in Hemisphere */
__device__ vec3 randomDirectionHemisphere(vec3 normal, curandState *s) {
    vec3 res(curand_normal(s), curand_normal(s), curand_normal(s));
    res.normalize();
    normal.normalize();
    res = res.dot(normal) < 0 ? -1 * res : res;
    return res.normalized();
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

class BSDF {};

class Lambertian : public BSDF {

    public:

        /* Evaluate BSDF */
        __device__ static vec3 evalBSDF(ray& r, vec3 sampleDirection, IntersectionPoint p, Material m) {
            vec3 N = p.normal;
            vec3 V = -1.0f * r.direction();
            vec3 L = sampleDirection;

            float NDotL = N.dot(L);
            float NDotV = N.dot(V);

            if (NDotL <= 0.0f || NDotV <= 0.0f) return vec3(0.0f, 0.0f, 0.0f);

            vec3 out = (1.0f / PI) * m.albedo;
            return out * max(min(NDotL, 1.0f), 0.0f);
        }

        /* Sample BSDF */
        __device__ static vec3 sampleBSDF(ray& r, IntersectionPoint p, Material m, curandState*s) {
            vec3 N = p.normal;
            
            float u1 = curand_uniform(s);
            float u2 = curand_uniform(s);

            ONB onb = constructONB(N);
            vec3 res = sampleHemisphere(u1, u2);
            
            return inverseONB(onb, res);
        }

        /* Evaluate BSDF PDF */
        __device__ static float EvalPDF(ray& r, vec3 sampleDirection, IntersectionPoint p, Material m) {
            vec3 N = p.normal;
            vec3 L = sampleDirection;

            return abs(L.dot(N)) * (1.0f/PI);
        }

};

class SmoothDielectric : public BSDF {
    public:
        /* Evaluate BSDF */
        __device__ static vec3 evalBSDF(ray& r, vec3 sampleDirection, IntersectionPoint p, Material m, bool transmission, float p_transmission) {
            return m.specularColor;
        }

        /* Sample BSDF */
        __device__ static vec3 sampleBSDF(ray& r, IntersectionPoint p, Material m, curandState*s, bool *transmission, float *p_transmission) {
            vec3 N = p.normal;
            float p_reflect = getFresnelRatio(N, r.direction(), m.f0, 1.0f, p.inside ? m.IOR : 1.0f, !p.inside ? 1.0f : m.IOR);
            float p_refract = 1 - p_reflect;
            p_reflect *= (m.f0 > 0);
            p_refract *= (m.transparency > 0);
            *p_transmission = p_refract;
            if (p_reflect == 0.0f && p_refract == 0.0f)
                return vec3(0.0f, 0.0f, 0.0f);
            float u = curand_uniform(s);
            if (u < p_reflect/(p_reflect + p_refract)) {
                *transmission = false;
                return reflect(r.direction(), N).normalized();
            } else {
                *transmission = true;
                return refract(r.direction(), N, (p.inside ? p.material.IOR : 1.0f / p.material.IOR)).normalized();
            }
        }

        /* Evaluate BSDF PDF */
        __device__ static float EvalPDF(ray& r, vec3 sampleDirection, IntersectionPoint p, Material m, bool transmission, float p_transmission) {
            float n1 = p.inside ? 1.0f : m.IOR;
            float n2 = !p.inside ? 1.0f : m.IOR;
            return (transmission ? p_transmission : 1 - p_transmission);
        }
};



__device__ vec3 tracePath(Shape **scene, int n_objects, ray& r, curandState *s) {
    vec3 contribution(0.0f, 0.0f, 0.0f);
    int n_bounces = 5;
    vec3 coefficient = vec3(1.0f, 1.0f, 1.0f);

    for (int i = 0; i <= n_bounces; i++) {
        
        /* Get Nearest Intersection Point of Ray*/
        IntersectionPoint min_p = getNearestIntersection(r, scene, n_objects);
        
        /* Return if ray missed */
        if (!min_p.intersects){
            break;
        }

        Material material = min_p.material;

        if (material.emissive.norm() > 0){
            contribution = contribution + material.emissive.cwiseProduct(coefficient);
        }

        vec3 newPosition = min_p.position;
        vec3 newDirection;
        vec3 f;
        float pdf;

        switch (material.bsdfType)
        {
        case 0:
            newDirection = Lambertian::sampleBSDF(r, min_p, material, s).normalized();
            f = Lambertian::evalBSDF(r, newDirection, min_p, material);
            pdf = Lambertian::EvalPDF(r, newDirection, min_p, material);
            break;
        
        case 1:
            bool transmission;
            float p_transmission;
            newDirection = SmoothDielectric::sampleBSDF(r, min_p, material, s, &transmission, &p_transmission).normalized();
            f = SmoothDielectric::evalBSDF(r, newDirection, min_p, material, transmission, p_transmission);
            pdf = SmoothDielectric::EvalPDF(r, newDirection, min_p, material, transmission, p_transmission);
            break;
        
        default:
            vec3 newDirection = Lambertian::sampleBSDF(r, min_p, material, s).normalized();
            vec3 f = Lambertian::evalBSDF(r, newDirection, min_p, material);
            float pdf = Lambertian::EvalPDF(r, newDirection, min_p, material);
            break;
        }

        if (pdf > 0.0f)
		    coefficient = coefficient.cwiseProduct(f / pdf); 
        else
            break;

        float p = max(coefficient.x(), max(coefficient.y(), coefficient.z()));

        float u = curand_uniform(s);

        if (p < u)
            break;
        
        coefficient = coefficient * (1 / p);

        r = ray(newPosition, newDirection); 
    }
    return contribution;
}