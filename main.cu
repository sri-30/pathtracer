#include <stdio.h>
#include <iostream>

//#include "vec3.h"
#include "color.h"
#include "shapes.h"
#include "util.h"
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define MATT_RED Material(color3(0.4, 0, 0), color3(0.4, 0, 0))
#define MATT_GREEN Material(color3(0, 0.4, 0), color3(0, 0.4, 0))
#define MATT_BLUE Material(color3(0, 0, 0.4), color3(0, 0, 0.4))
#define MATT_WHITE Material(color3(0.5, 0.5, 0.5), color3(0.5, 0.5, 0.5))

#define EPSILON 0.0001


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ float lerp(float v0, float v1, float t) {
  return (1 - t) * v0 + t * v1;
}

__device__ vec3 lerp_vec(vec3 v0, vec3 v1, float t) {
  return (1 - t) * v0 + t * v1;
}


typedef struct config_s {
    vec3 camera_pos;
    int max_x;
    int max_y;
    float focal_length;
    float view_port_height;
} config_t;

__device__ IntersectionPoint getNearestIntersection(ray& r, Shape** scene, int n_objects) {
    IntersectionPoint min_p;
    min_p.intersects = false;
    for (int k = 0; k < n_objects; k++) {
        Shape *obj = scene[k];
        RayPath p = obj->getIntersections(r);
        IntersectionPoint point;
        if (p.n_intersections == 1) {
            point = p.first;
        } else if (p.n_intersections == 2) {
            point = (p.second.distance < p.first.distance) ? p.second : p.first;
        } else {
            point.intersects = false;
        }
        if (point.intersects && (!min_p.intersects || point.distance < min_p.distance)) {
            min_p = point;
            min_p.inside = min_p.normal.dot(r.direction()) > 0;
            if (min_p.inside) {min_p.normal = -1 * min_p.normal;}
        }
    }
    return min_p;
}

/* Uniformly Distributed Random Directions in Hemisphere */
__device__ vec3 randomDirectionHemisphere(vec3 normal, curandState *s) {
    vec3 res(curand_normal(s), curand_normal(s), curand_normal(s));
    res.normalize();
    normal.normalize();
    res = res.dot(normal) < 0 ? -1 * res : res;
    return res.normalized();
}

__device__ Eigen::Quaternionf getRotationToZAxis(vec3 input) {

	// Handle special case when input is exact or near opposite of (0, 0, 1)
	if (input.z() < -0.99999f) return Eigen::Quaternionf(1.0f, 0.0f, 0.0f, 0.0f);

	return Eigen::Quaternionf(1.0f + input.z(), input.y(), -input.x(), 0.0f).normalized();
}


__device__ Eigen::Quaternionf getRotationFromZAxis(vec3 input) {

	// Handle special case when input is exact or near opposite of (0, 0, 1)
	if (input.z() < -0.99999f) return Eigen::Quaternionf(1.0f, 0.0f, 0.0f, 0.0f);

	return Eigen::Quaternionf(1.0f + input.z(), input.y(), input.x(), 0.0f).normalized();
}

__device__ Eigen::Quaternionf invertRotation(Eigen::Quaternionf q) {
    return Eigen::Quaternionf(q.w(), -q.x(), -q.y(), -q.z());
}

/* Cosine-Weighted Distribution Sampling */
__device__ vec3 sampleHemisphere(float u1, float u2) {
    float z = u1 * 2.0f - 1.0f;
    float a = u2 * 2*PI;
    float r = sqrt(1.0f - z * z);
    float x = r * cos(a);
    float y = r * sin(a);
    return vec3(x, y, z);
}

__device__ vec3 sampleGGXWalter(vec3 Vlocal, float alpha, float x, float y) {
	float alphaSquared = pow(alpha, 2);

	// Calculate cosTheta and sinTheta needed for conversion to H vector
	float cosThetaSquared = (1.0f - x) / ((alphaSquared - 1.0f) * x + 1.0f);
	float cosTheta = sqrt(cosThetaSquared);
	float sinTheta = sqrt(1.0f - cosThetaSquared);
	float phi = 2 * PI * y;

	// Convert sampled spherical coordinates to H vector
	return vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta).normalized();
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

// __device__ vec3 getGGXMicrofacet(float u1, float u2, float roughness, vec3 N)
// {
// 	// Get an orthonormal basis from the normal
// 	vec3 B = N.cross3(vec3(1.0, 0, 0));
// 	vec3 T = B.cross(N);

// 	// GGX NDF sampling
// 	float a2 = roughness * roughness;
// 	float cosThetaH = sqrt(max(0.0f, (1.0-u1)/((a2-1.0)*u1+1) ));
// 	float sinThetaH = sqrt(max(0.0f, 1.0f - cosThetaH * cosThetaH));
// 	float phiH = u2 *PI * 2.0f;

// 	// Get our GGX NDF sample (i.e., the half vector)
// 	return T * (sinThetaH * cos(phiH)) +
//            B * (sinThetaH * sin(phiH)) +
//            N * cosThetaH;
// }


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
            //H = getGGXMicrofacet(curand_uniform(s), curand_uniform(s), material.specularRoughness, min_p.normal);
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
        specularRayDir = (lerp_vec(specularRayDir, diffuseRayDir, material.specularRoughness*material.specularRoughness)).normalized();

        /* Refraction Ray Direction - according to Snell's Law */
        vec3 refractionRayDir = refract(r.direction(), min_p.normal, min_p.inside ? min_p.material.IOR : 1.0f / min_p.material.IOR).normalized();
        
        refractionRayDir = (lerp_vec(refractionRayDir, (min_p.normal + sampleHemisphere(curand_uniform(s), curand_uniform(s))).normalized(), material.refractionRoughness*material.refractionRoughness).normalized());
        
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

__global__ void render(vec3 *fb, config_t config, Shape** scene, int n_objects, LightSource** lights, int n_lights) {
    int max_x = config.max_x;
    int max_y = config.max_y;
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x || j >= max_y))
        return;
    // if ((i < 200 || i > 205) || (j < 200 || j > 205))
    //     return;
    int pixel_index = j*max_x + i;

    vec3 camera_pos = config.camera_pos;
    float focal_length = config.focal_length;
    float viewport_height = config.view_port_height;
    float viewport_width = viewport_height * (static_cast<float>(max_x)/max_y);
    
    vec3 viewport_u = vec3(viewport_width, 0, 0);
    vec3 viewport_v = vec3(0, -viewport_height, 0);

    vec3 pixel_delta_u = viewport_u / (float) max_x;
    vec3 pixel_delta_v = viewport_v / (float) max_y;

    vec3 viewport_upper_left = camera_pos - vec3(0, 0, focal_length) - viewport_u/2.0 - viewport_v/2.0;
    vec3 pixel00_loc = viewport_upper_left + (pixel_delta_u + pixel_delta_v) * 0.5;


    color3 ambient(0.3, 0.3, 0.3);
    int n_samples = 100;
    color3 p_color(0, 0, 0);
    color3 rayColor(1, 1, 1);
    color3 incomingLight(0, 0, 0);
    color3 totalLight(0, 0, 0);
    int n_origin;
    curandState s;
    // printf("A: %9.6f B: %9.6f C: %0.6f\n", direction[0], direction[1], direction[2]);
    // printf("A1: %9.6f B1: %9.6f C1: %0.6f\n", pixel_delta_v[0], pixel_delta_v[1], pixel_delta_v[2]);
    curand_init(pixel_index, 0, 0, &s);
    for (int sample = 0; sample < n_samples; sample++) {
        vec3 p_center = pixel00_loc + (pixel_delta_u * (j - 0.5f + curand_uniform(&s))) + (pixel_delta_v * (i - 0.5f + curand_uniform(&s)));
        vec3 direction = (p_center - camera_pos).normalized();
        ray r = ray(camera_pos, direction);
        totalLight += tracePath(scene, n_objects, r, &s);
    }
    p_color = totalLight/((float)n_samples);
    fb[pixel_index] = p_color;
}

__global__ void constructScene(Shape **scene) {
    if (threadIdx.x == 0 && blockIdx.x == 0){
        Eigen::Affine3f t1 = IDENTITY;
        t1.translation() = Eigen::Translation3f(0.5, 0, -3.0).translation();
        
        /* Floor */
        Eigen::Affine3f tplane1 = IDENTITY;
        tplane1.linear() = Eigen::AngleAxisf(PI/2, vec3(0, 1, 0)).toRotationMatrix();
        tplane1.translation() = Eigen::Translation3f(-1.0, 0, 0.0).translation();

        /* Back Wall */
        Eigen::Affine3f tplane2 = IDENTITY;
        //t2.linear() = Eigen::AngleAxisf(PI/2, vec3(0, 1, 0)).toRotationMatrix();
        tplane2.translation() = Eigen::Translation3f(0.0, 0, -5.0).translation();

        /* Left Wall */
        Eigen::Affine3f tplane3 = IDENTITY;
        tplane3.linear() = Eigen::AngleAxisf(PI/2, vec3(1, 0, 0)).toRotationMatrix();
        tplane3.translation() = Eigen::Translation3f(0.0, -1.0, -5.0).translation();
        
        /* Right Wall */
        Eigen::Affine3f tplane4 = IDENTITY;
        tplane4.linear() = Eigen::AngleAxisf(-PI/2, vec3(1, 0, 0)).toRotationMatrix();
        tplane4.translation() = Eigen::Translation3f(0.0, 1.0, -5.0).translation();
                
        /* Ceiling */
        Eigen::Affine3f tplane5 = IDENTITY;
        tplane5.linear() = Eigen::AngleAxisf(-PI/2, vec3(0, 1, 0)).toRotationMatrix();
        tplane5.translation() = Eigen::Translation3f(1.5, 0.0, 0.0).translation();

        Eigen::Affine3f t3 = IDENTITY;
        t3.translation() = Eigen::Translation3f(-0.5, 0.5, -4.0).translation();

        Eigen::Affine3f t4 = IDENTITY;
        //t4.linear() = Eigen::AngleAxisf(PI/4, vec3(1, 0, 0)).toRotationMatrix();
        t4.translation() = Eigen::Translation3f(-0.5, -0.5, -3.0).translation();

        Eigen::Affine3f t5 = IDENTITY;
        t5.translation() = Eigen::Translation3f(0.0, 0.0, -2.0).translation();
        t5.linear() = Eigen::AngleAxisf(PI/4, vec3(1, 0, 0)).toRotationMatrix();

        Material light_material;
        light_material.emissive = vec3(1.0f, 0.9f, 0.7f) * 10.0f;   
        
        //scene[0] = new Cylinder(MATT_RED, t2);

        Material base;
        base.albedo = vec3(0.9f, 0.25f, 0.25f);
        base.emissive = vec3(0.0f, 0.0f, 0.0f);        
        base.f0 = 0.0f;
        base.specularRoughness = 0.0f;
        base.specularColor = vec3(1.0f, 1.0f, 1.0f) * 0.8f;
        base.IOR = 1.1f;
        base.transparency = 0.0f;

        Material blue_wall = base;
        blue_wall.IOR = 1.0f;
        Material white_wall = base;
        white_wall.IOR = 1.0f;
        Material metallic_ball = base;
        Material dielectric_ball = base;
        dielectric_ball.IOR = 1.0f;
        Material white_ball = base;
        Material floor = base;

        Material glass;
        glass.albedo = vec3(0.9f, 0.25f, 0.25f);
        glass.emissive = vec3(0.0f, 0.0f, 0.0f);        
        glass.f0 = 0.02f;
        glass.specularRoughness = 0;
        glass.specularColor = vec3(1.0f, 1.0f, 1.0f) * 0.8f;
        glass.IOR = 1.1f;
        glass.transparency = 1.0f;
        glass.refractionRoughness = 0;
        glass.refractionColor = vec3(0.0f, 0.5f, 1.0f);

        floor.albedo = vec3(0.4, 0.01, 0.3);

        metallic_ball.f0 = 0.3;

        blue_wall.albedo = vec3(0, 0, 0.4f);
        white_wall.albedo = vec3(0.2f, 0.2f, 0.2f);
        metallic_ball.albedo = vec3(0.4f, 0, 0);
        dielectric_ball.albedo = vec3(0, 0.4f, 0);
        white_ball.albedo = vec3(0.9f, 0.25f, 0.25f);

        scene[0] = new Plane(white_wall, tplane1, -5, -5, 15, 15);
        scene[1] = new Plane(white_wall, tplane2, -5, -5, 15, 15);
        scene[2] = new Plane(white_wall, tplane3, -5, -5, 15, 15);
        scene[3] = new Plane(white_wall, tplane4, -5, -5, 15, 15);
        scene[4] = new Plane(light_material, tplane5, -5, -5, 15, 15);
        
        scene[5] = new Sphere(dielectric_ball, t3);
        scene[6] = new Sphere(metallic_ball, t4);
        //scene[7] = new Sphere(dielectric_ball, t3);
        //scene[8] = new Cube(glass, t5);
    }
}

// __global__ void constructTransforms(Eigen::Affine3f **transforms) {
//         Eigen::Affine3f a = IDENTITY;
//         a.linear() = Eigen::AngleAxisf(10, vec3(0, 0, 1)).toRotationMatrix() * Eigen::Matrix3f::Identity();
//         a.translation() = Eigen::Translation3f(0, 0, -5.0).translation();
// }


__global__ void constructLights(LightSource **lights) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        lights[0] = new LightSource(vec3(0.5, 0.5, 0), color3(1, 1, 1), 120);
}


int main() {

    int nx = 400;
    int ny = 400;

    int tx = 8;
    int ty = 8;

    int num_pixels = nx*ny;

    size_t fb_size = num_pixels*sizeof(color3);

    // allocate space for scene
    int n_objs = 7;
    Shape **scene;

    int n_lights = 1;
    LightSource **lights;

    checkCudaErrors(cudaMalloc((void **)&scene, n_objs*sizeof(void**)));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMalloc((void **)&lights, n_lights*sizeof(void**)));
    checkCudaErrors(cudaDeviceSynchronize());
    constructScene<<<1, 1>>>(scene);
    checkCudaErrors(cudaDeviceSynchronize());
    constructLights<<<1, 1>>>(lights);
    checkCudaErrors(cudaDeviceSynchronize());

    // allocate FB
    color3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    float v_height = 5.0;
    float v_width = nx/ny * v_height;
    float fov = 45;
    float focal_length = (v_width/2) / tan(DEG_TO_RAD(fov/2));

    config_t config = {vec3(0, 0, 0), nx, ny, focal_length, v_height};

    render<<<blocks, threads>>>(fb, config, scene, n_objs, lights, n_lights);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // Output FB as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        std::clog << "\rScanlines remaining: " << (ny - j) << ' ' << std::flush;
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            float r = fb[pixel_index].x();
            float g = fb[pixel_index].y();
            float b = fb[pixel_index].z();
            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(scene));
    checkCudaErrors(cudaFree(lights));
}