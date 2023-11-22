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
            point = (p.first.distance < p.second.distance) ? p.first : p.second;
        } else {
            point.intersects = false;
        }
        if (point.intersects && !min_p.intersects || point.distance < min_p.distance) {
            min_p = point;
            if (min_p.normal.dot(r.direction()) > 0){
                min_p.normal = -1 * min_p.normal;}
        }
    }
    return min_p;
}



__device__ vec3 randomDirectionHemisphere(vec3 normal, curandState *s) {
    vec3 res(curand_normal(s), curand_normal(s), curand_normal(s));
    res.normalize();
    normal.normalize();
    res = res.dot(normal) < 0 ? -1 * res : res;
    return res.normalized();
}

__device__ vec3 cosineWeightedRandomDirectionHemisphere(vec3 normal, curandState *s) {
    float rv1 = curand_normal(s);
    float rv2 = curand_normal(s);
    
	vec3  uu = normal.cross(vec3(0.0,1.0,1.0)).normalized();
	vec3  vv = uu.cross(normal).normalized();
	
	float ra = sqrt(rv2);
	float rx = ra*cos(6.2831*rv1); 
	float ry = ra*sin(6.2831*rv1);
	float rz = sqrt( 1.0-rv2 );
	vec3  rr = vec3( rx*uu + ry*vv + rz*normal );
    
    return rr.normalized();
}

// __device__ vec3 sampleLights(LightSource **lights, int n_lights, vec3 position) {
//     vec3 res(0, 0, 0);
//     for (int i = 0; i < n_lights; i++) {
//         res += lights[i]->emission_color*(lights[i]->getIntensity((lights[i]->position - position).norm()));
//     }
//     return res;
// } 

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
__device__ vec3 sampleHemisphere(float x, float y) {
    float a = sqrt(x);
    float b = 2*PI*y;

    return vec3(a*cos(b), a*sin(b), sqrt(1 - x));
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

__device__ vec3 evalFresnelSchlick(vec3 f0, float NdotS)
{
	return f0 + (vec3(1, 1, 1) - f0) * pow(1.0f - NdotS, 5.0f);
}

__device__ float luminance(vec3 rgb)
{
	return rgb.dot(vec3(0.2126f, 0.7152f, 0.0722f));
}

__device__ vec3 tracePath(Shape **scene, int n_objects, ray& r, curandState *s) {
    vec3 contribution(0, 0, 0);
    vec3 tp(1, 1, 1);
    int n_bounces = 100;
    vec3 coefficient = vec3(1, 1, 1);
    int refractive_index_current = 1;

    for (int i = 0; i <= n_bounces; i++) {
        /* Get Intersection */
        IntersectionPoint min_p = getNearestIntersection(r, scene, n_objects);

        if (!min_p.intersects)
            break;

        vec3 V = min_p.position.normalized();

        vec3 N = min_p.normal;

        // if (V.dot(N) <= 0.0f)
        //     break;

        Eigen::Quaternionf qRotationToZ = getRotationToZAxis(N);
	    vec3 Vlocal = qRotationToZ * V;
        const vec3 Nlocal = vec3(0.0f, 0.0f, 1.0f);

        Material material = min_p.material;
        float alpha = pow(material.roughness, 2);
        color3 emittance = material.color_emission * material.emissive;


        // Sample diffuse ray using cosine-weighted hemisphere sampling 
		vec3 rayDirectionLocal = sampleHemisphere(curand_uniform(s), curand_uniform(s));

        vec3 L = rayDirectionLocal;

		// Function 'diffuseTerm' is predivided by PDF of sampling the cosine weighted hemisphere
		vec3 sampleWeight = material.color_reflection * (Nlocal.dot(L));

        // Sample a half-vector of specular BRDF. Note that we're reusing random variable 'u' here, but correctly it should be an new independent random number
		vec3 Hspecular = sampleGGXWalter(Vlocal, alpha, curand_uniform(s), curand_uniform(s));
		// Check if specular sample is valid (does not reflect under the hemisphere)
		float VdotH = Vlocal.dot(Hspecular);
        // vec3 foo = evalFresnelSchlick(material.specular, min(1.0f, VdotH));
        // printf("A: %9.6f B: %9.6f C: %0.6f\n", foo[0], foo[1], foo[2]);
		// if (VdotH > 0.00001f)
		// {
            vec3 fresnel = evalFresnelSchlick(material.specular, min(1.0f, VdotH));
            //sampleWeight = (vec3(1, 1, 1) - fresnel).cwiseProduct(material.color_reflection * (Nlocal.dot(L))) + fresnel;
		// } else {
		// 	break;
		// }

        // if (luminance(sampleWeight) == 0.0f)
        //     break;

        vec3 newDirection = (qRotationToZ.inverse() * rayDirectionLocal).normalized();

        // if (newDirection.dot(N) <= 0.0f)
        //     break;


        // vec3 N = min_p.normal;

        // Material material = min_p.material;

        // /* Get current emittance */
        // color3 emittance = material.color_emission * material.emissive;
        
        // /* Generate New Ray */
        // vec3 newOrigin = min_p.position;
        // vec3 newDirection = sampleHemisphere(curand_uniform(s), curand_uniform(s));

        // float cos_theta = newDirection.dot(min_p.normal);

        // const float p = 1 / (2 * PI);

        // /* Compute Fresnel - (Schlick Approximation) and kd */
        // vec3 V = min_p.position - newDirection;
        // vec3 H = newDirection + (V - newDirection)/2;
        // vec3 L = newDirection;

        // vec3 fresnel = material.specular + (vec3(1, 1, 1) - material.specular) * pow(1 - (max(H.dot(V), 0.0f)), 5);

        // vec3 kd = vec3(1, 1, 1) - fresnel;
        
        // /* Compute Diffuse lighting - Lambertian Model */
        // color3 diffuse = material.color_reflection / PI;

        // /* Compute Specular lighting - Cook-Torrance */
        // float alpha = pow(material.roughness, 2);
        
        // /* GGX/Trowbridge-Reitz */
        // float D = pow(alpha, 2)/(PI * pow((pow(max(N.dot(H), 0.0f), 2)) * (pow(alpha, 2) - 1) + 1, 2));
        
        // /* Smith/Schlick-Beckmann */
        // float G = GeometryShadowing(L, N, alpha/2) * GeometryShadowing(V, N, alpha/2);

        // vec3 specular = (D*G*fresnel)/(4*(max(V.dot(N), 0.0f), max(L.dot(N), 0.0f)));

        /* Add current to contribution */
        contribution += coefficient.cwiseProduct(emittance);

        // /* Update coefficient */
        coefficient = coefficient.cwiseProduct(sampleWeight);

        r = ray(min_p.position, newDirection);
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
    // if ((i <= 5) && (j < 105 || j > 110))
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

    //uint32_t rngState = ((uint32_t)(i) * (uint32_t)(1973) + (uint32_t)(j) * (uint32_t) (9277) ) | uint(1);


    color3 ambient(0.3, 0.3, 0.3);
    int n_samples = 1000;
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
        vec3 p_center = pixel00_loc + (pixel_delta_u * (j + 0 * curand_uniform(&s))) + (pixel_delta_v * (i + 0 * curand_uniform(&s)));
        vec3 direction = (p_center - camera_pos).normalized();
        ray r = ray(camera_pos, direction);
        totalLight += tracePath(scene, n_objects, r, &s);
    }
    p_color = totalLight/n_samples;
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
        t4.translation() = Eigen::Translation3f(-0.5, -0.5, -3.0).translation();

        Eigen::Affine3f t5 = IDENTITY;
        t5.translation() = Eigen::Translation3f(-0.5, 0.5, -3.0).translation();

        Material light_material;
        light_material.color_reflection = color3(0, 0, 0);
        light_material.color_emission = color3(1, 1, 1);
        light_material.emissive = 20;

        Material glass;
        glass.refractive = true;
        glass.refractive_index = 1.5f;
        
        //scene[0] = new Cylinder(MATT_RED, t2);

        Material blue_wall = MATT_BLUE;
        Material white_wall = MATT_WHITE;
        Material metallic_ball = MATT_RED;
        metallic_ball.metalness = 1;
        Material dielectric_ball = MATT_GREEN;
        dielectric_ball.metalness = 0;

        scene[0] = new Plane(white_wall, tplane1, -5, -5, 15, 15);
        scene[1] = new Plane(white_wall, tplane2, -5, -5, 15, 15);
        scene[2] = new Plane(blue_wall, tplane3, -5, -5, 15, 15);
        scene[3] = new Plane(blue_wall, tplane4, -5, -5, 15, 15);
        scene[4] = new Plane(white_wall, tplane5, -5, -5, 15, 15);
        
        scene[5] = new Sphere(light_material, t1);
        scene[6] = new Sphere(metallic_ball, t4);
        scene[7] = new Sphere(dielectric_ball, t3);
        //scene[8] = new Sphere(glass, t5);
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
    int n_objs = 8;
    Shape **scene;

    int n_lights = 1;
    LightSource **lights;

    Eigen::Affine3f **transforms;
    checkCudaErrors(cudaMalloc((void **)&scene, n_objs*sizeof(void**)));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMalloc((void **)&lights, n_lights*sizeof(void**)));
    checkCudaErrors(cudaDeviceSynchronize());
    // checkCudaErrors(cudaMalloc((void **)&transforms, n_objs*sizeof(void**)));
    // checkCudaErrors(cudaDeviceSynchronize());
    constructScene<<<1, 1>>>(scene);
    checkCudaErrors(cudaDeviceSynchronize());
    //constructTransforms<<<1, 1>>>(transforms);
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

    curandState *devStates;
    //checkCudaErrors(cudaMalloc((void **)&devStates, blocks * threads *sizeof(curandState)));

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