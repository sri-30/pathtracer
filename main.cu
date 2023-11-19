#include <stdio.h>
#include <iostream>

//#include "vec3.h"
#include "color.h"
#include "shapes.h"
#include "util.h"
#include <math.h>

#define MATT_RED Material(color3(1, 0, 0), color3(1, 0, 0))
#define MATT_GREEN Material(color3(0, 1, 0), color3(0, 1, 0))
#define MATT_BLUE Material(color3(0, 0, 1), color3(0, 0, 1))

#define EPSILON 0.0001

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#define EIGEN_NO_CUDA

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

__device__ bool checkIntersections(ray& r, int n_origin, Shape** scene, int n_objects) {
    for (int i = 0; i < n_objects; i++) {
        IntersectionPoint p = scene[i]->getIntersection(r);
        if (i != n_origin && p.intersects) {
            return true;
        }
    }
    return false;
}

__global__ void render(vec3 *fb, config_t config, Shape** scene, int n_objects, LightSource** lights, int n_lights) {
    int max_x = config.max_x;
    int max_y = config.max_y;
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x || j >= max_y))
        return;
    // if((i >= 200) || (j <= 200))
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

    vec3 p_center = pixel00_loc + (pixel_delta_u * j) + (pixel_delta_v * i);
    vec3 direction = (p_center - camera_pos);
    int n_bounces = 1;
    int n_samples = 1;
    color3 p_color(0, 0, 0);
    int n_origin;
    // printf("A: %9.6f B: %9.6f C: %0.6f\n", direction[0], direction[1], direction[2]);
    // printf("A1: %9.6f B1: %9.6f C1: %0.6f\n", pixel_delta_v[0], pixel_delta_v[1], pixel_delta_v[2]);
    for (int i = 0; i < n_samples; i++) {
        ray r = ray(camera_pos, direction);
        for (int j = 0; j < n_bounces; j++) {
            IntersectionPoint min_p;
            min_p.intersects = false;
            for (int k = 0; k < n_objects; k++) {
                Shape *obj = scene[k];
                IntersectionPoint p = obj->getIntersection(r);
                if (p.intersects && p.position.z() < camera_pos.z()) {
                    if (!min_p.intersects) {
                        min_p = p;
                    } else {
                        if (p.position.z() > min_p.position.z()) {
                            min_p = p;
                            n_origin = k;
                        }
                    }
                }
            }

            if (!min_p.intersects) {
                break;
            }
            vec3 hitpoint = r.at(min_p.distance);
            for (int k = 0; k < n_lights; k++) {
                LightSource *l = lights[k];

                vec3 P = min_p.position;
                vec3 N = min_p.normal;
                vec3 O = r.origin();
                vec3 L = ((l->position) - P).normalized();
                vec3 V = (config.camera_pos - P).normalized();

                float I = l->getIntensity((l->position - P).norm());

                bool shadow = false;
                float distanceToLight = (l->position - P).norm();
                ray shadowRay = ray(P + N * EPSILON, L);

                for (int x = 0; x < n_objects; x++) {
                    IntersectionPoint p_ = scene[x]->getIntersection(shadowRay);
                    float intersection_distance = (l->position - p_.position).norm();
                    if (p_.intersects && intersection_distance < distanceToLight) {
                        shadow = true;
                        break;
                    }
                }

                if (shadow)
                    continue;

                // printf("Intensity: %9.6f, Diffuse term: %9.6f\n", I, max(0.0, N.normalized().dot(L)));

                color3 diffuse = (min_p.material.color_reflection) * (I) * (max(0.0, N.normalized().dot(L)));

                vec3 R = reflect(L, N).normalized();
                color3 specular = l->emission_color * I * pow(max(0.0, R.dot(V)), 50);
                //printf("Magnitude: %9.6f\n", diffuse.norm());
                // if (specular.norm() > 0.1)
                p_color = p_color + diffuse * 0.8 + specular * 0.8;
            }
            p_color = p_color +  min_p.material.color_reflection * 0.4;
        }
        // if (p_color.norm() > 0) {
        //     printf("Colour2: %9.6f\n%9.6f\n%9.6f\n\n", p_color[0], p_color[1], p_color[2]);
        // }
    }
    fb[pixel_index] = p_color;
}

__global__ void constructScene(Shape **scene) {
    if (threadIdx.x == 0 && blockIdx.x == 0){
        Eigen::Affine3f t1 = IDENTITY;
        //t1.linear() = Eigen::AngleAxisf(10, vec3(0, 0, 1)).toRotationMatrix();
        t1.translation() = Eigen::Translation3f(0, 0, -6.0).translation();
        
        Eigen::Affine3f t2 = IDENTITY;
        t2.linear() = Eigen::AngleAxisf(3*PI/4, vec3(1, 0, 0)).toRotationMatrix();
        t2.translation() = Eigen::Translation3f(-1, -1, -7.0).translation();

        Eigen::Affine3f t3 = IDENTITY;
        t3.linear() = Eigen::AngleAxisf(PI/4, vec3(1, 0, 0)).toRotationMatrix() * Eigen::AngleAxisf(PI/4, vec3(0, 1, 0)).toRotationMatrix();
        t3.translation() = Eigen:: Translation3f(0, 1.5, -6.0).translation();
        
        scene[0] = new Cylinder(MATT_RED, t2);
        // scene[1] = new Sphere(MATT_BLUE, t1);
        // scene[2] = new Plane(MATT_GREEN, t2, -1, -1, 5, 5);
        // scene[3] = new Cube(MATT_RED, t3);
    }
}

// __global__ void constructTransforms(Eigen::Affine3f **transforms) {
//         Eigen::Affine3f a = IDENTITY;
//         a.linear() = Eigen::AngleAxisf(10, vec3(0, 0, 1)).toRotationMatrix() * Eigen::Matrix3f::Identity();
//         a.translation() = Eigen::Translation3f(0, 0, -5.0).translation();
// }


__global__ void constructLights(LightSource **lights) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        lights[0] = new LightSource(vec3(0, 3, -4.0), color3(1, 1, 1), 400);
}


int main() {

    int nx = 800;
    int ny = 800;

    int tx = 8;
    int ty = 8;

    int num_pixels = nx*ny;

    size_t fb_size = num_pixels*sizeof(color3);

    // allocate space for scene
    int n_objs = 1;
    Shape **scene;

    int n_lights = 1;
    LightSource **lights;

    Eigen::Affine3f **transforms;

    checkCudaErrors(cudaMalloc((void **)&scene, n_objs*sizeof(void**)));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMalloc((void **)&lights, n_lights*sizeof(void**)));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMalloc((void **)&transforms, n_objs*sizeof(void**)));
    checkCudaErrors(cudaDeviceSynchronize());
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