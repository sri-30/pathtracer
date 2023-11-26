#include <stdio.h>
#include <iostream>
#include <math.h>
#include <curand_kernel.h>

#include "shapes.h"
#include "vec_math.h"
#include "bsdf.h"


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
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

__global__ void render(vec3 *fb, config_t config, Shape** scene, int n_objects) {
    int max_x = config.max_x;
    int max_y = config.max_y;
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x || j >= max_y))
        return;
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

    int n_samples = 4000;
    vec3 p_color(0, 0, 0);
    vec3 totalLight(0, 0, 0);
    int n_origin;
    curandState s;

    curand_init(pixel_index, 0, 0, &s);
    for (int sample = 0; sample < n_samples; sample++) {
        vec3 p_center = pixel00_loc + (pixel_delta_u * (i - 0.5f + curand_uniform(&s))) + (pixel_delta_v * (max_y - j - 0.5f + curand_uniform(&s)));
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
        Eigen::Affine3f floor = IDENTITY;
        floor.linear() = Eigen::AngleAxisf(PI/2, vec3(1, 0, 0)).toRotationMatrix();
        floor.translation() = Eigen::Translation3f(0.0, -1.0, 0.0).translation();

        /* Back Wall */
        Eigen::Affine3f back = IDENTITY;
        back.translation() = Eigen::Translation3f(0.0, 0, -5.0).translation();

        /* Left Wall */
        Eigen::Affine3f left = IDENTITY;
        left.linear() = Eigen::AngleAxisf(PI/2, vec3(0, 1, 0)).toRotationMatrix();
        left.translation() = Eigen::Translation3f(-1.0, 0.0, -5.0).translation();
        
        /* Right Wall */
        Eigen::Affine3f right = IDENTITY;
        right.linear() = Eigen::AngleAxisf(-PI/2, vec3(0, 1, 0)).toRotationMatrix();
        right.translation() = Eigen::Translation3f(1.0, 0.0, -5.0).translation();
                
        /* Ceiling */
        Eigen::Affine3f ceiling = IDENTITY;
        ceiling.linear() = Eigen::AngleAxisf(-PI/2, vec3(1, 0, 0)).toRotationMatrix();
        ceiling.translation() = Eigen::Translation3f(0.0, 1.0, 0.0).translation();

        /* Light */
        Eigen::Affine3f light = IDENTITY;
        light.linear() = Eigen::AngleAxisf(-PI/2, vec3(1, 0, 0)).toRotationMatrix();
        light.translation() = Eigen::Translation3f(0.0, 0.97, -3.25).translation();

        /* Front Wall */
        Eigen::Affine3f tplane6 = IDENTITY;
        tplane6.linear() = Eigen::AngleAxisf(PI, vec3(0, 1, 0)).toRotationMatrix();
        tplane6.translation() = Eigen::Translation3f(0.0, 0.0, 7.0).translation();

        Eigen::Affine3f t3 = IDENTITY;
        t3.translation() = Eigen::Translation3f(0.5, -0.5, -4.0).translation();

        Eigen::Affine3f t4 = IDENTITY;
        t4.translation() = Eigen::Translation3f(-0.5, -0.5, -3.5).translation();

        Eigen::Affine3f t5 = IDENTITY;
        t5.translation() = Eigen::Translation3f(0.0, -0.5, -3.0).translation();
        t5.linear() = Eigen::AngleAxisf(PI/4, vec3(0, 1, 0)).toRotationMatrix();
        t5.scale(0.5f);

        Eigen::Affine3f t6 = IDENTITY;
        t6.translation() = Eigen::Translation3f(0.6, -0.5, -3.0).translation();
        t6.linear() = Eigen::AngleAxisf(PI/4, vec3(0, 1, 0)).toRotationMatrix();
        t6.scale(0.4f);

        Eigen::Affine3f t7 = IDENTITY;
        t7.translation() = Eigen::Translation3f(-0.6, -0.5, -3.0).translation();
        t7.linear() = Eigen::AngleAxisf(PI/4, vec3(0, 1, 0)).toRotationMatrix();
        t7.scale(0.4f);


        Material light_material;
        light_material.emissive = vec3(1.0f, 0.9f, 0.7f) * 20.0f;   

        Material base;
        base.albedo = vec3(0.6f, 0.6f, 0.6f);

        Material green;
        green.albedo = vec3(0.0f, 1.0f, 0.0f);

        Material red;
        red.albedo = vec3(1.0f, 0.0f, 0.0f);

        Material glass;  
        glass.f0 = 0.02f;
        glass.specularRoughness = 0;
        glass.specularColor = vec3(1.0f, 1.0f, 1.0f) * 0.8f;
        glass.IOR = 1.5f;
        glass.transparency = 1.0f;
        glass.refractionRoughness = 0.0f;

        Material metal;
        metal.albedo = vec3(1.0f, 1.0f, 1.0f);    
        metal.f0 = 1.0f;
        metal.specularRoughness = 0.05f;
        metal.specularColor = vec3(0.7f, 0.1f, 0.8f);   

        Material dielectric;
        dielectric.albedo = vec3(0.9f, 0.3f, 0.7f);
        dielectric.emissive = vec3(0.0f, 0.0f, 0.0f);        
        dielectric.f0 = 0.1f;
        dielectric.specularRoughness = 0.2f;
        dielectric.specularColor = vec3(0.9f, 0.9f, 0.9f);

        scene[0] = new Plane(base, floor, -15.0f, -15.0f, 15.0f, 15.0f);
        scene[1] = new Plane(red, left, -15.0f, -15.0f, 15.0f, 15.0f);
        scene[2] = new Plane(green, right, -15.0f, -15.0f, 15.0f, 15.0f);
        scene[3] = new Plane(base, back, -15.0f, -15.0f, 15.0f, 15.0f);
        scene[4] = new Plane(base, ceiling, -15.0f, -15.0f, 15.0f, 15.0f);
        scene[5] = new Plane(light_material, light, -0.5f, -0.5f, 0.5f, 0.5f);
        scene[6] = new Plane(base, tplane6, -15.0f, -15.0f, 15.0f, 15.0f);
        scene[7] = new Sphere(dielectric, t5);
        scene[8] = new Cube(dielectric, t6);
        scene[9] = new Cylinder(dielectric, t7);

        //scene[8] = new Sphere(metal, t4);
        //scene[9] = new Sphere(glass, t5);
        //scene[8] = new Cube(glass, t5);
    }
}

int main() {

    int nx = 800;
    int ny = 800;

    int tx = 8;
    int ty = 8;

    int num_pixels = nx*ny;

    size_t fb_size = num_pixels*sizeof(vec3);

    /* Allocate space for Scene*/
    int n_objs = 10;
    Shape **scene;

    checkCudaErrors(cudaMalloc((void **)&scene, n_objs*sizeof(void**)));
    checkCudaErrors(cudaDeviceSynchronize());

    constructScene<<<1, 1>>>(scene);
    checkCudaErrors(cudaDeviceSynchronize());

    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    float v_height = 5.0;
    float v_width = nx/ny * v_height;
    float fov = 45;
    float focal_length = (v_width/2) / tan(DEG_TO_RAD(fov/2));

    config_t config = {vec3(0, 0, 0), nx, ny, focal_length, v_height};

    render<<<blocks, threads>>>(fb, config, scene, n_objs);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    /* Output as PPM Image */
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
}