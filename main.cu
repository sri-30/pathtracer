#include <stdio.h>
#include <iostream>

#include "vec3.h"
#include "color.h"
#include "shapes.h"
#include "util.h"

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

__global__ void render(vec3 *fb, int max_x, int max_y, Shape** scene, int n_objects) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;

    vec3 camera_pos(400, 400, -5);

    // For i in numSamples
    // Generate Ray for Pixel
    vec3 p_center(j+0.5, i+0.5, 0);
    vec3 direction = (p_center - camera_pos).normalize();
    ray r(p_center, direction);

    fb[pixel_index] = vec3(0, 0, 0);

    // Shape scene_objects[] = {Sphere(vec3(400, 400, 8), 12)};
    // // Trace Path for ray
    for (int i = 0; i < n_objects; i++) {
        Shape *s = scene[i];
        IntersectionPoint p = s->getIntersection(r);
        if (p.intersects) {     
            fb[pixel_index] = p.color_reflection;
        }
    }

    // Add color to pixel
    // Divide pixel color by numSamples
}

__global__ void constructScene(Shape **scene) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        scene[0] = new Sphere(vec3(400, 400, 8), 12);
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

    checkCudaErrors(cudaMalloc((void **)&scene, n_objs*sizeof(void**)));
    checkCudaErrors(cudaDeviceSynchronize());
    constructScene<<<1, 1>>>(scene);
    checkCudaErrors(cudaDeviceSynchronize());

    // allocate FB
    color3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    render<<<blocks, threads>>>(fb, nx, ny, scene, n_objs);
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

}