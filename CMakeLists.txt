cmake_minimum_required (VERSION 3.8)
project (pathtracer LANGUAGES CXX CUDA)

find_package(Eigen3 REQUIRED)



add_executable (pathtracer
                main.cu)

add_compile_options(--expt-relaxed-constexpr)

target_link_libraries (pathtracer Eigen3::Eigen)
