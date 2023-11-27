# How to run?
CD into build  
Run `cmake ..`  
Run `cmake . --build`
Run `Debug/Pathtracer > out.ppm`

# Dependencies
Eigen - clone and build eigen in root folder with CMake  
CUDA - must be run on CUDA compatible device

# Code Structure
## `main.cu`
Contains the kernel and code run on the host (calling the kernel):
1. Construct scene and allocate memory for it
2. Calculate screen dimensions
3. Launch CUDA kernel

### Kernel
Each thread does the computation for a single pixel on the screen. A `CUDASYNCHRONIZE` call waits for this computation to finish before writing the pixel values to a file.

## `bsdf.h`
Contains the `tracepath` function which is called for each pixel (for the number of samples specified). This contains the computation of the intersections between the ray and various objects in the scene as well as the computation of the BRDF and the sample value.

## `materials.h`
Contains the class `material`, which holds all of the details of a particular material. TODO: implement helper functions for creating materials to prevent creation of physically impossible materials.

## `ray.h`
Contains a class for representing a ray in parametric form.

## `shapes.h`
Contains all of the classes for shape representations, which all inherit from a base `shape` class, with one virtual function: `getRayPath`, which returns the intersection points a ray has with an object. Since all of the shapes are solid, an intersecting ray will have an entry point and an exit point, which are named `first` and `second`.

## `shapes_unit.h`
Messy hack for unit testing compatibility. Must be fixed later

## `vec_math.h`
Contains various mathematical functions and constants required. `reflect` and `refract` implementations mostly mirror their implementations in `GLSL`.

## `bsdf_alternative.h`
Work-in-progress implementation for better customizability and representation of BRDFs.

# Tests
Tests are done via GoogleTest. Currently only unit-tests are run on the classes in `shapes.h` for testing intersections. Other tests are more qualitative and involve rendering the scenes in the `scenes` folder and comparing to the expected outcome.