#include "vec3.h"

class camera {
    private:
        vec3 position;
    public:
        __host__ __device__ vec3 getPosition() {
            return this->position;
        }
};
