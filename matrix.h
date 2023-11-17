#include <math.h>

class Matrix4 {
    private:
        float e[16];
    public:
        __device__ Matrix4() {this->e[0] = 1; this->e[5] = 1; this->e[10] = 1; this->e[15] = 1;}
        __device__ float operator[](int i) {return e[i];}
        __device__ float& get(int i, int j) {return e[i*4+j];}
        __device__ float getNum(int i, int j) {return e[i*4 + j];}
        __device__ void add(float e1[16]) {
            for (int i = 0; i < 16; i++) {
                e[i] += e1[i];
            }
        }
        __device__ void mul(float e1[16]) {
            float res[16] = {0};
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    for (int k = 0; k < 4; k++) {
                        res[i*4+j] += e1[i*4+j+k]*e[(i+k)*4+j];
                    }
                }
            }
            for (int i = 0; i < 16; i++) {
                e[i] = res[i];
            }
        }
        __device__ void translate()
        __device__ void rotate(float angle, vec3 axis) {
            float c = cos(angle);
            float s = sin(angle);
            float m[16] = {c + pow(axis[0], 2)*(1 - c), axis[0]*axis[1]*(1-c)-axis[2]*s, axis[1]*s + axis[0]*axis[2]*(1-c), 0,
                            axis[2]*s + axis[0]*axis[1]*(1-c), c+pow(axis[1], 2)*(1-c), -axis[0]*s + axis[1]*axis[2]*(1-c), 0,
                            -axis[1]*s + axis[0]*axis[2]*(1-c), axis[0]*s + axis[1]*axis[2]*(1-c), c+pow(axis[2], 2)*(1-c), 0,
                            0, 0, 0, 1}
            this->mul(m);
        }
        __device__ vec3 operator*(vec3 v) {return vec3(e[0]*e[0]+e[1]*e[2]+e[2]*e[2]+e[3]*e[3],
                                                        e[4]*e[0]+e[5]*e[2]+e[6]*e[2]+e[7]*e[3],
                                                        e[8]*e[0]+e[9]*e[2]+e[10]*e[2]+e[11]*e[3]);}
};