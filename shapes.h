#include "vec3.h"
#include "ray.h"
#include "color.h"

#define NULL_INTERSECTION IntersectionPoint(false, 0, false, false, vec3(0, 0, 0), vec3(0, 0, 0))

class Shape;

class IntersectionPoint {
    public:
        __device__ IntersectionPoint() {}
        __device__ IntersectionPoint(bool intersects, float distance, bool reflects, bool emits, color3 color_emission, color3 color_reflection) 
        : intersects(intersects), distance(distance), reflects(reflects), emits(emits), color_emission(color_emission), color_reflection(color_reflection) {};
        bool intersects;
        float distance;
        bool reflects;
        bool emits;
        color3 color_reflection;
        color3 color_emission;
};

class Shape {
    public:
        __device__ virtual bool intersects(ray& r) = 0;
        __device__ virtual IntersectionPoint getIntersection(ray& r) = 0;
};

class Sphere : public Shape {
    private:
        vec3 c;
        float r;
    public:
        __device__ Sphere(vec3 center, float radius) : c(center), r(radius) {}
        __device__ vec3& center() {
            return this->c;
        }

        __device__ float radius() {
            return this->r;
        }

        __device__ bool intersects(ray& r) {
            // u = ray direction
            // o = origin of ray
            // c = center
            // a = u dot u
            // b = 2(u dot (o - c))
            // c = (o - c) dot (o - c) - r^2
            // discriminant = b^2 - 4ac
            // if disc < 0 return false

            float a = r.direction().dot(r.direction());
            float b = 2*(r.direction().dot(r.origin() - this->c));
            float c = (r.origin() - this->c).dot(r.origin() - this->c) - (this->r * this->r);
            float discriminant = b*b - 4*a*c;
            return discriminant >= 0;
        }

        __device__ IntersectionPoint getIntersection(ray& r) {
            float a = r.direction().dot(r.direction());
            float b = 2*(r.direction().dot(r.origin() - this->c));
            float c = (r.origin() - this->c).dot(r.origin() - this->c) - (this->r * this->r);
            float discriminant = b*b - 4*a*c;
            IntersectionPoint res;
            res.intersects = false;
            res.reflects = false;
            res.emits = false;
            if (discriminant >= 0) {
                res.intersects = true;
                res.reflects = true;
                res.emits = false;
                res.color_reflection = vec3(0, 1, 0);
                res.distance = (-b - sqrt(discriminant) ) / (2.0*a);
            }
            return res;
        }
};