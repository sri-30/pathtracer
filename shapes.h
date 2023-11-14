#include "vec3.h"
#include "ray.h"
#include "color.h"

class Shape;

class Material {
    public:
        __device__ Material() {}
        __device__ Material(color3 color_reflection, color3 color_emission) : color_reflection(color_reflection), color_emission(color_emission) {}

        color3 color_reflection;
        color3 color_emission;
        
        float reflectance;

};

class IntersectionPoint {
    public:
        __device__ IntersectionPoint() : intersects(false) {}
        __device__ IntersectionPoint(bool intersects, float distance, Material material, vec3 normal) 
        : intersects(intersects), distance(distance), material(material), normal(normal) {};
        bool intersects;
        float distance;
        Material material;
        vec3 position;
        vec3 normal;
};

class LightSource {
    public:
        __device__ LightSource(vec3 position, color3 emission_color = color3(1, 1, 1), float emission_intensity = 1)
            : position(position), emission_color(emission_color), emission_intensity(emission_intensity) {}
        vec3 position;
        color3 emission_color;
        float emission_intensity;

        __device__ float getIntensity(float t) {return this->emission_intensity / ((4*atan(1.0)) * 4 * pow(t, 2));}
};

class Shape {
    public:
        Material material;
        __device__ virtual bool intersects(ray& r) = 0;
        __device__ virtual IntersectionPoint getIntersection(ray& r) = 0;
        __device__ virtual vec3 getNormal(vec3 position) = 0;
        __device__ virtual Material getMaterial(vec3 position) = 0;
};

class Plane : public Shape {
    private:
        vec3 l;
        vec3 n;
        float xMin;
        float yMin;
        float zMin;
        
        float xMax;
        float yMax;
        float zMax;
    public:
        __device__ Plane(vec3 normal, vec3 pointOnPLane, Material material, float xMin = std::numeric_limits<float>::lowest(), float yMin = std::numeric_limits<float>::lowest(), float zMin = std::numeric_limits<float>::lowest(), 
        float xMax = std::numeric_limits<float>::max(), float yMax = std::numeric_limits<float>::max(), float zMax = std::numeric_limits<float>::max()) : l(pointOnPLane), n(normal),
         xMin(xMin), yMin(yMin), zMin(zMin), xMax(xMax), yMax(yMax), zMax(zMax){this->material = material;}
        __device__ vec3 getPoint() {return this->l;}
        __device__ vec3 getNormal(vec3 position) {return (this->n);}
        __device__ bool intersects (ray& r) {return (r.direction().dot(this->n) != 0);}
        __device__ Material getMaterial(vec3 position) {return this->material;}
        __device__ IntersectionPoint getIntersection(ray &r) {
            IntersectionPoint res = IntersectionPoint();
            if (r.direction().dot(this->n) != 0) {
                float d = ((r.origin() - this->l).dot(this->n))*(1/(r.direction().dot(this->n)));
                vec3 pos = r.at(d);
                if (this -> xMin < pos.x() && this -> yMin < pos.y() && this -> zMin < pos.z()
                    && this -> xMax > pos.x() && this -> yMax > pos.y() && this -> zMax > pos.z()) {
                    res.distance = d;
                    res.intersects = true;
                    res.normal = this->getNormal(pos);
                    res.position = pos;
                    res.material = this->getMaterial(pos);               
                }
            }
            return res;
        }
};


class Sphere : public Shape {
    private:
        vec3 c;
        float r;
    public:
        __device__ Sphere(vec3 center, float radius, Material material) : c(center), r(radius) {this->material = material;}
        __device__ vec3& center() {
            return this->c;
        }

        __device__ float radius() {
            return this->r;
        }

        __device__ Material getMaterial(vec3 position) {
            return this->material;
        }

        __device__ vec3 getNormal(vec3 position) {
            return (position - (this->c)).normalize();
        }

        __device__ bool intersects(ray& r) {

            vec3 toOrigin = r.origin() - this->c;
            float a = r.direction().dot(r.direction());
            float b = 2*(r.direction().dot(toOrigin));
            float c = (toOrigin).dot(toOrigin) - (this->r * this->r);
            float discriminant = b*b - 4*a*c;
            return discriminant >= 0;
        }

        __device__ IntersectionPoint getIntersection(ray& r) {
            vec3 toOrigin = r.origin() - this->c;
            float a = r.direction().dot(r.direction());
            float b = 2.0f*(r.direction().dot(toOrigin));
            float c = (toOrigin).dot(toOrigin) - (this->r * this->r);
            float discriminant = b*b - 4*a*c;
            IntersectionPoint res;
            res.intersects = false;
            if (discriminant >= 0) {
                float d = (-b - sqrt(discriminant) ) / (2.0*a);
                vec3 pos = r.at(d);
                res.intersects = true;
                res.material = this->getMaterial(pos);
                res.position = pos;
                res.distance = d;
                res.normal = this->getNormal(r.origin() + r.direction() * res.distance);
            }
            return res;
        }
};