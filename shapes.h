#include "vec3.h"
#include "ray.h"
#include "color.h"
#include <Eigen/Core>
#include <Eigen/LU>

typedef Eigen::Matrix<float, 4, 4> Matrix4;

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
    protected:
        Matrix4 transform;
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

class Cube : public Shape {
    private:
        Material material;
        float xMin;
        float xMax;
        float yMin;
        float yMax;
        float zMin;
        float zMax;
    public:
        __device__ Cube(vec3 cornerLeft, vec3 cornerRight, Material material) : material(material) {
            this->xMin = min(cornerLeft.x(), cornerRight.x());
            this->yMin = min(cornerLeft.y(), cornerRight.y());
            this->zMin = min(cornerLeft.z(), cornerRight.z());
            this->xMax = max(cornerLeft.x(), cornerRight.x());
            this->yMax = max(cornerLeft.y(), cornerRight.y());
            this->zMax = max(cornerLeft.z(), cornerRight.z());
        }

        __device__ vec3 getNormal(vec3 position) {
            vec3 vMax = vec3(this->xMax, this->yMax, this->zMax);
            vec3 vMin = vec3(this->xMin, this->yMin, this->zMin);
            vec3 center = (vMax + vMin) * (0.5f);
            vec3 res = position - center;
            return res.normalized();
        }

        __device__ Material getMaterial(vec3 position) {
            return this->material;
        }

        __device__ bool intersects(ray& r) {
            float tx1 = (xMin - r.origin()[0]) / r.direction()[0];
            float tx2 = (xMax - r.origin()[0]) / r.direction()[0];
            float ty1 = (yMin - r.origin()[1]) / r.direction()[1];
            float ty2 = (yMax - r.origin()[1]) / r.direction()[1];
            float tz1 = (zMin - r.origin()[2]) / r.direction()[2];
            float tz2 = (zMax - r.origin()[2]) / r.direction()[2];

            float tNear = max(min(tx1, tx2), max(min(ty1, ty2), min(tz1, tz2)));
            float tFar = min(max(tx1, tx2), min(max(ty1, ty2), max(tz1, tz2)));
            
            return !(tNear > tFar || tFar < 0);
        }

        __device__ IntersectionPoint getIntersection(ray& r) {
            float tx1 = (xMin - r.origin()[0]) / r.direction()[0];
            float tx2 = (xMax - r.origin()[0]) / r.direction()[0];
            float ty1 = (yMin - r.origin()[1]) / r.direction()[1];
            float ty2 = (yMax - r.origin()[1]) / r.direction()[1];
            float tz1 = (zMin - r.origin()[2]) / r.direction()[2];
            float tz2 = (zMax - r.origin()[2]) / r.direction()[2];

            float tNear = max(min(tx1, tx2), max(min(ty1, ty2), min(tz1, tz2)));
            float tFar = min(max(tx1, tx2), min(max(ty1, ty2), max(tz1, tz2)));

            // printf("Near: %9.6f, Far: %9.6f\n", tNear, tFar);

            IntersectionPoint res;
            res.intersects = false;

            if (!(tNear > tFar || tFar < 0)) {
                res.intersects = true;
                res.distance = tNear;
                res.position = r.at(tNear);
                res.material = this->getMaterial(res.position);
                res.normal = this->getNormal(res.position);
            }
            return res;
        }
};

class Sphere : public Shape {
    private:
        vec3 c;
        float r;
    public:
        __device__ Sphere(Material material, Matrix4 transform = Matrix4::Identity(4, 4)) {this->transform = transform; this->material = material;}
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
            return (position - (this->c)).normalized();
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
            vec3 localOrigin = vec423(this->transform.inverse() * vec324(r.origin()));
            vec3 localDirection = vec423(this->transform.inverse() * vec324(r.direction()));
            // printf("Direction dot product: %9.6f\n", localDirection.dot(r.direction())/(localDirection.norm() * r.direction().norm()));
            // printf("LocalOrigin: %9.6f %9.6f %9.6f\n", localOrigin[0], localOrigin[1], localOrigin[2]);
            //printf("LocalDirection: %9.6f %9.6f %9.6f\n", localDirection[0], localDirection[1], localDirection[2]);
            ray localRay = ray(localOrigin, localDirection);
            vec3 toOrigin = localRay.origin() - vec3(0, 0, -4);
            float a = localRay.direction().dot(localRay.direction());
            float b = 2.0f*(localRay.direction().dot(toOrigin));
            float c = (toOrigin).dot(toOrigin) - 1;
            float discriminant = b*b - 4*a*c;
            IntersectionPoint res;
            res.intersects = false;
            // printf("Discriminant: %9.6f\n", discriminant);
            if (discriminant >= 0) {
                res.intersects = true;
                float d = (-b - sqrt(discriminant) ) / (2.0*a);
                vec3 pos = localRay.at(d);
                res.position = vec423(this->transform * vec324(pos));
                
                res.material = this->getMaterial(pos);
                
                // printf("Position: %9.6f %9.6f %9.6f\n", res.position[0], res.position[1], res.position[2]);
                res.distance = d;
                res.normal = vec423(this->transform * vec324(this->getNormal(localRay.at(res.distance))));
            }
            return res;
        }
};