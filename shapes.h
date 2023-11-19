#include "vec3.h"
#include "ray.h"
#include "color.h"
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <cmath>

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

typedef Eigen::Matrix<float, 4, 4> Matrix4;

#define IDENTITY Eigen::Affine3f::Identity()

#define PI 3.14159265358979323846
#define DEG_TO_RAD(X) (X*PI)/180

__device__ Matrix4 translate(Matrix4 m, float x, float y, float z) {
    return (Matrix4{
            {1.0, 0.0, 0.0, x},
            {0.0, 1.0, 0.0, y},
            {0.0, 0.0, 1.0, z},
            {0.0, 0.0, 0.0, 1.0}}) * (m);
}

__device__ Matrix4 scale(Matrix4 m, float x, float y, float z) {
    return (Matrix4{
            {x, 0.0, 0.0, 0.0},
            {0.0, y, 0.0, 0.0},
            {0.0, 0.0, z, 0.0},
            {0.0, 0.0, 0.0, 1.0}}) * (m);
}

/* Rodrigues' Rotation Formula */
__device__ Matrix4 rotate(Matrix4 m, vec3 axis, float angle) {
    // vec3 k = axis.normalized();
    // Matrix4 K ({
    //     {0, -k.z(), -k.y(), 0},
    //     {k.z(), 0, -k.x(), 0},
    //     {-k.y(), k.x(), 0, 0},
    //     {0, 0, 0, 1}
    // });
    // return (IDENTITY + sin(DEG_TO_RAD(angle))*K + (1.0f-cos(DEG_TO_RAD(angle)))*K*K) * m;
    
}

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

struct RayPath {
    IntersectionPoint first;
    IntersectionPoint second;
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
        Eigen::Affine3f transform;
        __device__ Shape(Eigen::Affine3f transform) {
            this->transform = transform;
        }
    public:
        Material material;
        __device__ virtual bool intersects(ray& r) = 0;
        __device__ virtual IntersectionPoint getIntersection(ray& r) = 0;
        __device__ virtual vec3 getNormal(vec3 position) = 0;
        __device__ virtual Material getMaterial(vec3 position) = 0;
};

/* Plane on origin with normal (0, 0, 1) */
class Plane : public Shape {
    private:
        float xMin;
        float yMin;
        
        float xMax;
        float yMax;
    public:
        __device__ Plane(Material material, Eigen::Affine3f transform, float xMin = std::numeric_limits<float>::lowest(), float yMin = std::numeric_limits<float>::lowest(),
        float xMax = std::numeric_limits<float>::max(), float yMax = std::numeric_limits<float>::max()) : Shape(transform),
         xMin(xMin), yMin(yMin), xMax(xMax), yMax(yMax){this->material = material;}
        __device__ vec3 getPoint() {return vec3(0, 0, 0);}
        __device__ vec3 getNormal(vec3 position) {return vec3(0, 0, 1);}
        __device__ bool intersects (ray& r) {
            vec3 localOrigin = (this->transform.inverse() * r.origin());
            vec3 localDirection = (this->transform.linear().inverse() * r.direction());
            return localDirection.dot(vec3(0, 0, 1)) != 0;
            }
        __device__ Material getMaterial(vec3 position) {return this->material;}
        __device__ IntersectionPoint getIntersection(ray &r) {
            vec3 localOrigin = this->transform.inverse() * r.origin();
            vec3 localDirection = (this->transform.inverse().linear() * r.direction()).normalized();
            ray localRay = ray(localOrigin, localDirection);
            IntersectionPoint res = IntersectionPoint();
            if (localRay.direction().dot(vec3(0, 0, 1)) != 0) {
                float d = ((-1 * localRay.origin()).dot(vec3(0, 0, 1)))/(localRay.direction().dot(vec3(0, 0, 1)));
                vec3 pos = localRay.at(d);
                if (this -> xMin < pos.x() && this -> yMin < pos.y()
                    && this -> xMax > pos.x() && this -> yMax > pos.y()) {
                    res.distance = d;
                    res.intersects = true;
                    res.normal =  this->transform.linear() * (this->getNormal(pos));
                    res.position = this->transform * (pos);
                    res.material = this->getMaterial(pos);               
                }
            }
            return res;
        }
};

/* Cube centred at origin with side length 1 */
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
        __device__ Cube(Material material, Eigen::Affine3f transform) : material(material), Shape(transform) {
            vec3 cornerLeft(-0.5, -0.5, -0.5);
            vec3 cornerRight(0.5, 0.5, 0.5);
            this->xMin = min(cornerLeft.x(), cornerRight.x());
            this->yMin = min(cornerLeft.y(), cornerRight.y());
            this->zMin = min(cornerLeft.z(), cornerRight.z());
            this->xMax = max(cornerLeft.x(), cornerRight.x());
            this->yMax = max(cornerLeft.y(), cornerRight.y());
            this->zMax = max(cornerLeft.z(), cornerRight.z());
        }

        __device__ vec3 getNormal(vec3 position) {
            // vec3 vMax = vec3(this->xMax, this->yMax, this->zMax);
            // vec3 vMin = vec3(this->xMin, this->yMin, this->zMin);
            // vec3 center = (vMax + vMin) * (0.5f);
            // vec3 res = position - center;
            float t = max(abs(position.x()), max(abs(position.y()), abs(position.z())));
            vec3 res;
            if (abs(position.x()) == t) {
                res = vec3(position.x(), 0, 0);
            } else if (abs(position.y()) == t) {
                res = vec3(0, position.y(), 0);
            } else {
                res = vec3(0, 0, position.z());
            }
            // printf("Position: %2.3f %2.3f %2.3f, Norm: %2.3f %2.3f %2.3f\n", position[0], position[1], position[2], res.normalized()[0], res.normalized()[1], res.normalized()[2]);
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
            vec3 localOrigin = this->transform.inverse() * r.origin();
            vec3 localDirection = this->transform.inverse().linear() * r.direction().normalized();
            ray localRay = ray(localOrigin, localDirection);

            float tx1 = (xMin - localRay.origin()[0]) / localRay.direction()[0];
            float tx2 = (xMax - localRay.origin()[0]) / localRay.direction()[0];
            float ty1 = (yMin - localRay.origin()[1]) / localRay.direction()[1];
            float ty2 = (yMax - localRay.origin()[1]) / localRay.direction()[1];
            float tz1 = (zMin - localRay.origin()[2]) / localRay.direction()[2];
            float tz2 = (zMax - localRay.origin()[2]) / localRay.direction()[2];

            float tNear = max(min(tx1, tx2), max(min(ty1, ty2), min(tz1, tz2)));
            float tFar = min(max(tx1, tx2), min(max(ty1, ty2), max(tz1, tz2)));

            // printf("Near: %9.6f, Far: %9.6f\n", tNear, tFar);

            IntersectionPoint res;
            res.intersects = false;

            if (!(tNear > tFar || tFar < 0)) {
                vec3 pos = localRay.at(tNear);
                res.intersects = true;
                res.distance = tNear;
                res.position = this->transform * pos;
                res.material = this->getMaterial(res.position);
                res.normal = this->transform.linear() * this->getNormal(pos);
            }
            return res;
        }
};


/* Cylinder of radius 0.5 with centre origin and limits (0, 0, -0.5) and (0, 0, 0.5) */
class Cylinder : public Shape {
    float zmin = -0.5f;
    float zmax = 0.5f;
    public:
        __device__ Cylinder(Material material, Eigen::Affine3f transform) : Shape(transform) {this->material = material;}
        __device__ Material getMaterial(vec3 position) {
            return this->material;
        }

        __device__ vec3 getNormal(vec3 position) {
            return (position - vec3(0, 0, position.z())).normalized();
        }
        __device__ bool intersects(ray& r) {
            return false;
        }

        __device__ IntersectionPoint getIntersectionDisc(ray& localRay) {

            IntersectionPoint res;
            res.intersects = false;
            /* Check intersection with discs */
            vec3 p(0, 0, zmax);
            vec3 n_disc(0, 0, 1);

            if (localRay.direction().dot(n_disc) == 0)
                return res;



            float t;

            t = (p - localRay.origin()).dot(n_disc)/(localRay.direction().dot(n_disc));

            vec3 pos = localRay.at(t);

            if ((pos - p).norm() > 0.5)
                return res;

            res.intersects = true;
            res.position = this->transform * pos;
            res.material = this->getMaterial(pos);
            res.normal = this->transform.linear() * n_disc;
            res.distance = t;
            return res;
        }

        __device__ IntersectionPoint getIntersection(ray& r) {
            vec3 localOrigin = this->transform.inverse() * r.origin();
            vec3 localDirection = this->transform.inverse().linear() * r.direction().normalized();
            ray localRay(localOrigin, localDirection);
            float a = pow(localDirection.x(), 2) + pow(localDirection.y(), 2);
            float b = 2*(localDirection.x() * localOrigin.x() + localDirection.y() * localOrigin.y());
            float c = pow(localOrigin.x(), 2) + pow(localOrigin.y(), 2) - 0.25;   
            float discriminant = b*b - 4*a*c;

            IntersectionPoint res;
            res.intersects = false;

            if (discriminant < 0)
                return getIntersectionDisc(localRay);
            
            
            float d = (-b - sqrt(discriminant) ) / (2.0*a);
            vec3 pos = localRay.at(d);

            if (pos.z() < zmin || pos.z() > zmax)
                return getIntersectionDisc(localRay);

            res.intersects = true;
            res.position = (this->transform * pos);
            res.material = this->getMaterial(pos);
            res.distance = d;
            // printf("Rows: %d, Cols: %d\n", this->transform.rows(), this->transform.cols());
            res.normal = (this->transform.linear() * this->getNormal(localRay.at(res.distance)));
            return res;
        }
};


/* Sphere centred at origin with radius 0.5f */
class Sphere : public Shape {
    public:
        __device__ Sphere(Material material, Eigen::Affine3f transform) : Shape(transform) {this->material = material;}
        __device__ Material getMaterial(vec3 position) {
            return this->material;
        }

        __device__ vec3 getNormal(vec3 position) {
            return (position).normalized();
        }

        __device__ bool intersects(ray& r) {
            return false;
        }

        __device__ IntersectionPoint getIntersection(ray& r) {
            vec3 localOrigin = this->transform.inverse() * r.origin();
            vec3 localDirection = this->transform.inverse().linear() * r.direction().normalized();
            // printf("Direction dot product: %9.6f\n", localDirection.dot(r.direction())/(localDirection.norm() * r.direction().norm()));
            // printf("LocalDirection: %9.6f %9.6f %9.6f\n", localDirection[0], localDirection[1], localDirection[2]);
            ray localRay = ray(localOrigin, localDirection);
            float a = localDirection.dot(localDirection);
            float b = 2.0f*(localDirection.dot(localOrigin));
            float c = (localOrigin).dot(localOrigin) - 0.25;
            float discriminant = b*b - 4*a*c;
            IntersectionPoint res;
            res.intersects = false;
            if (discriminant < 0) {
                return res;
            }
            // printf("A: %9.6f B: %9.6f C: %0.6f\n", a, b, c);
            //printf("Sphere LocalDirection: %9.6f %9.6f %9.6f\n",localRay.direction()[0],localRay.direction()[1],localRay.direction()[2]);
            //printf("LocalOrigin: %9.6f %9.6f %9.6f\n",localRay.origin()[0],localRay.origin()[1],localRay.origin()[2]);
            // printf("Discriminant: %9.6f\n", discriminant);
            res.intersects = true;
            float d = (-b - sqrt(discriminant) ) / (2.0*a);
            vec3 pos = localRay.at(d);
            res.position = (this->transform * pos);
            res.material = this->getMaterial(pos);
            res.distance = d;
            // printf("Rows: %d, Cols: %d\n", this->transform.rows(), this->transform.cols());
            res.normal = (this->transform.linear() * this->getNormal(localRay.at(res.distance)));
            return res;
        }
};