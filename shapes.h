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
        float emissive;

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
    int n_intersections;
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
        __device__ ray transformRay(ray& r) {
            return ray(transformPointLocal(r.origin()), transformVectorLocal(r.direction()));
        }
        __device__ vec3 transformPointLocal(vec3 v) {return this->transform.inverse() * v;}
        __device__ vec3 transformVectorLocal(vec3 v) {return this->transform.inverse().linear() * v;}
        __device__ vec3 transformPointWorld(vec3 v) {return this->transform * v;}
        __device__ vec3 transformVectorWorld(vec3 v) {return this->transform.linear() * v;}
        __device__ vec3 transformNormalWorld(vec3 n) {return this->transform.linear() * n;}
        __device__ virtual RayPath getRayPath(ray& r) = 0;
        __device__ RayPath getIntersections(ray& r) {
            return getRayPath(transformRay(r));
        }
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
        __device__ Material getMaterial(vec3 position) {return this->material;}
        __device__ RayPath getRayPath(ray &r) {
            RayPath res;
            if (r.direction().dot(vec3(0, 0, 1)) != 0) {
                res.n_intersections = 1;
                float d = ((-1 * r.origin()).dot(vec3(0, 0, 1)))/(r.direction().dot(vec3(0, 0, 1)));
                vec3 pos = r.at(d);
                if (this -> xMin < pos.x() && this -> yMin < pos.y()
                    && this -> xMax > pos.x() && this -> yMax > pos.y()) {
                    res.first.distance = d;
                    res.first.intersects = true;
                    res.first.normal =  transformNormalWorld(vec3(0, 0, 1));
                    res.first.position = transformPointWorld(pos);
                    res.first.material = material;               
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

        __device__ RayPath getRayPath(ray& r) {
            float dirfrac_x = 1.0f / r.direction().x();
            float dirfrac_y = 1.0f / r.direction().y();
            float dirfrac_z = 1.0f / r.direction().z();
            // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
            // r.org is origin of ray
            float t1 = (xMin - r.origin().x())*dirfrac_x;
            float t2 = (xMax - r.origin().x())*dirfrac_x;
            float t3 = (yMin - r.origin().y())*dirfrac_y;
            float t4 = (yMax - r.origin().y())*dirfrac_y;
            float t5 = (zMin - r.origin().z())*dirfrac_z;
            float t6 = (zMax - r.origin().z())*dirfrac_z;

            float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
            float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

            RayPath res;
            res.n_intersections = 0;

            // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
            float t;
            if (tmax < 0)
            {
                t = tmax;
                return res;
            }

            // if tmin > tmax, ray doesn't intersect AABB
            if (tmin > tmax)
            {
                t = tmax;
                return res;
            }

            t = tmin;
            vec3 pos = r.at(t);
            res.n_intersections = 2;
            res.first.distance = t;
            res.first.intersects = true;
            res.first.material = material;
            res.first.position = transformPointWorld(pos);
            res.first.normal = transformNormalWorld(getNormal(pos));

            pos = r.at(tmax);
            res.second.distance = tmax;
            res.second.intersects = true;
            res.second.material = material;
            res.second.position = transformPointWorld(pos);
            res.second.normal = transformNormalWorld(getNormal(pos));
            return res;
        }
};


/* Cylinder of radius 0.5 with centre origin and limits (0, 0, -0.5) and (0, 0, 0.5) */
class Cylinder : public Shape {
    float zmin = -0.5f;
    float zmax = 0.5f;
    public:
        __device__ Cylinder(Material material, Eigen::Affine3f transform) : Shape(transform) {this->material = material;}

        __device__ vec3 getNormal(vec3 position) {
            return (position - vec3(0, 0, position.z())).normalized();
        }

        __device__ IntersectionPoint getIntersectionDisc(ray& r, vec3 disc_normal) {

            IntersectionPoint res;
            res.intersects = false;
            vec3 p = disc_normal * 0.5;

            if (r.direction().dot(disc_normal) == 0)
                return res;



            float t;

            t = (p - r.origin()).dot(disc_normal)/(r.direction().dot(disc_normal));

            vec3 pos = r.at(t);

            if ((pos - p).norm() > 0.5)
                return res;

            res.intersects = true;
            res.position = transformPointWorld(pos);
            res.material = material;
            res.normal = transformNormalWorld(disc_normal);
            res.distance = t;
            return res;
        }

        __device__ RayPath getRayPath(ray& r) {
            float a = pow(r.direction().x(), 2) + pow(r.direction().y(), 2);
            float b = 2*(r.direction().x() * r.origin().x() + r.direction().y() * r.origin().y());
            float c = pow(r.origin().x(), 2) + pow(r.origin().y(), 2) - 0.25;   
            float discriminant = b*b - 4*a*c;

            RayPath res;
            res.n_intersections = 0;

            IntersectionPoint disc1 = getIntersectionDisc(r, vec3(0, 0, 1));
            IntersectionPoint disc2 = getIntersectionDisc(r, vec3(0, 0, -1));
            IntersectionPoint cylinder1;
            IntersectionPoint cylinder2;

            if (discriminant > 0) {
                float d = (-b - sqrt(discriminant) ) / (2.0*a);
                vec3 pos = r.at(d);

                cylinder1.intersects = (zmin < pos.z() && pos.z() < zmax);
                cylinder1.position = transformPointWorld(pos);
                cylinder1.material = material;
                cylinder1.distance = d;
                cylinder1.normal = transformNormalWorld(getNormal(pos));

                d = (-b + sqrt(discriminant) ) / (2.0*a);
                pos = r.at(d);

                cylinder2.intersects = (zmin < pos.z() && pos.z() < zmax);
                cylinder2.position = transformPointWorld(pos);
                cylinder2.material = material;
                cylinder2.distance = d;
                cylinder2.normal = transformNormalWorld(getNormal(pos));
            }

            
            if (disc1.intersects && disc2.intersects) {
                /* Case 1 - Ray intersects both discs */
                res.n_intersections = 2;
                res.first = disc1;
                res.second = disc2;
            } else if (disc1.intersects && discriminant > 0) {
                /* Case 2 - Ray intersects disc one and cylinder */
                res.n_intersections = 2;
                res.first = disc1;
                res.second = cylinder1.intersects ? cylinder1 : cylinder2;
            } else if (disc2.intersects && discriminant > 0) {
                /* Case 3 - Ray intersects disc two and cylinder */
                res.n_intersections = 2;
                res.first = disc2;
                res.second = cylinder1.intersects ? cylinder1 : cylinder2;
            } else if (discriminant > 0 && cylinder1.intersects && cylinder2.intersects) {
                /* Case 4 - Ray intersects cylinder twice */
                res.n_intersections = 2;
                res.first = cylinder1;
                res.second = cylinder2;
            } else {
                /* Case 5 - No ray intersections */
                res.n_intersections = 0;
            }

            return res;
        }
};


/* Sphere centred at origin with radius 0.5f */
class Sphere : public Shape {
    public:
        __device__ Sphere(Material material, Eigen::Affine3f transform) : Shape(transform) {this->material = material;}

        __device__ vec3 getNormal(vec3 position) {
            return (position).normalized();
        }

        __device__ RayPath getRayPath(ray& r) {
            float a = r.direction().dot(r.direction());
            float b = 2.0f*(r.direction().dot(r.origin()));
            float c = (r.origin()).dot(r.origin()) - 0.25;
            float discriminant = b*b - 4*a*c;
            RayPath res;
            if (discriminant < 0) {
                res.n_intersections = 0;
            } else if (discriminant == 0.0f) {
                res.n_intersections = 1;
            } else {
                res.n_intersections = 2;
            }

            float d;
            vec3 pos;

            vec3 wOrigin = transformPointWorld(vec3(0, 0, 0));

            switch (res.n_intersections)
            {
            case 1:
                d = (-b ) / (2.0*a);
                pos = r.at(d);
                res.first.intersects = true;
                res.first.position = transformPointWorld(pos);
                res.first.material = material;
                res.first.distance = d;
                res.first.normal = this->transformNormalWorld(this->getNormal(pos));
                break;
            case 2:
                d = (-b - sqrt(discriminant)) / (2.0*a);
                pos = r.at(d);
                res.first.intersects = true;
                res.first.position = transformPointWorld(pos);
                res.first.material = material;
                res.first.distance = d;
                res.first.normal = this->transformNormalWorld(this->getNormal(pos));

                d = (-b - sqrt(discriminant)) / (2.0*a);
                pos = r.at(d);
                res.second.intersects = true;
                res.second.position = transformPointWorld(pos);
                res.second.material = material;
                res.second.distance = d;
                res.second.normal = this->transformNormalWorld(this->getNormal(pos));
                break;
            default:
                break;
            }
            // printf("A: %9.6f B: %9.6f C: %0.6f\n", a, b, c);
            //printf("Sphere LocalDirection: %9.6f %9.6f %9.6f\n",localRay.direction()[0],localRay.direction()[1],localRay.direction()[2]);
            //printf("LocalOrigin: %9.6f %9.6f %9.6f\n",localRay.origin()[0],localRay.origin()[1],localRay.origin()[2]);
            // printf("Discriminant: %9.6f\n", discriminant);
            return res;
        }
};


// class Triangle {
//     vec3 v0;
//     vec3 v1;
//     vec3 v2;
//     Material material;

//     public:
//         __device__ Triangle(Material material, vec3 v0, vec3 v1, vec3 v2) : v0(v0), v1(v1), v2(v2), material(material) {}
//         __device__ IntersectionPoint getIntersection(ray& r) {
//             IntersectionPoint res;
//             res.intersects = false;
//             const float EPSILON = 0.0000001;
//             vec3 edge1, edge2, rayVecXe2, s, sXe1;
//             float det, invDet, u, v;
//             edge1 = v1 - v0;
//             edge2 = v2 - v0;
//             rayVecXe2 = r.direction().cross(edge2);
//             det = edge1.dot(rayVecXe2);

//             if (det > -EPSILON && det < EPSILON)
//                 return res;    // This ray is parallel to this triangle.

//             invDet = 1.0 / det;
//             s = r.origin() - v0;
//             u = invDet * s.dot(rayVecXe2);

//             if (u < 0.0 || u > 1.0)
//                 return res;

//             sXe1 = s.cross(edge1);
//             v = invDet * r.direction().dot(sXe1);

//             if (v < 0.0 || u + v > 1.0)
//                 return res;

//             // At this stage we can compute t to find out where the intersection point is on the line.
//             float t = invDet * edge2.dot(sXe1);

//             if (t > EPSILON) // ray intersection
//             {
//                 res.intersects = true;
//                 res.distance = t;
//                 res.position = r.at(t);
//                 res.material = material;
//                 res.normal = edge1.cross(edge1);
//                 return res;
//             }
//             else // This means that there is a line intersection but not a ray intersection.
//                 return res;
//         }
// };
