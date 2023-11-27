#pragma once

#include "vec_math.h"
#include "ray.h"
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <cmath>
#include "materials.h"


struct IntersectionPoint {
        bool intersects = false;
        float distance;
        Material material;
        vec3 position;
        vec3 normal;
        bool inside;
};

/* With the exception of the Plane, every shape has two intersections with a ray (if it intersects at all) */
struct RayPath {
    int n_intersections;
    IntersectionPoint first;
    IntersectionPoint second;
};


class Shape {
    protected:
        Eigen::Affine3f transform;
        Eigen::Affine3f transform_linear;
        Eigen::Affine3f transform_inv;
        Eigen::Affine3f transform_inv_linear;
        vec3 wOrigin;
        __device__ Shape(Eigen::Affine3f transform) {
            this->transform = transform;
            this->transform_inv = transform.inverse();
            this->wOrigin = this->transform * vec3(0, 0, 0);
        }
    public:
        Material material;
        __device__ ray transformRay(ray& r) {
            return ray(transformPointLocal(r.origin()), transformVectorLocal(r.direction()));
        }
        
        __device__ vec3 transformPointLocal(vec3 v) {return this->transform_inv * v;}
        
        __device__ vec3 transformVectorLocal(vec3 v) {return this->transform_inv.linear() * v;}
        
        __device__ vec3 transformPointWorld(vec3 v) {return this->transform * v;}
        
        __device__ vec3 transformVectorWorld(vec3 v) {return this->transform.linear() * v;}
        
        /* Because of a bug with Eigen and CUDA - transposing vectors isn't possible 
         *  For now we don't scale any objects - will fix or find workaround later  */
        __device__ vec3 transformNormalWorld(vec3 n) {return this->transform.linear() * n;}
        
        __device__ virtual RayPath getRayPath(ray& r) = 0;
        
        __device__ RayPath getIntersections(ray& r) {
            RayPath res = getRayPath(transformRay(r));
            if (res.n_intersections == 1 && res.first.distance <= SMALL_NUMBER) {
                res.n_intersections = 0;
            } else if (res.n_intersections == 2) {
                if (res.first.distance <= SMALL_NUMBER && res.second.distance <= SMALL_NUMBER) {
                    res.n_intersections = 0;
                } else if (res.first.distance <= SMALL_NUMBER) {
                    res.n_intersections = 1;
                    res.first = res.second;
                } else if (res.second.distance <= SMALL_NUMBER) {
                    res.n_intersections = 1;
                }
            }
            return res;
        }
};


__device__ IntersectionPoint getNearestIntersection(ray& r, Shape** scene, int n_objects) {
    IntersectionPoint min_p;
    min_p.intersects = false;
    for (int k = 0; k < n_objects; k++) {
        Shape *obj = scene[k];
        RayPath p = obj->getIntersections(r);
        IntersectionPoint point;
        if (p.n_intersections == 1) {
            point = p.first;
        } else if (p.n_intersections == 2) {
            point = (p.second.distance < p.first.distance) ? p.second : p.first;
        } else {
            point.intersects = false;
        }
        if (point.intersects && (!min_p.intersects || point.distance < min_p.distance)) {
            min_p = point;
            min_p.inside = min_p.normal.dot(r.direction()) > 0;
            if (min_p.inside) {min_p.normal = -1 * min_p.normal;}
        }
    }
    return min_p;
}

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
        __device__ Material getMaterial(vec3 position) {return this->material;}
        __device__ RayPath getRayPath(ray &r) {
            RayPath res;
            if (r.direction().dot(vec3(0, 0, 1)) != 0) {
                
                float d = ((-1 * r.origin()).dot(vec3(0, 0, 1)))/(r.direction().dot(vec3(0, 0, 1)));
                res.n_intersections = 1;
                vec3 pos = r.at(d);
                if (this -> xMin < pos.x() && this -> yMin < pos.y()
                    && this -> xMax > pos.x() && this -> yMax > pos.y()) {
                    res.first.distance = d;
                    res.first.intersects = true;
                    res.first.normal =  (transformPointWorld(vec3(0, 0, 1).dot(r.direction()) > 0 ? vec3(0, 0, -1) : vec3(0, 0, 1)) - wOrigin).normalized();
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
            this->xMin = std::min(cornerLeft.x(), cornerRight.x());
            this->yMin = std::min(cornerLeft.y(), cornerRight.y());
            this->zMin = std::min(cornerLeft.z(), cornerRight.z());
            this->xMax = std::max(cornerLeft.x(), cornerRight.x());
            this->yMax = std::max(cornerLeft.y(), cornerRight.y());
            this->zMax = std::max(cornerLeft.z(), cornerRight.z());
        }

        __device__ vec3 getNormal(vec3 position) {
            float t = std::max(abs(position.x()), std::max(abs(position.y()), abs(position.z())));
            vec3 res;
            if (abs(position.x()) == t) {
                res = vec3(position.x(), 0, 0);
            } else if (abs(position.y()) == t) {
                res = vec3(0, position.y(), 0);
            } else {
                res = vec3(0, 0, position.z());
            }
            return (transformPointWorld(res) - wOrigin).normalized();
        }

        __device__ RayPath getRayPath(ray& r) {
            float x = 1.0f / r.direction().x();
            float y = 1.0f / r.direction().y();
            float z = 1.0f / r.direction().z();

            float t1 = (xMin - r.origin().x())*x;
            float t2 = (xMax - r.origin().x())*x;
            float t3 = (yMin - r.origin().y())*y;
            float t4 = (yMax - r.origin().y())*y;
            float t5 = (zMin - r.origin().z())*z;
            float t6 = (zMax - r.origin().z())*z;

            float tmin = std::max(std::max(std::min(t1, t2), std::min(t3, t4)), std::min(t5, t6));
            float tmax = std::min(std::min(std::max(t1, t2), std::max(t3, t4)), std::max(t5, t6));

            RayPath res;
            res.n_intersections = 0;

            /* Cube is behind the origin entirely */
            float t;
            if (tmax < 0)
            {
                t = tmax;
                return res;
            }

            /* Ray doesn't intersect Cube */
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
            res.first.normal = getNormal(pos);

            pos = r.at(tmax);
            res.second.distance = tmax;
            res.second.intersects = true;
            res.second.material = material;
            res.second.position = transformPointWorld(pos);
            res.second.normal = getNormal(pos);
            return res;
        }
};


/* Cylinder of radius 0.5 with centre origin and limits (0, 0, -0.5) and (0, 0, 0.5) */
class Cylinder : public Shape {
    float zmin = -0.5f;
    float zmax = 0.5f;
    public:
        __device__ Cylinder(Material material, Eigen::Affine3f transform) : Shape(transform) {this->material = material;}

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
            res.normal = (transformPointWorld(disc_normal) - wOrigin).normalized();
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
                cylinder1.normal = (transformPointWorld(vec3(pos[0], pos[1], 0)) - wOrigin).normalized();

                d = (-b + sqrt(discriminant) ) / (2.0*a);
                pos = r.at(d);

                cylinder2.intersects = (zmin < pos.z() && pos.z() < zmax);
                cylinder2.position = transformPointWorld(pos);
                cylinder2.material = material;
                cylinder2.distance = d;
                cylinder2.normal = (transformPointWorld(vec3(pos[0], pos[1], 0)) - wOrigin).normalized();
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

        __device__ RayPath getRayPath(ray& r) {
            float a = r.direction().dot(r.direction());
            float b = 2.0f*(r.direction().dot(r.origin()));
            float c = (r.origin()).dot(r.origin()) - 0.25;
            float discriminant = b*b - 4*a*c;
            RayPath res;
            if (discriminant < 0) {
                res.n_intersections = 0;
            } else {
                res.n_intersections = 2;
            }

            float d;
            vec3 pos;

            if (res.n_intersections == 2) {
                d = (-b - sqrt(discriminant)) / (2.0*a);
                pos = r.at(d);
                res.first.intersects = true;
                res.first.position = transformPointWorld(pos);
                res.first.material = material;
                res.first.distance = d;
                res.first.normal = (this->transformPointWorld(pos) - wOrigin).normalized();

                d = (-b + sqrt(discriminant)) / (2.0*a);
                pos = r.at(d);
                res.second.intersects = true;
                res.second.position = transformPointWorld(pos);
                res.second.material = material;
                res.second.distance = d;
                res.second.normal = (this->transformPointWorld(pos) - wOrigin).normalized();
            }
            return res;
        }
};
