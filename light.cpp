#include <shapes.h>

// float fresnel_exact(vec3 i, vec3 m, float n1, float n2) {
//     float c = i.dot(m);
//     float g = sqrtf((pow(n1, 2))/(pow(n2, 2)) - 1 + pow(c, 2));
//     return (1/2)*(pow(g-c, 2)/(pow(g+c, 2)))*(1+)

// }

/* Compute Refraction Probability */
/* If random number < refraction then refract */
/* Else */
/* Compute Specular Probability */
/* If random number < specular then reflect */
/* Else */
/* Choose random cosine weighted hemisphere direction and reflect */
/* If material is emissive - multiply contribution by coefficient and emission */
/* Multiply coefficient by BRDF/PDF */

float GGXNormal(vec3 N, vec3 H, float roughness) {
    float a2 = pow(roughness, 2);
    float NdotH = N.dot(H);
    float d = ((NdotH * a2 - NdotH) * NdotH + 1);
    return a2 / (d * d * PI);
}

float GeometryMaskingTerm(float NdotL, float NdotV, float roughness)
{
	// Karis notes they use alpha / 2 (or roughness^2 / 2)
	float k = roughness*roughness / 2;

	// Compute G(v) and G(l).  These equations directly from Schlick 1994
	//     (Though note, Schlick's notation is cryptic and confusing.)
	float g_v = NdotV / (NdotV*(1 - k) + k);
	float g_l = NdotL / (NdotL*(1 - k) + k);
	return g_v * g_l;
}

vec3 schlickFresnel(vec3 f0, float lDotH)
{
	return f0 + (vec3(1.0f, 1.0f, 1.0f) - f0) * pow(1.0f - lDotH, 5.0f);
}

vec3 getGGXMicrofacet(float u1, float u2, float roughness, vec3 N)
{

	// Get an orthonormal basis from the normal
	vec3 B = getPerpendicularVector(N);
	vec3 T = cross(B, N);

	// GGX NDF sampling
	float a2 = roughness * roughness;
	float cosThetaH = sqrt(max(0.0f, (1.0-u1)/((a2-1.0)*u1+1) ));
	float sinThetaH = sqrt(max(0.0f, 1.0f - cosThetaH * cosThetaH));
	float phiH = u2 * M_PI * 2.0f;

	// Get our GGX NDF sample (i.e., the half vector)
	return T * (sinThetaH * cos(phiH)) +
           B * (sinThetaH * sin(phiH)) +
           hitNorm * cosThetaH;
}