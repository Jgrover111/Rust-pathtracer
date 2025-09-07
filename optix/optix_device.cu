#include <optix.h>
#include <optix_device.h>
#include <stdint.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <math_constants.h>

// Keep this in sync with optix_wrapper.cpp
struct Params
{
  CUdeviceptr              out_rgb;      // float3*
  CUdeviceptr              out_bayer;    // float*
  int                      width;
  int                      height;
  int                      spp;
  int                      frame;
  int                      bayer_pattern; // 0:RGGB,1:BGGR,2:GRBG,3:GBRG
  float3                   cam_eye;
  float3                   cam_u;
  float3                   cam_v;
  float3                   cam_w;
  OptixTraversableHandle   handle;

  // Geometry arrays
  CUdeviceptr              d_vertices;   // float3*
  CUdeviceptr              d_indices;    // uint3*
  CUdeviceptr              d_normals;    // float3*
  uint32_t                 num_triangles;

  // Rectangular area light centered at light_pos with half extents light_half
  float3                   light_pos;
  float3                   light_emit;
  float3                   light_normal;
  float2                   light_half;
};

extern "C" {
__constant__ Params params;
}

// SBT data for each primitive
struct HitgroupData
{
  float3 kd; // diffuse reflectance
  float3 ke; // emission
};

enum {
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_SHADOW   = 1,
    RAY_TYPE_COUNT    = 2
};

// --- float3 helpers (device-only) -----------------------------------------

static __forceinline__ __device__ float3 make3(float s) { return make_float3(s,s,s); }

static __forceinline__ __device__ float3 operator+(const float3& a, const float3& b) {
  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}
static __forceinline__ __device__ float3 operator-(const float3& a, const float3& b) {
  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
static __forceinline__ __device__ float3 operator*(const float3& a, float s) {
  return make_float3(a.x*s, a.y*s, a.z*s);
}
static __forceinline__ __device__ float3 operator*(float s, const float3& a) {
  return make_float3(a.x*s, a.y*s, a.z*s);
}
static __forceinline__ __device__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
static __forceinline__ __device__ float3 mul(const float3& a, const float3& b) {
  return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}
static __forceinline__ __device__ float3 operator/(const float3& a, float s) {
  float inv = 1.0f/s;
  return make_float3(a.x*inv, a.y*inv, a.z*inv);
}
static __forceinline__ __device__ float3& operator+=(float3& a, const float3& b) {
    a.x += b.x; a.y += b.y; a.z += b.z; return a;
}
static __forceinline__ __device__ float dot(const float3& a, const float3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
static __forceinline__ __device__ float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y*b.z - a.z*b.y,
                       a.z*b.x - a.x*b.z,
                       a.x*b.y - a.y*b.x);
}
static __forceinline__ __device__ float length(const float3& v) {
    return sqrtf(dot(v, v));
}
static __forceinline__ __device__ float3 normalize(const float3& v) {
    const float len = length(v);
    return (len > 0.f) ? v / len : make_float3(0.f, 0.f, 0.f);
}
static __forceinline__ __device__ float  dot3(const float3& a, const float3& b) {
  return a.x*b.x + a.y*b.y + a.z*b.z;
}
static __forceinline__ __device__ float3 cross3(const float3& a, const float3& b) {
  return make_float3(a.y*b.z - a.z*b.y,
                     a.z*b.x - a.x*b.z,
                     a.x*b.y - a.y*b.x);
}
static __forceinline__ __device__ float  length3(const float3& a) {
  return sqrtf(dot3(a,a));
}
static __forceinline__ __device__ float3 normalize3(const float3& a) {
  return a / fmaxf(length3(a), 1e-20f);
}
static __forceinline__ __device__ float  clamp01(float x) {
  return fminf(fmaxf(x, 0.0f), 1.0f);
}
static __forceinline__ __device__ unsigned int pcg(unsigned long long& state) {
  unsigned long long oldstate = state;
  state = oldstate * 6364136223846793005ULL + 1442695040888963407ULL;
  unsigned int xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  unsigned int rot = oldstate >> 59u;
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static __forceinline__ __device__ float rnd(unsigned long long& state) {
  return float(pcg(state) >> 8) * (1.0f / float(0x01000000u));
}

static __forceinline__ __device__ void make_onb(const float3& n, float3& t, float3& b)
{
  if (n.z < -0.9999999f) {
    t = make_float3(0.0f, -1.0f, 0.0f);
    b = make_float3(-1.0f, 0.0f, 0.0f);
  } else {
    const float a = 1.0f / (1.0f + n.z);
    const float naxy = -n.x * n.y * a;
    t = make_float3(1.0f - n.x * n.x * a, naxy, -n.x);
    b = make_float3(naxy, 1.0f - n.y * n.y * a, -n.y);
  }
}

// --- payload packing -------------------------------------------------------

static __forceinline__ __device__ void packPtr(void* ptr, unsigned int& u0, unsigned int& u1) {
  uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
  u0 = static_cast<unsigned int>(uptr & 0xFFFFFFFFu);
  u1 = static_cast<unsigned int>(uptr >> 32);
}

template<typename T>
static __forceinline__ __device__ T* unpackPtr() {
  uint64_t u0 = static_cast<uint64_t>(optixGetPayload_0());
  uint64_t u1 = static_cast<uint64_t>(optixGetPayload_1());
  uint64_t uptr = (u1 << 32) | u0;
  return reinterpret_cast<T*>(uptr);
}

// --- per-ray data ----------------------------------------------------------

struct PRD {
    float3 radiance;
    float3 throughput;
    float3 origin;
    float3 direction;
    unsigned long long seed;
    int    depth;
    int    done;
    // MIS: previous bounce's BSDF pdf and validity flag
    float  prev_pdf_bsdf;
    int    prev_pdf_valid;
};

// --- access helpers for CUdeviceptr arrays --------------------------------

template<typename T>
static __forceinline__ __device__ T* ptr_at(CUdeviceptr p) {
  return reinterpret_cast<T*>(p);
}

// --- Miss programs ---------------------------------------------------------

extern "C" __global__ void __miss__ms_radiance()
{
  PRD* prd = unpackPtr<PRD>();
  prd->radiance = make3(0.0f);
  prd->done = 1;
}

extern "C" __global__ void __miss__ms_shadow()
{
  // 0 means not occluded; anyhit will set to 1
  optixSetPayload_0(0u);
}

// --- Anyhit for shadow rays: mark occluded and terminate -------------------

extern "C" __global__ void __anyhit__ah_shadow()
{
    optixSetPayload_0(1u);
    optixTerminateRay();
}

// --- Closest hit for radiance rays ----------------------------------------

extern "C" __global__ void __closesthit__ch()
{
    PRD& prd = *unpackPtr<PRD>();

    const unsigned int prim = optixGetPrimitiveIndex();

    // Geometry from Params (NOT from SBT)
    const uint3*  indices  = reinterpret_cast<const uint3*>(params.d_indices);
    const float3* vertices = reinterpret_cast<const float3*>(params.d_vertices);
    const uint3 tri = indices[prim];

    const float3 v0 = vertices[tri.x];
    const float3 v1 = vertices[tri.y];
    const float3 v2 = vertices[tri.z];

    const float2 bc = optixGetTriangleBarycentrics();
    const float b1 = bc.x;
    const float b2 = bc.y;
    const float b0 = 1.0f - b1 - b2;

    const float3 P  = v0 * b0 + v1 * b1 + v2 * b2;
    const float3 Ng = normalize3(reinterpret_cast<const float3*>(params.d_normals)[prim]);

    // Fetch per-triangle materials from SBT
    const HitgroupData* hg = reinterpret_cast<const HitgroupData*>(optixGetSbtDataPointer());
    const float3 kd = hg->kd;
    const float3 ke = hg->ke;

    // Sample direct illumination from rectangular area light
    unsigned long long& seed = prd.seed;
    float3 Lo = make3(0.0f);
    {
        float u = (rnd(seed) * 2.0f - 1.0f) * params.light_half.x;
        float v = (rnd(seed) * 2.0f - 1.0f) * params.light_half.y;
        float3 lp = make_float3(params.light_pos.x + u, params.light_pos.y, params.light_pos.z + v);
        float3 L = lp - P;
        float dist2 = fmaxf(dot(L, L), 1e-6f);
        float dist = sqrtf(dist2);
        float3 wi = L / dist;
        float3 light_n = make_float3(0.0f, -1.0f, 0.0f);
        float cosS = fmaxf(0.0f, dot(Ng, wi));
        float cosL = fmaxf(0.0f, dot(params.light_normal, wi * -1.0f));
        if (cosS > 0.0f && cosL > 0.0f) {
            unsigned int occluded = 0u;
            optixTrace(
                params.handle,
                P + Ng * 1e-3f,
                wi,
                0.0f,
                dist - 1e-3f,
                0.0f,
                1,
                OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                RAY_TYPE_SHADOW,
                RAY_TYPE_COUNT,
                RAY_TYPE_SHADOW,
                occluded);
            if (!occluded) {
                float area = 4.0f * params.light_half.x * params.light_half.y;
                float pdf_area = 1.0f / area;
                float pdf_light = pdf_area * dist2 / cosL;
                float pdf_bsdf = cosS * (1.0f / CUDART_PI_F);
                float w = pdf_light / (pdf_light + pdf_bsdf);
                float3 f = kd * (1.0f / CUDART_PI_F);
                float3 contrib = params.light_emit * f * (cosS * cosL / dist2) / pdf_area;
                Lo += contrib * w;
            }
        }
    }

    // Accumulate emission with MIS and direct light
    float3 emission = ke;
    if (prd.prev_pdf_valid && (ke.x > 0.0f || ke.y > 0.0f || ke.z > 0.0f)) {
        float3 L = P - prd.origin;
        float dist2 = fmaxf(dot(L, L), 1e-6f);
        float area = 4.0f * params.light_half.x * params.light_half.y;
        float3 light_n = make_float3(0.0f, -1.0f, 0.0f);
        float cosL = fmaxf(0.0f, dot(params.light_normal, prd.direction * -1.0f));
        float pdf_light = dist2 / (cosL * area);
        float w_bsdf = prd.prev_pdf_bsdf / (prd.prev_pdf_bsdf + pdf_light);
        emission = emission * w_bsdf;
    }
    prd.radiance += emission + Lo;

    // Sample diffuse direction (cosine-weighted)
    float r1 = rnd(seed);
    float r2 = rnd(seed);
    const float phi = 2.0f * CUDART_PI_F * r1;
    const float cosTheta = sqrtf(1.0f - r2);
    const float sinTheta = sqrtf(r2);
    float3 localDir = make_float3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);

    float3 tangent, bitangent;
    make_onb(Ng, tangent, bitangent);
    float3 newDir = normalize(localDir.x * tangent + localDir.y * bitangent + localDir.z * Ng);

    prd.origin = P + Ng * 1e-3f;
    prd.direction = newDir;
    prd.throughput = mul(prd.throughput, kd);
    prd.prev_pdf_bsdf = cosTheta * (1.0f / CUDART_PI_F);
    prd.prev_pdf_valid = 1;

    prd.depth++;
    if (prd.depth >= 5) {
        prd.done = 1;
        return;
    }
    if (prd.depth >= 3) {
        float p = fmaxf(prd.throughput.x, fmaxf(prd.throughput.y, prd.throughput.z));
        if (rnd(seed) > p) {
            prd.done = 1;
            return;
        }
        prd.throughput = prd.throughput / fmaxf(p, 1e-3f);
    }
}

// --- Raygen program --------------------------------------------------------

static __forceinline__ __device__ float3 sample_camera_dir(int x, int y, const float2& jitter)
{
  const uint3  dim = optixGetLaunchDimensions();
  const float  fx  = (float(x) + jitter.x) / float(dim.x);
  const float  fy  = (float(y) + jitter.y) / float(dim.y);
  const float2 d   = make_float2(2.0f*fx - 1.0f, 1.0f - 2.0f*fy);
  return normalize3(params.cam_w + d.x * params.cam_u + d.y * params.cam_v);
}

static __forceinline__ __device__ int bayer_channel_for(int x, int y, int pattern)
{
  const int mx = x & 1;
  const int my = y & 1;
  switch(pattern){
    case 0: return (my==0) ? (mx==0 ? 0:1) : (mx==0 ? 1:2);
    case 1: return (my==0) ? (mx==0 ? 2:1) : (mx==0 ? 1:0);
    case 2: return (my==0) ? (mx==0 ? 1:0) : (mx==0 ? 2:1);
    default:return (my==0) ? (mx==0 ? 1:2) : (mx==0 ? 0:1);
  }
}

extern "C" __global__ void __raygen__rg()
{
  const uint3  idx = optixGetLaunchIndex();
  const int    x   = int(idx.x);
  const int    y   = int(idx.y);
  const int    W   = params.width;
  const int    H   = params.height;
  const int    dst = y * W + x;

  // Trace radiance
  const float3 org = params.cam_eye;
  float3 sum_rgb = make3(0.0f);
  float sum_bayer = 0.0f;
  const int ch = bayer_channel_for(x, y, params.bayer_pattern);
  unsigned long long seed =
      ((unsigned long long)params.frame * 9781ULL) ^
      ((unsigned long long)dst * 6271ULL) ^
      0x853c49e6748fea9bULL;

  for (int s = 0; s < params.spp; ++s) {
    PRD prd;
    prd.radiance = make3(0.0f);
    prd.throughput = make3(1.0f);
    prd.origin = org;
    prd.direction = sample_camera_dir(x, y, make_float2(rnd(seed), rnd(seed)));
    prd.depth = 0;
    prd.done = 0;
    prd.seed = seed;
    prd.prev_pdf_bsdf = 0.0f;
    prd.prev_pdf_valid = 0;

    while (!prd.done) {
      prd.radiance = make3(0.0f);
      float3 throughput = prd.throughput;
	  float throughput_ch = (ch==0 ? throughput.x : (ch==1 ? throughput.y : throughput.z));
      unsigned int u0, u1; packPtr(&prd, u0, u1);
      optixTrace(
        params.handle,
        prd.origin,
        prd.direction,
        0.0f, 1e16f, 0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,
        2,
        0,
        u0, u1);
      if (params.out_rgb)
        sum_rgb += prd.radiance * throughput;
      if (params.out_bayer)
        sum_bayer += (ch==0 ? prd.radiance.x : (ch==1 ? prd.radiance.y : prd.radiance.z)) * throughput_ch;
    }

    seed = prd.seed;
  }

  // Write outputs
  if (params.out_rgb) {
		float3 rad = sum_rgb / float(params.spp);
    float3* out = ptr_at<float3>(params.out_rgb);
    out[dst] = rad;
  }
  if (params.out_bayer) {
		float rad = fmaxf(sum_bayer / float(params.spp), 0.f);
    float* out = ptr_at<float>(params.out_bayer);
    out[dst] = rad;
  }
}
