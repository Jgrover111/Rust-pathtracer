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
  CUdeviceptr              out_raw16;    // uint16_t*
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

  // Geometry and material arrays
  CUdeviceptr              d_vertices;   // float3*
  CUdeviceptr              d_indices;    // uint3*
  CUdeviceptr              d_kd;         // float3*
  CUdeviceptr              d_ke;         // float3*
  uint32_t                 num_triangles;

  // Simple point light
  float3                   light_pos;
  float3                   light_emit;
};

extern "C" {
__constant__ Params params;
}

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
static __forceinline__ __device__ unsigned int lcg(unsigned int& state) {
  state = state * 1664525u + 1013904223u;
  return state;
}

static __forceinline__ __device__ float rnd(unsigned int& state) {
  return float(lcg(state) & 0x00FFFFFFu) / float(0x01000000u);
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
    int    depth;
    int    done;
};
struct RadiancePRD {
  float3 radiance;
};
struct PRDShadow {
    int occluded;
};

// --- access helpers for CUdeviceptr arrays --------------------------------

template<typename T>
static __forceinline__ __device__ T* ptr_at(CUdeviceptr p) {
  return reinterpret_cast<T*>(p);
}

// --- Miss programs ---------------------------------------------------------

extern "C" __global__ void __miss__ms_radiance()
{
  RadiancePRD* prd = unpackPtr<RadiancePRD>();
  prd->radiance = make3(0.0f);
}

extern "C" __global__ void __miss__ms_shadow()
{
  // 0 means not occluded; anyhit will set to 1
  optixSetPayload_0(0u);
}

// --- Anyhit for shadow rays: mark occluded and terminate -------------------

extern "C" __global__ void __anyhit__ah_shadow()
{
    PRDShadow& sprd = *unpackPtr<PRDShadow>();
    sprd.occluded = 1;
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
    const float3 Ng = normalize(cross(v1 - v0, v2 - v0));

    // Fetch per-triangle materials from Params
    const float3* kd_arr = reinterpret_cast<const float3*>(params.d_kd);
    const float3* ke_arr = reinterpret_cast<const float3*>(params.d_ke);
    const float3 kd = kd_arr[prim];
    const float3 ke = ke_arr[prim];

    // Direct light
    const float3 Lpos = params.light_pos;
    float3 L = Lpos - P;
    const float dist = fmaxf(length(L), 1e-3f);
    const float3 wi = L / dist;

    // Shadow ray
    PRDShadow sprd;
    sprd.occluded = 0;
    unsigned int u0, u1;
    packPtr(&sprd, u0, u1);

    optixTrace(
        params.handle,
        /*ray origin*/ P + Ng * 1e-3f,
        /*ray dir*/    wi,
        /*tmin*/       0.0f,
        /*tmax*/       dist - 1e-3f,
        /*time*/       0.0f,
        /*visibility mask*/ 1,
        /*ray flags*/  OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        /*SBT offsets*/
        RAY_TYPE_SHADOW,      // SBT offset
        RAY_TYPE_COUNT,       // SBT stride
        RAY_TYPE_SHADOW,      // missSBTIndex
        /*payload*/ u0, u1
    );

    const float visibility = sprd.occluded ? 0.0f : 1.0f;
    const float nDotL = fmaxf(0.0f, dot(Ng, wi));

    float3 Lo = kd * params.light_emit * (visibility * nDotL);
    // add emission if you use it
    Lo += ke;

    prd.radiance = Lo;
    prd.done = true;
}

// --- Raygen program --------------------------------------------------------

static __forceinline__ __device__ uint16_t to_u16(float x) {
  x = clamp01(x);
  return (uint16_t)(x * 65535.0f + 0.5f);
}

static __forceinline__ __device__ float3 sample_camera_dir(int x, int y, const float2& jitter)
{
  const uint3  dim = optixGetLaunchDimensions();
  const float  fx  = (float(x) + jitter.x) / float(dim.x);
  const float  fy  = (float(y) + jitter.y) / float(dim.y);
  const float2 d   = make_float2(2.0f*fx - 1.0f, 1.0f - 2.0f*fy);
  return normalize3(params.cam_w + d.x * params.cam_u + d.y * params.cam_v);
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
  float3 sum = make3(0.0f);
  unsigned int seed = params.frame * 9781u + dst * 6271u;

  for (int s = 0; s < params.spp; ++s) {
    RadiancePRD prd;
    prd.radiance = make3(0.0f);
    unsigned int u0, u1;
    packPtr(&prd, u0, u1);

    float2 jitter = make_float2(rnd(seed), rnd(seed));
    const float3 dir = sample_camera_dir(x, y, jitter);

    optixTrace(
      params.handle,
      org, dir,
      0.0f, 1e16f, 0.0f,
      OptixVisibilityMask(1),
      OPTIX_RAY_FLAG_DISABLE_ANYHIT, // anyhit only used by shadow ray
      /* SBT record offsets: */
      0,  // RAY_RADIANCE
      2,  // RAY_TYPE_COUNT
      0,  // missSBTIndex = RAY_RADIANCE
      u0, u1
    );

    sum += prd.radiance;
  }

  const float3 rad = sum / float(params.spp);

  // Write outputs
  if (params.out_rgb) {
    float3* out = ptr_at<float3>(params.out_rgb);
    out[dst] = rad;
  }
  if (params.out_raw16) {
    unsigned short* out = ptr_at<unsigned short>(params.out_raw16);

    // Bayer mapping
    const int pat = params.bayer_pattern; // 0:RGGB,1:BGGR,2:GRBG,3:GBRG
    const bool xOdd = (x & 1) != 0;
    const bool yOdd = (y & 1) != 0;

    float val = 0.0f;
    if (pat == 0) {            // RGGB
      val = (!xOdd && !yOdd) ? rad.x : (xOdd && yOdd) ? rad.z : rad.y;
    } else if (pat == 1) {     // BGGR
      val = (!xOdd && !yOdd) ? rad.z : (xOdd && yOdd) ? rad.x : rad.y;
    } else if (pat == 2) {     // GRBG
      val = (!xOdd && !yOdd) ? rad.y : (xOdd && yOdd) ? rad.y : (xOdd ? rad.x : rad.z);
    } else {                   // GBRG
      val = (!xOdd && !yOdd) ? rad.y : (xOdd && yOdd) ? rad.y : (xOdd ? rad.z : rad.x);
    }
    out[dst] = to_u16(val);
  }
}
