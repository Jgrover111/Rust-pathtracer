#include <optix.h>
#include <optix_device.h>
#include <stdint.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <math_constants.h>

#define GUIDE_PHI_RES 16
#define GUIDE_THETA_RES 8
#define GUIDE_BIN_COUNT (GUIDE_PHI_RES * GUIDE_THETA_RES)

// Precomputed constants to avoid repeated division at runtime
#define INV_UINT16       (1.0f / 65536.0f)
#define INV_PLASTIC      (1.0f / 1.32471795724474602596f)
#define INV_PLASTIC_SQR  (1.0f / (1.32471795724474602596f * 1.32471795724474602596f))

// Keep this in sync with optix_wrapper.cpp
struct Params
{
  CUdeviceptr              out_rgb;      // float3*
  CUdeviceptr              out_bayer;    // float*
  CUdeviceptr              noise_map;    // float*
  CUdeviceptr              flags;       // uint32_t*
  CUdeviceptr              active_pixels; // uint32_t*
  CUdeviceptr              sample_counts; // int*
  int                      width;
  int                      height;
  int                      spp;         // samples per launch
  int                      max_spp;     // total samples per pixel limit
  int                      max_depth;
  int                      frame;
  int                      bayer_pattern; // 0:RGGB,1:BGGR,2:GRBG,3:GBRG
  float                    noise_threshold;
  int                      min_adaptive_samples;
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
  CUdeviceptr              d_guiding;
};

extern "C" {
__constant__ Params params;
}

// SBT data for each primitive
struct HitgroupData
{
  float3 Base_colour;
  float  Metallic;
  float  Roughness;
  float  IOR;
  float  Alpha;
  float  Transmission;
  float3 Emission;
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
static __forceinline __device__ float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
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

static __forceinline__ __device__ float select(const float3& v, int ch) {
  // Branchless channel selection via pointer arithmetic
  return (&v.x)[ch];
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

// Generate spatiotemporal blue noise sample using an R2 sequence with a
// per-pixel Cranley-Patterson rotation. The hash introduces a high frequency
// offset so neighbouring pixels are decorrelated while preserving the low
// discrepancy of the base sequence.
static __forceinline__ __device__ unsigned int hash_u32(unsigned int x)
{
  x ^= x >> 17;
  x *= 0xed5ad4bbU;
  x ^= x >> 11;
  x *= 0xac4c1b51U;
  x ^= x >> 15;
  x *= 0x31848babU;
  x ^= x >> 14;
  return x;
}

static __forceinline__ __device__ float2 blue_noise(int x, int y, int s, int frame)
{
  unsigned int h = hash_u32(x * 1973u + y * 9277u + frame * 26699u);
  float rx = (h & 0xffff) * INV_UINT16;
  float ry = ((h >> 16) & 0xffff) * INV_UINT16;
  float fs = s + 0.5f;
  float u = fmodf(rx + fs * INV_PLASTIC, 1.0f);
  float v = fmodf(ry + fs * INV_PLASTIC_SQR, 1.0f);
  return make_float2(u, v);
}

static __forceinline__ __device__ float sample_tri(float u)
{
  return (u < 0.5f) ? sqrtf(2.0f * u) - 1.0f : 1.0f - sqrtf(2.0f * (1.0f - u));
}

static __forceinline__ __device__ float2 olpf_jitter(int x, int y, int s, int frame)
{
  float2 u = blue_noise(x, y, s, frame);
  return make_float2(sample_tri(u.x), sample_tri(u.y));
}

template<typename T>
static __forceinline__ __device__ float next_rand(T& prd)
{
  if (prd.rng_dim < 2) {
    float2 u = blue_noise(prd.px, prd.py, prd.sample, params.frame);
    float r = prd.rng_dim == 0 ? u.x : u.y;
    prd.rng_dim++;
    return r;
  }
  prd.rng_dim++;
  return rnd(prd.seed);
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

// Per-ray data carried through the path tracer.
// Must be defined before any functions (like sample_guided_dir)
// that access its members.
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
    int    px;
    int    py;
    int    sample;
    int    rng_dim;
};

struct PRDScalar {
    float  radiance;
    float  throughput;
    float3 origin;
    float3 direction;
    unsigned long long seed;
    int    depth;
    int    done;
    float  prev_pdf_bsdf;
    int    prev_pdf_valid;
    int    ch;
    int    px;
    int    py;
    int    sample;
    int    rng_dim;
};

static __forceinline__ __device__ float luminance(const float3& c)
{
    return c.x * 0.2126f + c.y * 0.7152f + c.z * 0.0722f;
}

static __forceinline__ __device__ int guiding_bin(const float3& dir, const float3& Ng)
{
    float3 tangent, bitangent;
    make_onb(Ng, tangent, bitangent);
    float3 local = make_float3(dot(dir, tangent), dot(dir, bitangent), dot(dir, Ng));
    if (local.z <= 0.f) return -1;
    float phi = atan2f(local.y, local.x);
    if (phi < 0.f) phi += 2.f * CUDART_PI_F;
    int phi_idx = min(max(int(phi / (2.f * CUDART_PI_F) * GUIDE_PHI_RES), 0), GUIDE_PHI_RES - 1);
    int mu_idx = min(max(int(local.z * GUIDE_THETA_RES), 0), GUIDE_THETA_RES - 1);
    return mu_idx * GUIDE_PHI_RES + phi_idx;
}

static __forceinline__ __device__ float3 sample_guided_dir(PRD& prd, const float3& Ng, float& pdf, float& cosTheta)
{
    float* table = reinterpret_cast<float*>(params.d_guiding);
    float sum = 0.f;
    for (int i = 0; i < GUIDE_BIN_COUNT; ++i) sum += table[i];
    if (sum <= 0.f) {
        float r1 = next_rand(prd);
        float r2 = next_rand(prd);
        const float phi = 2.0f * CUDART_PI_F * r1;
        cosTheta = sqrtf(1.0f - r2);
        const float sinTheta = sqrtf(r2);
        float3 localDir = make_float3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
        float3 t, b;
        make_onb(Ng, t, b);
        float3 newDir = normalize(localDir.x * t + localDir.y * b + localDir.z * Ng);
        pdf = cosTheta * (1.0f / CUDART_PI_F);
        return newDir;
    }
    float r = next_rand(prd) * sum;
    float acc = 0.f;
    int idx = GUIDE_BIN_COUNT - 1;
    for (int i = 0; i < GUIDE_BIN_COUNT; ++i) {
        acc += table[i];
        if (r <= acc) { idx = i; break; }
    }
    int phi_idx = idx % GUIDE_PHI_RES;
    int mu_idx  = idx / GUIDE_PHI_RES;
    float phi = (phi_idx + next_rand(prd)) / float(GUIDE_PHI_RES) * 2.f * CUDART_PI_F;
    float mu  = (mu_idx + next_rand(prd)) / float(GUIDE_THETA_RES);
    cosTheta = mu;
    float sinTheta = sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));
    float3 localDir = make_float3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
    float3 t, b;
    make_onb(Ng, t, b);
    float3 newDir = normalize(localDir.x * t + localDir.y * b + localDir.z * Ng);
    float bin_mass = table[idx] / sum;
    float bin_area = (2.f * CUDART_PI_F / GUIDE_PHI_RES) * (1.f / GUIDE_THETA_RES);
    pdf = bin_mass / bin_area;
    return newDir;
}

static __forceinline__ __device__ float guiding_pdf(const float3& dir, const float3& Ng)
{
    if (params.d_guiding == 0) return 0.f;
    int idx = guiding_bin(dir, Ng);
    if (idx < 0) return 0.f;
    float* table = reinterpret_cast<float*>(params.d_guiding);
    float sum = 0.f;
    for (int i = 0; i < GUIDE_BIN_COUNT; ++i) sum += table[i];
    if (sum <= 0.f) return 0.f;
    float bin_mass = table[idx] / sum;
    float bin_area = (2.f * CUDART_PI_F / GUIDE_PHI_RES) * (1.f / GUIDE_THETA_RES);
    return bin_mass / bin_area;
}

static __forceinline__ __device__ float guiding_strength()
{
    if (params.d_guiding == 0) return 0.f;
    float* table = reinterpret_cast<float*>(params.d_guiding);
    float sum = 0.f;
    for (int i = 0; i < GUIDE_BIN_COUNT; ++i) sum += table[i];
    return sum / (sum + float(GUIDE_BIN_COUNT));
}

static __forceinline__ __device__ void update_guiding(const float3& dir, const float3& Ng, float weight)
{
    if (params.d_guiding == 0) return;
    int idx = guiding_bin(dir, Ng);
    if (idx < 0) return;
    float* table = reinterpret_cast<float*>(params.d_guiding);
    atomicAdd(&table[idx], weight);
}

static __forceinline__ __device__ float3 reflect(const float3& i, const float3& n)
{
  return i - 2.0f * dot(i, n) * n;
}

static __forceinline__ __device__ bool refract(const float3& i, const float3& n, float eta, float3& t)
{
  float cosi = dot(-n, i);
  float k = 1.0f - eta * eta * (1.0f - cosi * cosi);
  if (k < 0.0f) return false;
  t = eta * i + (eta * cosi - sqrtf(k)) * n;
  return true;
}

static __forceinline__ __device__ float3 fresnel_schlick(float cosTheta, const float3& F0)
{
  return F0 + (make3(1.0f) - F0) * powf(fmaxf(0.0f, 1.0f - cosTheta), 5.0f);
}

// Sample the GGX microfacet distribution using the Visible Normal Distribution
// Function (VNDF) technique. This method better matches the distribution of
// visible microfacets for the given view direction, reducing variance compared
// to sampling the full NDF. See "Sampling the GGX Distribution of Visible
// Normals" by Heitz (2018).
static __forceinline__ __device__ void sample_ggx_vndf(
    float2 u, float alpha, const float3& N, const float3& V, float3& L, float3& H, float& pdf)
{
  // Build an orthonormal basis and transform the view direction to local space
  float3 T, B;
  make_onb(N, T, B);
  float3 Vlocal = make_float3(dot(V, T), dot(V, B), dot(V, N));

  // Stretch view direction by the roughness parameter (Heitz 2018)
  float3 Vh = normalize(make_float3(alpha * Vlocal.x, alpha * Vlocal.y, Vlocal.z));

  // Orthonormal basis around the stretched view direction
  float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
  float3 T1 = lensq > 0.0f ? make_float3(-Vh.y, Vh.x, 0.0f) / sqrtf(lensq)
                           : make_float3(1.0f, 0.0f, 0.0f);
  float3 T2 = cross(Vh, T1);

  // Sample a point on a disk (polar coordinates)
  float r = sqrtf(u.x);
  float phi = 2.0f * CUDART_PI_F * u.y;
  float t1 = r * cosf(phi);
  float t2 = r * sinf(phi);

  // Adjust t2 based on view direction to obtain a visible normal
  float s = 0.5f * (1.0f + Vh.z);
  t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + s * t2;

  // Reproject onto hemisphere and unstretch
  float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
  float3 Hlocal = normalize(make_float3(alpha * Nh.x, alpha * Nh.y, fmaxf(0.0f, Nh.z)));

  // Transform the half-vector back to world space
  H = normalize(Hlocal.x * T + Hlocal.y * B + Hlocal.z * N);

  // Compute the reflected direction
  L = normalize(reflect(-V, H));

  // Compute PDF for the sampled direction. For VNDF sampling, this is
  // D(h) * G1(v) * (n·h) / (4 * (v·h))
  float NoH = fmaxf(dot(N, H), 0.0f);
  float NoV = fmaxf(dot(N, V), 0.0f);
  float VoH = fmaxf(dot(V, H), 0.0f);
  float alpha2 = alpha * alpha;
  float denom = NoH * NoH * (alpha2 - 1.0f) + 1.0f;
  float D = alpha2 / (CUDART_PI_F * denom * denom);
  float G1V = 2.0f * NoV / (NoV + sqrtf(alpha2 + (1.0f - alpha2) * NoV * NoV));
  pdf = (D * NoH * G1V) / (4.0f * fmaxf(VoH, 1e-6f));
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

// --- access helpers for CUdeviceptr arrays --------------------------------

template<typename T>
static __forceinline__ __device__ T* ptr_at(CUdeviceptr p) {
  return reinterpret_cast<T*>(p);
}

static __forceinline__ __device__ float atomicMaxf(float* addr, float val) {
  int* addr_as_i = reinterpret_cast<int*>(addr);
  int old = *addr_as_i, assumed;
  if (__int_as_float(old) >= val) return __int_as_float(old);
  do {
    assumed = old;
    old = atomicCAS(addr_as_i, assumed, __float_as_int(val));
  } while (__int_as_float(old) < val);
  return __int_as_float(old);
}

// --- Miss programs ---------------------------------------------------------

extern "C" __global__ void __miss__ms_radiance()
{
  PRD* prd = unpackPtr<PRD>();
  prd->radiance = make3(0.0f);
  prd->done = 1;
}

extern "C" __global__ void __miss__ms_radiance_scalar()
{
  PRDScalar* prd = unpackPtr<PRDScalar>();
  prd->radiance = 0.0f;
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
    const float3 Base_colour = hg->Base_colour;
    const float Roughness = hg->Roughness;
    const float IOR = hg->IOR;
    const float3 Emission = hg->Emission;
    const float Transmission = hg->Transmission;

    if (Transmission > 0.0f) {
        float3 N = dot(prd.direction, Ng) < 0.0f ? Ng : -Ng;
        float eta = dot(prd.direction, Ng) < 0.0f ? 1.0f / IOR : IOR;
        float3 I = prd.direction;
        float cosi = fmaxf(dot(-N, I), 0.0f);
        float f0 = (IOR - 1.0f) / (IOR + 1.0f);
        f0 = f0 * f0;
        float Fr = f0 + (1.0f - f0) * powf(1.0f - cosi, 5.0f);
        if (next_rand(prd) < Fr) {
            prd.origin = P + N * 1e-3f;
            prd.direction = reflect(I, N);
        } else {
            float3 T;
            if (!refract(I, N, eta, T)) {
                prd.origin = P + N * 1e-3f;
                prd.direction = reflect(I, N);
            } else {
                prd.origin = P - N * 1e-3f;
                prd.direction = normalize3(T);
                prd.throughput = mul(prd.throughput, Base_colour);
            }
        }
        prd.radiance += Emission;
        prd.prev_pdf_valid = 0;
        prd.prev_pdf_bsdf = 0.0f;
        prd.depth++;
        if (prd.depth >= params.max_depth) {
            prd.done = 1;
            return;
        }
        if (prd.depth >= params.max_depth - 2) {
            float p = fmaxf(prd.throughput.x, fmaxf(prd.throughput.y, prd.throughput.z));
            if (next_rand(prd) > p) {
                prd.done = 1;
                return;
            }
            prd.throughput = prd.throughput / fmaxf(p, 1e-3f);
        }
        return;
    }

    float f0 = (IOR - 1.0f) / (IOR + 1.0f);
    f0 = f0 * f0;
    float3 F0 = make3(f0);
    float3 V = -prd.direction;
    float NoV = fmaxf(dot(Ng, V), 0.0f);
    float3 F = fresnel_schlick(NoV, F0);
    float spec_prob = fminf(0.99f, fmaxf(0.01f, (F.x + F.y + F.z) * (1.0f / 3.0f)));
    float diff_prob = 1.0f - spec_prob;
    float3 kd = Base_colour * (make3(1.0f) - F);

    // Sample direct illumination from rectangular area light
    float3 Lo = make3(0.0f);
    {
        float u = (next_rand(prd) * 2.0f - 1.0f) * params.light_half.x;
        float v = (next_rand(prd) * 2.0f - 1.0f) * params.light_half.y;
        float3 lp = make_float3(params.light_pos.x + u, params.light_pos.y + v, params.light_pos.z);
        float3 L = lp - P;
        float dist2 = fmaxf(dot(L, L), 1e-6f);
        float dist = sqrtf(dist2);
        float3 wi = L / dist;
        float cosS = fmaxf(0.0f, dot(Ng, wi));
        float cosL = fmaxf(0.0f, dot(params.light_normal, -wi));
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
                float3 V = -prd.direction;
                float3 H = normalize3(V + wi);
                float VoH = fmaxf(dot(V, H), 0.0f);
                float NoH = fmaxf(dot(Ng, H), 0.0f);
                float alpha = fmaxf(Roughness * Roughness, 1e-4f);
                float alpha2 = alpha * alpha;
                float denom = NoH * NoH * (alpha2 - 1.0f) + 1.0f;
                float D = alpha2 / (CUDART_PI_F * denom * denom);
                float G1V = 2.0f * NoV /
                             (NoV + sqrtf(alpha2 + (1.0f - alpha2) * NoV * NoV));
                float G1L = 2.0f * cosS /
                             (cosS + sqrtf(alpha2 + (1.0f - alpha2) * cosS * cosS));
                float3 F = fresnel_schlick(VoH, F0);
                float3 f_spec = F * (D * G1V * G1L / fmaxf(4.0f * NoV * cosS, 1e-6f));
                float pdf_spec = D * NoH * G1V / fmaxf(4.0f * VoH, 1e-6f) * spec_prob;
                float pdf_diff = cosS * (1.0f / CUDART_PI_F) * diff_prob;
                float pdf_bsdf = pdf_spec + pdf_diff;
                float area = 4.0f * params.light_half.x * params.light_half.y;
                float pdf_area = 1.0f / area;
                float pdf_light = pdf_area * dist2 / cosL;
                float w = pdf_light / (pdf_light + pdf_bsdf);
                float3 f = kd * (1.0f / CUDART_PI_F) + f_spec;
                float3 contrib = params.light_emit * f * (cosS * cosL / dist2) / pdf_area;
                Lo += contrib * w;
            }
        }
    }

    // Accumulate emission with MIS and direct light
    float3 emission = Emission;
    if (prd.prev_pdf_valid && (Emission.x > 0.0f || Emission.y > 0.0f || Emission.z > 0.0f)) {
        float3 L = P - prd.origin;
        float dist2 = fmaxf(dot(L, L), 1e-6f);
        float area = 4.0f * params.light_half.x * params.light_half.y;
        float cosL = fmaxf(0.0f, dot(params.light_normal, -prd.direction));
        float pdf_light = dist2 / (cosL * area);
        float w_bsdf = prd.prev_pdf_bsdf / (prd.prev_pdf_bsdf + pdf_light);
        emission = emission * w_bsdf;
    }
    prd.radiance += emission + Lo;

    float choose = next_rand(prd);
    if (choose < spec_prob) {
        float2 u = make_float2(next_rand(prd), next_rand(prd));
        float3 V = -prd.direction;
        float3 newDir;
        float3 H;
        float pdf;
        float alpha = fmaxf(Roughness * Roughness, 1e-4f);
        sample_ggx_vndf(u, alpha, Ng, V, newDir, H, pdf);
        float NoL = fmaxf(dot(Ng, newDir), 0.0f);
        float NoV = fmaxf(dot(Ng, V), 0.0f);
        float VoH = fmaxf(dot(V, H), 0.0f);
        float NoH = fmaxf(dot(Ng, H), 0.0f);
        float alpha2 = alpha * alpha;
        float denom = NoH * NoH * (alpha2 - 1.0f) + 1.0f;
        float D = alpha2 / (CUDART_PI_F * denom * denom);
        float G1V = 2.0f * NoV /
                     (NoV + sqrtf(alpha2 + (1.0f - alpha2) * NoV * NoV));
        float G1L = 2.0f * NoL /
                    (NoL + sqrtf(alpha2 + (1.0f - alpha2) * NoL * NoL));
        float3 F = fresnel_schlick(VoH, F0);
        float3 spec_brdf = F * (D * G1V * G1L / fmaxf(4.0f * NoV * NoL, 1e-6f));
        prd.origin = P + Ng * 1e-3f;
        prd.direction = newDir;
        prd.throughput = mul(prd.throughput, spec_brdf * NoL / fmaxf(pdf * spec_prob, 1e-6f));
        prd.prev_pdf_bsdf = pdf * spec_prob;
        prd.prev_pdf_valid = 1;
    } else {
        float3 newDir;
        float cosTheta;
        float pdf_guided = 0.f;
        float guideWeight = guiding_strength();
        bool guided = (guideWeight > 0.f) && (next_rand(prd) < guideWeight);
        if (guided) {
            newDir = sample_guided_dir(prd, Ng, pdf_guided, cosTheta);
        } else {
            float r1 = next_rand(prd);
            float r2 = next_rand(prd);
            const float phi = 2.0f * CUDART_PI_F * r1;
            cosTheta = sqrtf(1.0f - r2);
            const float sinTheta = sqrtf(r2);
            float3 localDir = make_float3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
            float3 tangent, bitangent;
            make_onb(Ng, tangent, bitangent);
            newDir = normalize(localDir.x * tangent + localDir.y * bitangent + localDir.z * Ng);
            pdf_guided = guiding_pdf(newDir, Ng);
        }

        prd.origin = P + Ng * 1e-3f;
        prd.direction = newDir;

        float pdf_uniform = cosTheta * (1.0f / CUDART_PI_F);
        float final_pdf = guideWeight * pdf_guided + (1.0f - guideWeight) * pdf_uniform;
        prd.throughput = mul(prd.throughput, kd);
        prd.throughput = prd.throughput * (cosTheta / (CUDART_PI_F * final_pdf * diff_prob));
        prd.prev_pdf_bsdf = final_pdf * diff_prob;
        prd.prev_pdf_valid = 1;
        update_guiding(newDir, Ng, luminance(prd.throughput));
    }

    prd.depth++;
    if (prd.depth >= params.max_depth) {
        prd.done = 1;
        return;
    }
    if (prd.depth >= params.max_depth - 2) {
        float p = fmaxf(prd.throughput.x, fmaxf(prd.throughput.y, prd.throughput.z));
        if (next_rand(prd) > p) {
            prd.done = 1;
            return;
        }
        prd.throughput = prd.throughput / fmaxf(p, 1e-3f);
    }
}

extern "C" __global__ void __closesthit__ch_bayer()
{
    PRDScalar& prd = *unpackPtr<PRDScalar>();

    const unsigned int prim = optixGetPrimitiveIndex();

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

    const HitgroupData* hg = reinterpret_cast<const HitgroupData*>(optixGetSbtDataPointer());
    const float3 Base_colour = hg->Base_colour;
    const float Roughness = hg->Roughness;
    const float IOR = hg->IOR;
    const float3 Emission = hg->Emission;
    const float Transmission = hg->Transmission;

    const int ch = prd.ch;
    float albedo = select(Base_colour, ch);
    float emission = select(Emission, ch);

    if (Transmission > 0.0f) {
        float3 N = dot(prd.direction, Ng) < 0.0f ? Ng : -Ng;
        float eta = dot(prd.direction, Ng) < 0.0f ? 1.0f / IOR : IOR;
        float3 I = prd.direction;
        float cosi = fmaxf(dot(-N, I), 0.0f);
        float f0 = (IOR - 1.0f) / (IOR + 1.0f);
        f0 = f0 * f0;
        float Fr = f0 + (1.0f - f0) * powf(1.0f - cosi, 5.0f);
        if (next_rand(prd) < Fr) {
            prd.origin = P + N * 1e-3f;
            prd.direction = reflect(I, N);
        } else {
            float3 T;
            if (!refract(I, N, eta, T)) {
                prd.origin = P + N * 1e-3f;
                prd.direction = reflect(I, N);
            } else {
                prd.origin = P - N * 1e-3f;
                prd.direction = normalize3(T);
                prd.throughput *= albedo;
            }
        }
        prd.radiance += emission;
        prd.prev_pdf_valid = 0;
        prd.prev_pdf_bsdf = 0.0f;
        prd.depth++;
        if (prd.depth >= params.max_depth) {
            prd.done = 1;
            return;
        }
        if (prd.depth >= params.max_depth - 2) {
            float p = fminf(0.95f, fmaxf(0.05f, prd.throughput));
            if (next_rand(prd) > p) {
                prd.done = 1;
                return;
            }
            prd.throughput *= (1.0f / p);
        }
        return;
    }

    float f0 = (IOR - 1.0f) / (IOR + 1.0f);
    f0 = f0 * f0;
    float3 V = -prd.direction;
    float NoV = fmaxf(dot(Ng, V), 0.0f);
    float Fr = f0 + (1.0f - f0) * powf(fmaxf(0.0f, 1.0f - NoV), 5.0f);
    float spec_prob = fminf(0.99f, fmaxf(0.01f, Fr));
    float diff_prob = 1.0f - spec_prob;
    float kd = albedo * (1.0f - Fr);

    float Lo = 0.0f;
    {
        float u = (next_rand(prd) * 2.0f - 1.0f) * params.light_half.x;
        float v = (next_rand(prd) * 2.0f - 1.0f) * params.light_half.y;
        float3 lp = make_float3(params.light_pos.x + u, params.light_pos.y + v, params.light_pos.z);
        float3 L = lp - P;
        float dist2 = fmaxf(dot(L, L), 1e-6f);
        float dist = sqrtf(dist2);
        float3 wi = L / dist;
        float cosS = fmaxf(0.0f, dot(Ng, wi));
        float cosL = fmaxf(0.0f, dot(params.light_normal, -wi));
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
                float3 Vv = -prd.direction;
                float3 H = normalize3(Vv + wi);
                float VoH = fmaxf(dot(Vv, H), 0.0f);
                float NoH = fmaxf(dot(Ng, H), 0.0f);
                float alpha = fmaxf(Roughness * Roughness, 1e-4f);
                float alpha2 = alpha * alpha;
                float denom = NoH * NoH * (alpha2 - 1.0f) + 1.0f;
                float D = alpha2 / (CUDART_PI_F * denom * denom);
                float G1V = 2.0f * NoV /
                             (NoV + sqrtf(alpha2 + (1.0f - alpha2) * NoV * NoV));
                float G1L = 2.0f * cosS /
                             (cosS + sqrtf(alpha2 + (1.0f - alpha2) * cosS * cosS));
                float F = f0 + (1.0f - f0) * powf(fmaxf(0.0f, 1.0f - VoH), 5.0f);
                float f_spec = F * (D * G1V * G1L / fmaxf(4.0f * NoV * cosS, 1e-6f));
                float pdf_spec = D * NoH * G1V / fmaxf(4.0f * VoH, 1e-6f) * spec_prob;
                float pdf_diff = cosS * (1.0f / CUDART_PI_F) * diff_prob;
                float pdf_bsdf = pdf_spec + pdf_diff;
                float area = 4.0f * params.light_half.x * params.light_half.y;
                float pdf_area = 1.0f / area;
                float pdf_light = pdf_area * dist2 / cosL;
                float w = pdf_light / (pdf_light + pdf_bsdf);
                float f = kd * (1.0f / CUDART_PI_F) + f_spec;
                float Le = select(params.light_emit, ch);
                float contrib = Le * f * (cosS * cosL / dist2) / pdf_area;
                Lo += contrib * w;
            }
        }
    }

    if (prd.prev_pdf_valid && emission > 0.0f) {
        float3 L = P - prd.origin;
        float dist2 = fmaxf(dot(L, L), 1e-6f);
        float area = 4.0f * params.light_half.x * params.light_half.y;
        float cosL = fmaxf(0.0f, dot(params.light_normal, -prd.direction));
        float pdf_light = dist2 / (cosL * area);
        float w_bsdf = prd.prev_pdf_bsdf / (prd.prev_pdf_bsdf + pdf_light);
        emission *= w_bsdf;
    }
    prd.radiance += emission + Lo;

    float choose = next_rand(prd);
    if (choose < spec_prob) {
        float2 u = make_float2(next_rand(prd), next_rand(prd));
        float3 V = -prd.direction;
        float3 newDir;
        float3 H;
        float pdf;
        float alpha = fmaxf(Roughness * Roughness, 1e-4f);
        sample_ggx_vndf(u, alpha, Ng, V, newDir, H, pdf);
        float NoL = fmaxf(dot(Ng, newDir), 0.0f);
        float NoV = fmaxf(dot(Ng, V), 0.0f);
        float VoH = fmaxf(dot(V, H), 0.0f);
        float NoH = fmaxf(dot(Ng, H), 0.0f);
        float alpha2 = alpha * alpha;
        float denom = NoH * NoH * (alpha2 - 1.0f) + 1.0f;
        float D = alpha2 / (CUDART_PI_F * denom * denom);
        float G1V = 2.0f * NoV /
                     (NoV + sqrtf(alpha2 + (1.0f - alpha2) * NoV * NoV));
        float G1L = 2.0f * NoL /
                    (NoL + sqrtf(alpha2 + (1.0f - alpha2) * NoL * NoL));
        float F = f0 + (1.0f - f0) * powf(fmaxf(0.0f, 1.0f - VoH), 5.0f);
        float spec_brdf = F * (D * G1V * G1L / fmaxf(4.0f * NoV * NoL, 1e-6f));
        prd.origin = P + Ng * 1e-3f;
        prd.direction = newDir;
        prd.throughput *= spec_brdf * NoL / fmaxf(pdf * spec_prob, 1e-6f);
        prd.prev_pdf_bsdf = pdf * spec_prob;
        prd.prev_pdf_valid = 1;
        prd.prev_pdf_bsdf = pdf * spec_prob;
        prd.prev_pdf_valid = 1;
    } else {
        float r1 = next_rand(prd);
        float r2 = next_rand(prd);
        const float phi = 2.0f * CUDART_PI_F * r1;
        const float cosTheta = sqrtf(1.0f - r2);
        const float sinTheta = sqrtf(r2);
        float3 localDir = make_float3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);

        float3 tangent, bitangent;
        make_onb(Ng, tangent, bitangent);
        float3 newDir = normalize(localDir.x * tangent + localDir.y * bitangent + localDir.z * Ng);

        prd.origin = P + Ng * 1e-3f;
        prd.direction = newDir;
        prd.throughput *= kd / diff_prob;
        prd.prev_pdf_bsdf = cosTheta * (1.0f / CUDART_PI_F) * diff_prob;
        prd.prev_pdf_valid = 1;
    }

    prd.depth++;
    if (prd.depth >= params.max_depth) {
        prd.done = 1;
        return;
    }
    if (prd.depth >= params.max_depth - 2) {
        float p = fminf(0.95f, fmaxf(0.05f, prd.throughput));
        if (next_rand(prd) > p) {
            prd.done = 1;
            return;
        }
        prd.throughput *= (1.0f / p);
    }
}

// --- Raygen program --------------------------------------------------------

static __forceinline__ __device__ float3 sample_camera_dir(int x, int y, const float2& jitter)
{
  // Use the image dimensions from params rather than the launch dimensions so
  // that ray generation works with 1D launches over compacted pixel lists.
  float fx = (float(x) + 0.5f + jitter.x) / float(params.width);
  float fy = (float(y) + 0.5f + jitter.y) / float(params.height);
  fx = clamp01(fx);
  fy = clamp01(fy);
  const float2 d = make_float2(2.0f * fx - 1.0f, 1.0f - 2.0f * fy);
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
  const int    launch = int(idx.x);
  const int    W   = params.width;
  const int    H   = params.height;
  int dst = launch;
  if (params.active_pixels)
    dst = ptr_at<uint32_t>(params.active_pixels)[launch];
  const int    x   = dst % W;
  const int    y   = dst / W;

  // Trace radiance
  const float3 org = params.cam_eye;
  // Track running means and variances using Welford's algorithm.
  // We maintain RGB statistics even if only a Bayer image is requested so that
  // adaptive sampling decisions take all colour channels into account.
  // Track RGB variance even if we are not writing an RGB image to ensure
  // adaptive sampling decisions consider all colour channels.
  const bool write_rgb   = params.out_rgb   != 0;
  const bool write_bayer = params.out_bayer != 0;

  float3 mean_rgb = make3(0.0f);
  if (write_rgb)
    mean_rgb = ptr_at<float3>(params.out_rgb)[dst];
  float3 m2_rgb = make3(0.0f);
  float mean_bayer = 0.0f;
  if (write_bayer)
    mean_bayer = ptr_at<float>(params.out_bayer)[dst];
  float m2_bayer = 0.0f;
  const int ch = bayer_channel_for(x, y, params.bayer_pattern);
  unsigned long long seed =
      ((unsigned long long)params.frame * 9781ULL) ^
      ((unsigned long long)dst * 6271ULL) ^
      0x853c49e6748fea9bULL;

  int* sc = ptr_at<int>(params.sample_counts);
  int samples = sc[dst];
  float max_noise = params.noise_threshold;
  for (int s = 0; s < params.spp && samples < params.max_spp; ++s) {
    samples++;
    PRD prd;
    prd.radiance = make3(0.0f);
    prd.throughput = make3(1.0f);
    prd.origin = org;
    float2 jitter = olpf_jitter(x, y, s, params.frame);
    prd.direction = sample_camera_dir(x, y, jitter);
    prd.depth = 0;
    prd.done = 0;
    prd.seed = seed;
    prd.prev_pdf_bsdf = 0.0f;
    prd.prev_pdf_valid = 0;
    prd.px = x;
    prd.py = y;
    prd.sample = s;
    prd.rng_dim = 0;
    float3 sample_rgb = make3(0.0f);
    float sample_bayer = 0.0f;

    while (!prd.done) {
      prd.radiance = make3(0.0f);
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
      // Always accumulate RGB contributions for variance estimation
      float3 contrib_rgb = prd.radiance * prd.throughput;
      sample_rgb += contrib_rgb;
      if (write_bayer) {
        float throughput_ch =
            (ch == 0 ? prd.throughput.x : (ch == 1 ? prd.throughput.y : prd.throughput.z));
        float contrib =
            (ch == 0 ? prd.radiance.x : (ch == 1 ? prd.radiance.y : prd.radiance.z)) *
            throughput_ch;
        sample_bayer += contrib;
      }
    }

    float3 delta_rgb = sample_rgb - mean_rgb;
    mean_rgb += delta_rgb / float(samples);
    float3 delta2_rgb = sample_rgb - mean_rgb;
    m2_rgb += delta_rgb * delta2_rgb;
    if (write_bayer) {
      float delta_bayer = sample_bayer - mean_bayer;
      mean_bayer += delta_bayer / float(samples);
      float delta2_bayer = sample_bayer - mean_bayer;
      m2_bayer += delta_bayer * delta2_bayer;
    }

    // Adaptive sampling that also considers neighbouring pixel variance
    if (params.noise_threshold > 0.f && samples >= params.min_adaptive_samples) {
      float3 var_rgb = m2_rgb / float(samples - 1);
      float3 w = make_float3(0.2126f, 0.7152f, 0.0722f);
      float lum = dot3(mean_rgb, w);
      float lum_var = dot3(var_rgb, w);
      float stddev_rgb = sqrtf(fmaxf(lum_var, 0.f));
      float noise = stddev_rgb / fmaxf(lum, 1e-6f);

      if (write_bayer) {
        float var_bayer = m2_bayer / float(samples - 1);
        float stddev_bayer = sqrtf(fmaxf(var_bayer, 0.f));
        float noise_bayer = stddev_bayer / fmaxf(mean_bayer, 1e-6f);
        noise = fmaxf(noise, noise_bayer);
      }

      float* nm = ptr_at<float>(params.noise_map);
      atomicMaxf(&nm[dst], noise);
      if (x > 0) atomicMaxf(&nm[dst - 1], noise);
      if (x + 1 < W) atomicMaxf(&nm[dst + 1], noise);
      if (y > 0) atomicMaxf(&nm[dst - W], noise);
      if (y + 1 < H) atomicMaxf(&nm[dst + W], noise);
      if (x > 0 && y > 0) atomicMaxf(&nm[dst - W - 1], noise);
      if (x + 1 < W && y > 0) atomicMaxf(&nm[dst - W + 1], noise);
      if (x > 0 && y + 1 < H) atomicMaxf(&nm[dst + W - 1], noise);
      if (x + 1 < W && y + 1 < H) atomicMaxf(&nm[dst + W + 1], noise);

      max_noise = fmaxf(noise, nm[dst]);
      if (x > 0) max_noise = fmaxf(max_noise, nm[dst - 1]);
      if (x + 1 < W) max_noise = fmaxf(max_noise, nm[dst + 1]);
      if (y > 0) max_noise = fmaxf(max_noise, nm[dst - W]);
      if (y + 1 < H) max_noise = fmaxf(max_noise, nm[dst + W]);
      if (x > 0 && y > 0) max_noise = fmaxf(max_noise, nm[dst - W - 1]);
      if (x + 1 < W && y > 0) max_noise = fmaxf(max_noise, nm[dst - W + 1]);
      if (x > 0 && y + 1 < H) max_noise = fmaxf(max_noise, nm[dst + W - 1]);
      if (x + 1 < W && y + 1 < H) max_noise = fmaxf(max_noise, nm[dst + W + 1]);

      if (max_noise < params.noise_threshold)
        break;
    }

    seed = prd.seed;
  }
  sc[dst] = samples;

  unsigned int* fl = ptr_at<unsigned int>(params.flags);
  unsigned int active = (samples < params.max_spp) ? 1u : 0u;
  if (params.noise_threshold > 0.f && samples >= params.min_adaptive_samples)
    active = (max_noise >= params.noise_threshold) && (samples < params.max_spp);
  fl[dst] = active;

  // Write outputs
  if (write_rgb) {
    float3* out = ptr_at<float3>(params.out_rgb);
    out[dst] = mean_rgb;
  }
  if (write_bayer) {
    float* out = ptr_at<float>(params.out_bayer);
    out[dst] = fmaxf(mean_bayer, 0.f);
  }
}

extern "C" __global__ void __raygen__bayer()
{
  const uint3  idx = optixGetLaunchIndex();
  const int    launch = int(idx.x);
  const int    W   = params.width;
  const int    H   = params.height;
  int dst = launch;
  if (params.active_pixels)
    dst = ptr_at<uint32_t>(params.active_pixels)[launch];
  const int    x   = dst % W;
  const int    y   = dst / W;

  const float3 org = params.cam_eye;
  float mean = 0.0f;
  if (params.out_bayer)
    mean = ptr_at<float>(params.out_bayer)[dst];
  float m2 = 0.0f;
  const int ch = bayer_channel_for(x, y, params.bayer_pattern);
  unsigned long long seed =
      ((unsigned long long)params.frame * 9781ULL) ^
      ((unsigned long long)dst * 6271ULL) ^
      0x37c4d1e74c3fa19bULL;

  int* sc = ptr_at<int>(params.sample_counts);
  int samples = sc[dst];
  float max_noise = params.noise_threshold;
  for (int s = 0; s < params.spp && samples < params.max_spp; ++s) {
    samples++;
    PRDScalar prd;
    prd.radiance = 0.0f;
    prd.throughput = 1.0f;
    prd.origin = org;
    float2 jitter = olpf_jitter(x, y, s, params.frame);
    prd.direction = sample_camera_dir(x, y, jitter);
    prd.depth = 0;
    prd.done = 0;
    prd.seed = seed;
    prd.prev_pdf_bsdf = 0.0f;
    prd.prev_pdf_valid = 0;
    prd.ch = ch;
    prd.px = x;
    prd.py = y;
    prd.sample = s;
    prd.rng_dim = 0;
    float sample = 0.0f;

    while (!prd.done) {
      prd.radiance = 0.0f;
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
      float contrib = prd.radiance * prd.throughput;
      sample += contrib;
    }

    float delta = sample - mean;
    mean += delta / float(samples);
    float delta2 = sample - mean;
    m2 += delta * delta2;

    if (params.noise_threshold > 0.f && samples >= params.min_adaptive_samples) {
      float var = m2 / float(samples - 1);
      float stddev = sqrtf(fmaxf(var, 0.f));
      float noise = stddev / fmaxf(mean, 1e-6f);

      float* nm = ptr_at<float>(params.noise_map);
      atomicMaxf(&nm[dst], noise);
      if (x > 0) atomicMaxf(&nm[dst - 1], noise);
      if (x + 1 < W) atomicMaxf(&nm[dst + 1], noise);
      if (y > 0) atomicMaxf(&nm[dst - W], noise);
      if (y + 1 < H) atomicMaxf(&nm[dst + W], noise);
      if (x > 0 && y > 0) atomicMaxf(&nm[dst - W - 1], noise);
      if (x + 1 < W && y > 0) atomicMaxf(&nm[dst - W + 1], noise);
      if (x > 0 && y + 1 < H) atomicMaxf(&nm[dst + W - 1], noise);
      if (x + 1 < W && y + 1 < H) atomicMaxf(&nm[dst + W + 1], noise);

      max_noise = fmaxf(noise, nm[dst]);
      if (x > 0) max_noise = fmaxf(max_noise, nm[dst - 1]);
      if (x + 1 < W) max_noise = fmaxf(max_noise, nm[dst + 1]);
      if (y > 0) max_noise = fmaxf(max_noise, nm[dst - W]);
      if (y + 1 < H) max_noise = fmaxf(max_noise, nm[dst + W]);
      if (x > 0 && y > 0) max_noise = fmaxf(max_noise, nm[dst - W - 1]);
      if (x + 1 < W && y > 0) max_noise = fmaxf(max_noise, nm[dst - W + 1]);
      if (x > 0 && y + 1 < H) max_noise = fmaxf(max_noise, nm[dst + W - 1]);
      if (x + 1 < W && y + 1 < H) max_noise = fmaxf(max_noise, nm[dst + W + 1]);

      if (max_noise < params.noise_threshold)
        break;
    }

    seed = prd.seed;
  }
  sc[dst] = samples;
  unsigned int* fl = ptr_at<unsigned int>(params.flags);
  unsigned int active = (samples < params.max_spp) ? 1u : 0u;
  if (params.noise_threshold > 0.f && samples >= params.min_adaptive_samples)
    active = (max_noise >= params.noise_threshold) && (samples < params.max_spp);
  fl[dst] = active;

  float* out = ptr_at<float>(params.out_bayer);
  out[dst] = fmaxf(mean, 0.f);
}

