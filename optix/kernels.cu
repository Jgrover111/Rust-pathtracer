#include <cuda_runtime.h>
#include <vector_types.h>
#include <stdint.h>
#include "cuda_fixups.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__host__ __device__ inline float3 f3(float x,float y,float z){ return make_float3(x,y,z); }
__host__ __device__ inline float3 operator+(const float3& a, const float3& b){ return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__host__ __device__ inline float3 operator-(const float3& a, const float3& b){ return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
__host__ __device__ inline float3 operator*(const float3& a, float b){ return make_float3(a.x*b, a.y*b, a.z*b); }
__host__ __device__ inline float3 operator*(float b, const float3& a){ return a*b; }
__host__ __device__ inline float3 operator*(const float3& a, const float3& b){ return make_float3(a.x*b.x, a.y*b.y, a.z*b.z); }
__host__ __device__ inline float  dot(const float3& a, const float3& b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ inline float3 cross(const float3& a, const float3& b){ return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); }
__host__ __device__ inline float  length(const float3& v){ return sqrtf(dot(v,v)); }
__host__ __device__ inline float3 normalize(const float3& v){ float l=length(v); return (l>0)? v*(1.0f/l) : v; }
__host__ __device__ inline float3 clamp3(const float3& c, float lo, float hi){ return make_float3(fminf(fmaxf(c.x,lo),hi), fminf(fmaxf(c.y,lo),hi), fminf(fmaxf(c.z,lo),hi)); }

struct Rng { uint32_t s; };
__host__ __device__ inline uint32_t xorshift(Rng& r){ uint32_t x=r.s; x^=x<<13; x^=x>>17; x^=x<<5; r.s=x; return x; }
__host__ __device__ inline float randf(Rng& r){ return (xorshift(r) & 0x00FFFFFF) * (1.0f/16777216.0f); }

__constant__ float3 C_ROOM_MIN = { -1.f, 0.f, -1.f };
__constant__ float3 C_ROOM_MAX = {  1.f, 2.f,  1.f };
__constant__ float3 C_LIGHT_POS   = { 0.0f, 2.0f, 0.0f };
__constant__ float2 C_LIGHT_HALF  = { 0.25f, 0.25f };
__constant__ float3 C_LIGHT_EMIT  = { 12.0f, 12.0f, 12.0f };

__device__ inline float3 wall_color(int wall_id){
    if (wall_id==0) return f3(0.80f, 0.05f, 0.05f);
    if (wall_id==1) return f3(0.05f, 0.80f, 0.05f);
    return f3(0.80f, 0.80f, 0.80f);
}

struct Ray { float3 o; float3 d; };

__device__ inline bool point_on_light(const float3& p){
    return (fabsf(p.y - C_ROOM_MAX.y) < 1e-3f) &&
           (fabsf(p.x - C_LIGHT_POS.x) <= C_LIGHT_HALF.x) &&
           (fabsf(p.z - C_LIGHT_POS.z) <= C_LIGHT_HALF.y);
}

__device__ inline bool hit_room(const Ray& r, float& tHit, float3& n, int& wall_id){
    float3 t1 = (C_ROOM_MIN - r.o) / r.d;
    float3 t2 = (C_ROOM_MAX - r.o) / r.d;
    float3 tmin3 = make_float3(fminf(t1.x,t2.x), fminf(t1.y,t2.y), fminf(t1.z,t2.z));
    float3 tmax3 = make_float3(fmaxf(t1.x,t2.x), fmaxf(t1.y,t2.y), fmaxf(t1.z,t2.z));
    float tmin = fmaxf(fmaxf(tmin3.x, tmin3.y), tmin3.z);
    float tmax = fminf(fminf(tmax3.x, tmax3.y), tmax3.z);
    if (tmax < 0.f || tmin > tmax) return false;
    tHit = (tmin > 1e-4f) ? tmin : tmax;
    if (tHit < 1e-4f) return false;
    float3 p = r.o + r.d * tHit;
    const float eps = 1e-3f;
    if (fabsf(p.x - C_ROOM_MIN.x) < eps){ n = f3(+1,0,0); wall_id=0; return true; }
    if (fabsf(p.x - C_ROOM_MAX.x) < eps){ n = f3(-1,0,0); wall_id=1; return true; }
    if (fabsf(p.y - C_ROOM_MAX.y) < eps){ n = f3(0,-1,0); wall_id=2; return true; }
    if (fabsf(p.y - C_ROOM_MIN.y) < eps){ n = f3(0,+1,0); wall_id=3; return true; }
    if (fabsf(p.z - C_ROOM_MAX.z) < eps){ n = f3(0,0,-1); wall_id=4; return true; }
    if (fabsf(p.z - C_ROOM_MIN.z) < eps){ n = f3(0,0,+1); wall_id=5; return true; }
    n = make_float3(0,1,0); wall_id=3; return true;
}

__device__ inline void basis(const float3& n, float3& t, float3& b){
    if (fabsf(n.x) > fabsf(n.z)) t = normalize(f3(-n.y, n.x, 0.0f));
    else                         t = normalize(f3(0.0f, -n.z, n.y));
    b = cross(n, t);
}

__device__ inline float3 cosine_sample(const float3& n, Rng& rng){
    float u1 = randf(rng);
    float u2 = randf(rng);
    float r = sqrtf(u1);
    float phi = 2.0f * (float)M_PI * u2;
    float x = r * cosf(phi);
    float z = r * sinf(phi);
    float y = sqrtf(fmaxf(0.0f, 1.0f - u1));
    float3 t, bb; basis(n,t,bb);
    return normalize(t * x + n * y + bb * z);
}

__device__ inline bool visible_to_light(const float3& p, const float3& dir, float max_t){
    Ray sr; sr.o = p; sr.d = dir;
    float t; float3 n; int wall;
    if (!hit_room(sr, t, n, wall)) return false;
    if (t < max_t - 1e-4f) return false;
    float3 hp = sr.o + sr.d * t;
    return wall==2 && point_on_light(hp);
}

__device__ inline float3 direct_light_mis(const float3& p, const float3& n, const float3& albedo, Rng& rng){
    float u = ((randf(rng)*2.f) - 1.f) * C_LIGHT_HALF.x;
    float v = ((randf(rng)*2.f) - 1.f) * C_LIGHT_HALF.y;
    float3 lp = make_float3(C_LIGHT_POS.x + u, C_LIGHT_POS.y, C_LIGHT_POS.z + v);
    float3 L = lp - p;
    float dist2 = fmaxf(1e-6f, dot(L,L));
    float3 wi = L * rsqrtf(dist2);
    float cosS = fmaxf(0.f, dot(n, wi));
    float cosL = fmaxf(0.f, wi.y);
    if (cosS <= 0.f || cosL <= 0.f) return f3(0,0,0);
    if (!visible_to_light(p + n*1e-3f, wi, sqrtf(dist2))) return f3(0,0,0);
    float area = 4.f * C_LIGHT_HALF.x * C_LIGHT_HALF.y;
    float pdf_light = dist2 / (cosL * area);
    float pdf_bsdf  = cosS * (1.0f/(float)M_PI);
    float w = pdf_light / (pdf_light + pdf_bsdf);
    float3 f = albedo * (1.0f/(float)M_PI);
    return C_LIGHT_EMIT * (w * (dot(n, wi)) / pdf_light) * f;
}

__device__ inline Ray make_camera_ray(int x, int y, int W, int H, Rng& rng){
    float fx = (((x + randf(rng)) / (float)W) * 2.f) - 1.f;
    float fy = (((y + randf(rng)) / (float)H) * 2.f) - 1.f;
    float aspect = (float)W / (float)H;
    float3 cam_pos = make_float3(-0.20f, 1.05f, -0.95f);
    float3 target  = make_float3( 0.00f, 1.00f,  0.35f);
    float3 forward = normalize(target - cam_pos);
    float3 right   = normalize(cross(forward, make_float3(0,1,0)));
    float3 up      = normalize(cross(right, forward));
    float fov = 60.f * (float)M_PI/180.f;
    float scale = tanf(fov*0.5f);
    float3 dir = normalize(forward + fx*aspect*scale*right + (-fy)*scale*up);
    Ray r; r.o = cam_pos; r.d = dir; return r;
}

__device__ inline float3 radiance(Ray ray, Rng& rng, int max_depth){
    float3 Lsum = f3(0,0,0);
    float3 throughput = f3(1,1,1);
    float3 prev_n = f3(0,0,0);
    float  prev_pdf_bsdf = 0.0f;
    bool   have_prev_pdf = false;

    for (int depth=0; depth<max_depth; ++depth){
        float t; float3 n; int wall;
        if (!hit_room(ray, t, n, wall)) break;
        float3 p = ray.o + ray.d * t;

        if (point_on_light(p)){
            float3 Le = C_LIGHT_EMIT;
            if (have_prev_pdf){
                float3 Lvec = p - ray.o;
                float dist2 = fmaxf(1e-6f, dot(Lvec,Lvec));
                float3 wi = Lvec * rsqrtf(dist2);
                float cosL = fmaxf(0.f, wi.y);
                float area = 4.f * C_LIGHT_HALF.x * C_LIGHT_HALF.y;
                float pdf_light = dist2 / (cosL * area);
                float w_bsdf = prev_pdf_bsdf / (prev_pdf_bsdf + pdf_light);
                Lsum = Lsum + throughput * Le * w_bsdf;
            } else {
                Lsum = Lsum + throughput * Le;
            }
            break;
        }

        float3 albedo = wall_color(wall);
        Lsum = Lsum + throughput * direct_light_mis(p, n, albedo, rng);

        float3 wi = cosine_sample(n, rng);
        float cosS = fmaxf(0.f, dot(n, wi));
        float pdf_bsdf = cosS * (1.0f/(float)M_PI);
        throughput = throughput * albedo;
        have_prev_pdf = true;
        prev_pdf_bsdf = pdf_bsdf;
        prev_n = n;

        if (depth >= 2){
            float pcont = fminf(0.95f, fmaxf(0.05f, fmaxf(throughput.x, fmaxf(throughput.y, throughput.z))));
            if (randf(rng) > pcont) break;
            throughput = throughput * (1.0f / pcont);
        }

        ray.o = p + n * 1e-3f;
        ray.d = wi;
    }
    return Lsum;
}

__global__ void k_render_rgb(int W,int H,int spp,float* out_rgb){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=W || y>=H) return;
    Rng rng{ (uint32_t)(y*9781 + x*6271 + 0x68bc21u) };
    const int MAX_DEPTH = 8;
    float3 accum = f3(0,0,0);
    for (int s=0; s<spp; ++s){
        Ray ray = make_camera_ray(x,y,W,H,rng);
        float3 c = radiance(ray, rng, MAX_DEPTH);
        accum = accum + c;
    }
    float3 col = clamp3(accum * (1.0f / (float)spp), 0.f, 1e6f);
    int idx = (y*W + x)*3;
    out_rgb[idx+0] = col.x;
    out_rgb[idx+1] = col.y;
    out_rgb[idx+2] = col.z;
}

__device__ inline int bayer_channel_for(int x,int y,int pattern){
    int mx = x & 1;
    int my = y & 1;
    switch(pattern){
        case 0: return (my==0) ? (mx==0 ? 0:1) : (mx==0 ? 1:2);
        case 1: return (my==0) ? (mx==0 ? 2:1) : (mx==0 ? 1:0);
        case 2: return (my==0) ? (mx==0 ? 1:0) : (mx==0 ? 2:1);
        default:return (my==0) ? (mx==0 ? 1:2) : (mx==0 ? 0:1);
    }
}

__global__ void k_render_bayer_raw16(int W,int H,int spp,int pattern,uint16_t* out_raw){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=W || y>=H) return;
    Rng rng{ (uint32_t)(y*9781 + x*6271 + 0x37c4d1u) };
    const int MAX_DEPTH = 8;
    const int ch = bayer_channel_for(x,y,pattern);
    float accum = 0.0f;
    for (int s=0; s<spp; ++s){
        Ray ray = make_camera_ray(x,y,W,H,rng);
        float3 c = radiance(ray, rng, MAX_DEPTH);
        float v = (ch==0 ? c.x : (ch==1 ? c.y : c.z));
        accum += v;
    }
    float m = fminf(fmaxf(accum * (1.0f / (float)spp), 0.f), 1.f);
    out_raw[y*W + x] = (uint16_t)lrintf(m * 65535.0f);
}

extern "C" int gpu_render_rgb(int W,int H,int spp,float* out_rgb){
    if (!out_rgb || W<=0 || H<=0 || spp<=0) return 1;
    float* d_rgb=nullptr;
    cudaError_t err = cudaMalloc(&d_rgb, (size_t)W*(size_t)H*3*sizeof(float));
    if (err != cudaSuccess) return 1;
    dim3 B(16,16); dim3 G((W+B.x-1)/B.x, (H+B.y-1)/B.y);
    k_render_rgb<<<G,B>>>(W,H,spp,d_rgb);
    err = cudaGetLastError(); if (err != cudaSuccess){ cudaFree(d_rgb); return 1; }
    err = cudaMemcpy(out_rgb, d_rgb, (size_t)W*(size_t)H*3*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){ cudaFree(d_rgb); return 1; }
    cudaFree(d_rgb);
    return 0;
}

extern "C" int gpu_render_bayer_raw16(int W,int H,int spp,int pattern,uint16_t* out_raw16){
    if (!out_raw16 || W<=0 || H<=0 || spp<=0) return 1;
    uint16_t* d_raw=nullptr;
    cudaError_t err = cudaMalloc(&d_raw, (size_t)W*(size_t)H*sizeof(uint16_t));
    if (err != cudaSuccess) return 1;
    dim3 B(16,16); dim3 G((W+B.x-1)/B.x, (H+B.y-1)/B.y);
    k_render_bayer_raw16<<<G,B>>>(W,H,spp,pattern,d_raw);
    err = cudaGetLastError(); if (err != cudaSuccess){ cudaFree(d_raw); return 1; }
    err = cudaMemcpy(out_raw16, d_raw, (size_t)W*(size_t)H*sizeof(uint16_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){ cudaFree(d_raw); return 1; }
    cudaFree(d_raw);
    return 0;
}
