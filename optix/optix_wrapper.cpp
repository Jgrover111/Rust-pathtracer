// optix_wrapper.cpp  — build with OptiX 9 + CUDA 13 + MSVC 2022
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include <string>
#include <memory>
#include <cmath>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

// The PTX path is injected by CMake as -DOPTIX_PTX_PATH=".../optix_device.ptx"
#ifndef OPTIX_PTX_PATH
#  error "OPTIX_PTX_PATH not defined (set by CMake)."
#endif

// ----- tiny helpers ---------------------------------------------------------
#define OTK_CHECK(call)                                                          \
  do {                                                                           \
    auto _res = (call);                                                          \
    if (_res != OPTIX_SUCCESS) {                                                 \
      fprintf(stderr, "OptiX error %d at %s:%d\n", int(_res), __FILE__, __LINE__); \
      std::abort();                                                              \
    }                                                                            \
  } while (0)

#define CU_CHECK(call)                                                           \
  do {                                                                           \
    auto _res = (call);                                                          \
    if (_res != CUDA_SUCCESS) {                                                  \
      const char* name = nullptr;                                                \
      const char* msg  = nullptr;                                                \
      cuGetErrorName(_res, &name);                                               \
      cuGetErrorString(_res, &msg);                                              \
      fprintf(stderr, "CUDA error %s (%d): %s at %s:%d\n",                       \
              name ? name : "?", int(_res), msg ? msg : "?", __FILE__, __LINE__);\
      std::abort();                                                              \
    }                                                                            \
  } while (0)

static std::vector<char> readFile(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Failed to open PTX: %s\n", path); std::abort(); }
    return std::vector<char>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
}

static OptixResult createModuleCompat(OptixDeviceContext ctx,
                                      const OptixModuleCompileOptions* mopts,
                                      const OptixPipelineCompileOptions* popts,
                                      const char* ptx, size_t ptxSize,
                                      char* log, size_t* logSize,
                                      OptixModule* module)
{
#if defined(OPTIX_VERSION) && (OPTIX_VERSION >= 80000)
  // OptiX 8/9: prefer the 9-arg variant with input format when available.
  #if defined(OPTIX_MODULE_INPUT_FORMAT_PTX) || defined(OPTIX_MODULE_INPUT_TYPE_PTX)
    #if defined(OPTIX_MODULE_INPUT_FORMAT_PTX)
      const auto kFmt = OPTIX_MODULE_INPUT_FORMAT_PTX;
    #else
      const auto kFmt = OPTIX_MODULE_INPUT_TYPE_PTX;
    #endif
    return optixModuleCreate(ctx, mopts, popts, ptx, ptxSize, kFmt, log, logSize, module);
  #else
    // Some headers expose optixModuleCreate without the format parameter.
    return optixModuleCreate(ctx, mopts, popts, ptx, ptxSize, log, logSize, module);
  #endif
#else
  // OptiX 7.x
  return optixModuleCreateFromPTX(ctx, mopts, popts, ptx, ptxSize, log, logSize, module);
#endif
}


template<typename T>
static void upload(const std::vector<T>& h, CUdeviceptr& d)
{
  size_t bytes = h.size() * sizeof(T);
  if (bytes == 0) {
    if (d) { CU_CHECK(cuMemFree(d)); d = 0; }
    return;
  }

  if (d) {
    CUdeviceptr base = 0;
    size_t alloc_bytes = 0;
    CU_CHECK(cuMemGetAddressRange(&base, &alloc_bytes, d));
    if (bytes > alloc_bytes) {
      CU_CHECK(cuMemFree(d));
      d = 0;
    }
  }

  if (!d) {
    CU_CHECK(cuMemAlloc(&d, bytes));
  }
  CU_CHECK(cuMemcpyHtoD(d, h.data(), bytes));
}

static float3 normalize3(const float3& v)
{
  float len = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
  if (len <= 0.f) return make_float3(0.f,0.f,0.f);
  float inv = 1.0f / len;
  return make_float3(v.x*inv, v.y*inv, v.z*inv);
}

static float3 cross3(const float3& a, const float3& b)
{
  return make_float3(a.y*b.z - a.z*b.y,
                     a.z*b.x - a.x*b.z,
                     a.x*b.y - a.y*b.x);
}

// ----- SBT record types ------------------------------------------------------
template<typename T>
struct alignas( OPTIX_SBT_RECORD_ALIGNMENT ) SbtRecord
{
  uint8_t header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T       data;
};

struct EmptyData {};

struct HitData {
  float3 kd;
  float3 ke;
};

// Host/device params must match the device-side struct layout.
struct Params
{
  // output
  CUdeviceptr out_rgb;        // float3*
  CUdeviceptr out_bayer;      // float*
  int         width;
  int         height;
  int         spp;
  int         frame;
  int         bayer_pattern;  // 0..3

  // camera
  float3 cam_eye;
  float3 cam_u;
  float3 cam_v;
  float3 cam_w;

  // scene
  OptixTraversableHandle handle;
  CUdeviceptr d_vertices; // float3*
  CUdeviceptr d_indices;  // uint3*
  CUdeviceptr d_normals;  // float3*
  uint32_t    num_triangles;

  // simple rectangular area light used in device
  float3 light_pos;
  float3 light_emit;
  float3 light_normal;
  float2 light_half;
};

using RaygenRecord  = SbtRecord<EmptyData>;
using MissRecord    = SbtRecord<EmptyData>;
using HitgroupRecord= SbtRecord<HitData>;

// ray type indices used by device code
enum { RAY_RADIANCE = 0, RAY_SHADOW = 1, RAY_TYPE_COUNT = 2 };

// ----- state -----------------------------------------------------------------
struct State
{
  // CUDA/OptiX
  CUcontext            cuCtx  = nullptr;
  OptixDeviceContext   ctx    = nullptr;

  // Pipeline
  OptixModule          module = nullptr;
  OptixProgramGroup    pg_raygen       = nullptr;
  OptixProgramGroup    pg_raygen_bayer= nullptr;
  OptixProgramGroup    pg_miss_rad     = nullptr;
  OptixProgramGroup    pg_miss_rad_bayer = nullptr;
  OptixProgramGroup    pg_miss_sh      = nullptr;
  OptixProgramGroup    pg_hit_rad      = nullptr;
  OptixProgramGroup    pg_hit_rad_bayer= nullptr;
  OptixProgramGroup    pg_hit_sh       = nullptr;
  OptixPipeline        pipeline        = nullptr;

  // SBT
  RaygenRecord         rg_rec_rgb {};
  RaygenRecord         rg_rec_bayer {};
  std::array<MissRecord, RAY_TYPE_COUNT> ms_rec_rgb {};
  std::array<MissRecord, RAY_TYPE_COUNT> ms_rec_bayer {};
  // Hitgroup records are per-triangle and therefore sized at runtime.
  // Using std::vector here lets us resize the container to match the
  // geometry without relying on a compile-time constant, something that
  // std::array cannot provide.
  std::vector<HitgroupRecord> hg_rec_rgb;
  std::vector<HitgroupRecord> hg_rec_bayer;
  OptixShaderBindingTable sbt_rgb {};
  OptixShaderBindingTable sbt_bayer {};

  CUdeviceptr d_sbt_rg_rgb = 0;
  CUdeviceptr d_sbt_rg_bayer = 0;
  CUdeviceptr d_sbt_ms_rgb = 0;
  CUdeviceptr d_sbt_ms_bayer = 0;
  CUdeviceptr d_sbt_hg_rgb = 0;
  CUdeviceptr d_sbt_hg_bayer = 0;

  // Geometry (host)
  std::vector<float3> vertices;
  std::vector<uint3>  indices;
  std::vector<float3> normals;
  // Per-triangle diffuse/emissive colors used when building SBT records.
  std::vector<float3> kd;
  std::vector<float3> ke;
  // Per-triangle SBT indices for associating primitives with records.
  std::vector<uint32_t> sbt_index;

  // Geometry (device)
  CUdeviceptr d_vertices = 0;
  CUdeviceptr d_indices  = 0;
  CUdeviceptr d_normals  = 0;
  CUdeviceptr d_sbt_index = 0;

  // GAS
  CUdeviceptr           d_gas   = 0;
  OptixTraversableHandle gas    = 0;

  // Params
  Params                h_params {};
  CUdeviceptr           d_params = 0;

  // Output
  CUdeviceptr           d_out_rgb   = 0; // float3*
  CUdeviceptr           d_out_bayer = 0; // float*
  uint32_t              width=0, height=0;
  CUstream              stream = nullptr;

  // options
  int                   bayer_pattern = 0;

  ~State() {
    if (d_out_rgb)   cuMemFree(d_out_rgb);
    if (d_out_bayer) cuMemFree(d_out_bayer);
    if (d_params)    cuMemFree(d_params);
    if (stream)      cuStreamDestroy(stream);

    if (d_vertices)  cuMemFree(d_vertices);
    if (d_indices)   cuMemFree(d_indices);
    if (d_sbt_index)      cuMemFree(d_sbt_index);
    if (d_sbt_rg_rgb)     cuMemFree(d_sbt_rg_rgb);
    if (d_sbt_rg_bayer)   cuMemFree(d_sbt_rg_bayer);
    if (d_sbt_ms_rgb)     cuMemFree(d_sbt_ms_rgb);
    if (d_sbt_ms_bayer)   cuMemFree(d_sbt_ms_bayer);
    if (d_sbt_hg_rgb)     cuMemFree(d_sbt_hg_rgb);
    if (d_sbt_hg_bayer)   cuMemFree(d_sbt_hg_bayer);
    if (d_normals)   cuMemFree(d_normals);
    if (d_gas)       cuMemFree(d_gas);

    if (pipeline)    optixPipelineDestroy(pipeline);
    if (pg_raygen)        optixProgramGroupDestroy(pg_raygen);
    if (pg_raygen_bayer)  optixProgramGroupDestroy(pg_raygen_bayer);
    if (pg_miss_rad)      optixProgramGroupDestroy(pg_miss_rad);
    if (pg_miss_rad_bayer)optixProgramGroupDestroy(pg_miss_rad_bayer);
    if (pg_miss_sh)       optixProgramGroupDestroy(pg_miss_sh);
    if (pg_hit_rad)       optixProgramGroupDestroy(pg_hit_rad);
    if (pg_hit_rad_bayer) optixProgramGroupDestroy(pg_hit_rad_bayer);
    if (pg_hit_sh)        optixProgramGroupDestroy(pg_hit_sh);
    if (module)      optixModuleDestroy(module);
    if (ctx)         optixDeviceContextDestroy(ctx);
    if (cuCtx) {
      CUdevice dev = 0;
      cuCtxGetDevice(&dev);
      cuDevicePrimaryCtxRelease(dev);
	}
  }
};

// ----- geometry --------------------------------------------------------------
static void addQuad(std::vector<float3>& V, std::vector<uint3>& I,
                    const float3& a, const float3& b, const float3& c, const float3& d)
{
  uint32_t base = static_cast<uint32_t>(V.size());
  V.push_back(a); V.push_back(b); V.push_back(c); V.push_back(d);
  I.push_back(make_uint3(base+0, base+1, base+2));
  I.push_back(make_uint3(base+0, base+2, base+3));
}

static void buildCornell(State& s)
{
  s.vertices.clear(); s.indices.clear(); s.normals.clear(); s.kd.clear(); s.ke.clear(); s.sbt_index.clear();

  const float3 red   = make_float3(0.63f, 0.065f, 0.05f);
  const float3 green = make_float3(0.14f, 0.45f, 0.091f);
  const float3 white = make_float3(0.725f,0.71f,0.68f);
  const float3 black = make_float3(0,0,0);
  const float3 emit  = make_float3(15.f, 15.f, 15.f);

  // room bounds (NVIDIA-style Cornell; y=up)
  const float3 A = make_float3(-1.0f, 0.0f, -1.0f); // floor corners
  const float3 B = make_float3( 1.0f, 0.0f, -1.0f);
  const float3 C = make_float3( 1.0f, 0.0f,  1.0f);
  const float3 D = make_float3(-1.0f, 0.0f,  1.0f);

  const float3 Au = make_float3(-1.0f, 2.0f, -1.0f); // ceiling corners
  const float3 Bu = make_float3( 1.0f, 2.0f, -1.0f);
  const float3 Cu = make_float3( 1.0f, 2.0f,  1.0f);
  const float3 Du = make_float3(-1.0f, 2.0f,  1.0f);

  // floor (white)
  addQuad(s.vertices, s.indices, D,C,B,A);
  s.kd.push_back(white); s.kd.push_back(white);
  s.ke.push_back(black); s.ke.push_back(black);

  // ceiling (white)
  addQuad(s.vertices, s.indices, Au,Bu,Cu,Du);
  s.kd.push_back(white); s.kd.push_back(white);
  s.ke.push_back(black); s.ke.push_back(black);

  // back wall (white)
  addQuad(s.vertices, s.indices, B,Bu,Au,A);
  s.kd.push_back(white); s.kd.push_back(white);
  s.ke.push_back(black); s.ke.push_back(black);

  // right wall (green)
  addQuad(s.vertices, s.indices, C,Cu,Bu,B);
  s.kd.push_back(green); s.kd.push_back(green);
  s.ke.push_back(black); s.ke.push_back(black);

  // left wall (red)
  addQuad(s.vertices, s.indices, A,Au,Du,D);
  s.kd.push_back(red); s.kd.push_back(red);
  s.ke.push_back(black); s.ke.push_back(black);

  // ceiling light cutout (small rectangle) — make it emissive
  const float3 L0 = make_float3(-0.3f, 1.999f, -0.3f);
  const float3 L1 = make_float3( 0.3f, 1.999f, -0.3f);
  const float3 L2 = make_float3( 0.3f, 1.999f,  0.3f);
  const float3 L3 = make_float3(-0.3f, 1.999f,  0.3f);
  addQuad(s.vertices, s.indices, L0,L1,L2,L3);
  s.kd.push_back(white); s.kd.push_back(white);
  s.ke.push_back(emit);  s.ke.push_back(emit);

  // optional: a short box (white)
  {
    float y0 = 0.0f, y1 = 0.6f;
    float3 p0 = make_float3(-0.53f, y0, -0.75f);
    float3 p1 = make_float3(-0.13f, y0, -0.65f);
    float3 p2 = make_float3(-0.23f, y0, -0.25f);
    float3 p3 = make_float3(-0.63f, y0, -0.35f);

    float3 q0 = make_float3(-0.53f, y1, -0.75f);
    float3 q1 = make_float3(-0.13f, y1, -0.65f);
    float3 q2 = make_float3(-0.23f, y1, -0.25f);
    float3 q3 = make_float3(-0.63f, y1, -0.35f);

    auto addRect = [&](float3 a,float3 b,float3 c,float3 d){
      addQuad(s.vertices, s.indices, a,b,c,d);
      s.kd.push_back(white); s.kd.push_back(white);
      s.ke.push_back(black); s.ke.push_back(black);
    };
    addRect(p3,p2,p1,p0); // bottom
    addRect(q3,q2,q1,q0); // top
    addRect(p1,p0,q0,q1);
    addRect(p2,p1,q1,q2);
    addRect(p3,p2,q2,q3);
    addRect(p0,p3,q3,q0);
  }

  // compute per-triangle normals
  s.normals.resize(s.indices.size());
  for (size_t i = 0; i < s.indices.size(); ++i) {
    const uint3 tri = s.indices[i];
    const float3& a = s.vertices[tri.x];
    const float3& b = s.vertices[tri.y];
    const float3& c = s.vertices[tri.z];
    float3 ab = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
    float3 ac = make_float3(c.x - a.x, c.y - a.y, c.z - a.z);
    s.normals[i] = normalize3(cross3(ab, ac));
  }

  // the SBT index buffer stores one base index per triangle;
  // the ray type is selected via the `sbtOffset` parameter in optixTrace
  s.sbt_index.resize(s.indices.size());
  for (uint32_t i = 0; i < s.sbt_index.size(); ++i)
    s.sbt_index[i] = i;

  // upload geometry arrays (materials stored in SBT)
  upload(s.vertices, s.d_vertices);
  upload(s.indices,  s.d_indices);
  upload(s.normals,  s.d_normals);
  upload(s.sbt_index, s.d_sbt_index);
}

// ----- GAS -------------------------------------------------------------------
static void buildGAS(State& s)
{
  OptixBuildInput tri {};
  tri.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

  CUdeviceptr d_vertices = s.d_vertices;
  CUdeviceptr d_indices  = s.d_indices;

  const uint32_t numVertices = static_cast<uint32_t>(s.vertices.size());
  const uint32_t numTriangles= static_cast<uint32_t>(s.indices.size());
  s.h_params.num_triangles   = numTriangles;

  tri.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
  tri.triangleArray.numVertices         = numVertices;
  tri.triangleArray.vertexBuffers       = &d_vertices;
  tri.triangleArray.vertexStrideInBytes = sizeof(float3);

  tri.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  tri.triangleArray.indexBuffer         = d_indices;
  tri.triangleArray.numIndexTriplets    = numTriangles;
  tri.triangleArray.indexStrideInBytes  = sizeof(uint3);

  // Each SBT record needs a corresponding geometry flag entry. Since each
  // triangle contributes one base record (ray type selection happens at trace
  // time via `sbtOffset`), size the flags array to the primitive count.
  std::vector<unsigned int> tri_flags(numTriangles, OPTIX_GEOMETRY_FLAG_NONE);
  tri.triangleArray.flags         = tri_flags.data();
  tri.triangleArray.numSbtRecords = numTriangles;
  tri.triangleArray.sbtIndexOffsetBuffer        = s.d_sbt_index;
  tri.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof(uint32_t);
  tri.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

  OptixAccelBuildOptions accel_opts {};
  accel_opts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  accel_opts.operation  = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes sizes {};
  OTK_CHECK( optixAccelComputeMemoryUsage(s.ctx, &accel_opts, &tri, 1, &sizes) );

  CUdeviceptr d_temp = 0, d_out = 0;
  CU_CHECK(cuMemAlloc(&d_temp, sizes.tempSizeInBytes));
  CU_CHECK(cuMemAlloc(&d_out,  sizes.outputSizeInBytes));

  OptixAccelEmitDesc emit {};
  CUdeviceptr d_compacted = 0;
  size_t      compacted_bytes = 0;
  // Request compacted size
  OTK_CHECK( optixAccelBuild(s.ctx, 0, &accel_opts, &tri, 1, d_temp, sizes.tempSizeInBytes, d_out, sizes.outputSizeInBytes, &s.gas, nullptr, 0) );
  // For simplicity, no compaction step (fast enough and keeps code shorter)
  s.d_gas = d_out;

  CU_CHECK(cuMemFree(d_temp));
}

// ----- pipeline / module / program groups ------------------------------------
static void createPipeline(State& s)
{
  char log[8192]; size_t logSize = sizeof(log);

  OptixModuleCompileOptions mopts{};
  mopts.maxRegisterCount = 0;
//  #ifdef OPTIX_COMPILE_OPTIMIZATION_LEVEL_DEFAULT
//  mopts.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_DEFAULT;
//  #endif
//  #ifdef OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO
//  mopts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
//  #endif
  mopts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
//  mopts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

  OptixPipelineCompileOptions popts {};
  popts.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
  popts.traversableGraphFlags  = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING |
                                 OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  popts.numPayloadValues       = 2;
  popts.numAttributeValues     = 2;
//  popts.traversableGraphFlags = OPTIX_TRaversableGraphFlags(OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS |
//                                                            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING);
//  popts.exceptionFlags         = OPTIX_EXCEPTION_FLAG_DEBUG;
  popts.pipelineLaunchParamsVariableName = "params";

  OptixPipelineLinkOptions link{};
  link.maxTraceDepth = 2; // allow a secondary optixTrace for shadow rays
  // On OptiX 9, link.debugLevel no longer exists.
//  #if defined(OPTIX_VERSION) && (OPTIX_VERSION < 90000)
//    #ifdef OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO
//      link.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
//    #endif
//  #endif

  auto ptxBytes = readFile(OPTIX_PTX_PATH);
  OTK_CHECK(createModuleCompat(s.ctx, &mopts, &popts,
                               ptxBytes.data(), ptxBytes.size(),
                               log, &logSize, &s.module));

  OptixProgramGroupOptions pg_opts {};

  OptixProgramGroupDesc rgd {}; rgd.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  rgd.raygen.module            = s.module;
  rgd.raygen.entryFunctionName = "__raygen__rg";
  logSize = sizeof(log);
  OTK_CHECK( optixProgramGroupCreate(s.ctx, &rgd, 1, &pg_opts, log, &logSize, &s.pg_raygen) );

  OptixProgramGroupDesc rgd_b {}; rgd_b.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  rgd_b.raygen.module            = s.module;
  rgd_b.raygen.entryFunctionName = "__raygen__bayer";
  logSize = sizeof(log);
  OTK_CHECK( optixProgramGroupCreate(s.ctx, &rgd_b, 1, &pg_opts, log, &logSize, &s.pg_raygen_bayer) );

  OptixProgramGroupDesc md1 {}; md1.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  md1.miss.module            = s.module;
  md1.miss.entryFunctionName = "__miss__ms_radiance";
  logSize = sizeof(log);
  OTK_CHECK( optixProgramGroupCreate(s.ctx, &md1, 1, &pg_opts, log, &logSize, &s.pg_miss_rad) );

  OptixProgramGroupDesc md1b {}; md1b.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  md1b.miss.module            = s.module;
  md1b.miss.entryFunctionName = "__miss__ms_radiance_scalar";
  logSize = sizeof(log);
  OTK_CHECK( optixProgramGroupCreate(s.ctx, &md1b, 1, &pg_opts, log, &logSize, &s.pg_miss_rad_bayer) );

  OptixProgramGroupDesc md2 {}; md2.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  md2.miss.module            = s.module;
  md2.miss.entryFunctionName = "__miss__ms_shadow";
  logSize = sizeof(log);
  OTK_CHECK( optixProgramGroupCreate(s.ctx, &md2, 1, &pg_opts, log, &logSize, &s.pg_miss_sh) );

  OptixProgramGroupDesc hgd1 {}; hgd1.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hgd1.hitgroup.moduleCH            = s.module;
  hgd1.hitgroup.entryFunctionNameCH = "__closesthit__ch";
  // no AH for radiance
  logSize = sizeof(log);
  OTK_CHECK( optixProgramGroupCreate(s.ctx, &hgd1, 1, &pg_opts, log, &logSize, &s.pg_hit_rad) );

  OptixProgramGroupDesc hgd1b {}; hgd1b.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hgd1b.hitgroup.moduleCH            = s.module;
  hgd1b.hitgroup.entryFunctionNameCH = "__closesthit__ch_bayer";
  logSize = sizeof(log);
  OTK_CHECK( optixProgramGroupCreate(s.ctx, &hgd1b, 1, &pg_opts, log, &logSize, &s.pg_hit_rad_bayer) );

  OptixProgramGroupDesc hgd2 {}; hgd2.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  // shadow ray uses only AH
  hgd2.hitgroup.moduleAH            = s.module;
  hgd2.hitgroup.entryFunctionNameAH = "__anyhit__ah_shadow";
  logSize = sizeof(log);
  OTK_CHECK( optixProgramGroupCreate(s.ctx, &hgd2, 1, &pg_opts, log, &logSize, &s.pg_hit_sh) );

  std::array<OptixProgramGroup,8> groups = {
    s.pg_raygen, s.pg_raygen_bayer,
    s.pg_miss_rad, s.pg_miss_rad_bayer, s.pg_miss_sh,
    s.pg_hit_rad, s.pg_hit_rad_bayer, s.pg_hit_sh
  };
  OptixStackSizes stack_sizes{};
  for (OptixProgramGroup pg : groups) {
    // No external functions, so the pipeline handle is nullptr
    OTK_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes, nullptr));
  }

  unsigned int dcss_trav  = 0;
  unsigned int dcss_state = 0;
  unsigned int css        = 0;
  OTK_CHECK(optixUtilComputeStackSizes(
      &stack_sizes, link.maxTraceDepth,
      /*maxCCDepth*/ 0, /*maxDCDepth*/ 0,
      &dcss_trav, &dcss_state, &css));

  logSize = sizeof(log);
  OTK_CHECK( optixPipelineCreate(s.ctx, &popts, &link,
                                 groups.data(), (unsigned)groups.size(),
                                 log, &logSize, &s.pipeline) );

  OTK_CHECK(optixPipelineSetStackSize(
      s.pipeline,
      dcss_trav,
      dcss_state,
      css,
      /*maxTraversableGraphDepth*/              2));

  // SBT
  memset(&s.sbt_rgb, 0, sizeof(OptixShaderBindingTable));
  memset(&s.sbt_bayer, 0, sizeof(OptixShaderBindingTable));

  OTK_CHECK( optixSbtRecordPackHeader(s.pg_raygen, &s.rg_rec_rgb) );
  OTK_CHECK( optixSbtRecordPackHeader(s.pg_raygen_bayer, &s.rg_rec_bayer) );
  OTK_CHECK( optixSbtRecordPackHeader(s.pg_miss_rad, &s.ms_rec_rgb[RAY_RADIANCE]) );
  OTK_CHECK( optixSbtRecordPackHeader(s.pg_miss_rad_bayer, &s.ms_rec_bayer[RAY_RADIANCE]) );
  OTK_CHECK( optixSbtRecordPackHeader(s.pg_miss_sh, &s.ms_rec_rgb[RAY_SHADOW]) );
  OTK_CHECK( optixSbtRecordPackHeader(s.pg_miss_sh, &s.ms_rec_bayer[RAY_SHADOW]) );

  const uint32_t num_tris = s.h_params.num_triangles;
  const uint32_t total_records = num_tris * RAY_TYPE_COUNT;
  s.hg_rec_rgb.assign(static_cast<size_t>(total_records), {});
  s.hg_rec_bayer.assign(static_cast<size_t>(total_records), {});
  for (uint32_t i = 0; i < num_tris; ++i) {
    HitgroupRecord& rec_rad = s.hg_rec_rgb[i * RAY_TYPE_COUNT + RAY_RADIANCE];
    OTK_CHECK(optixSbtRecordPackHeader(s.pg_hit_rad, &rec_rad));
    rec_rad.data.kd = s.kd[i];
    rec_rad.data.ke = s.ke[i];
    HitgroupRecord& rec_sh = s.hg_rec_rgb[i * RAY_TYPE_COUNT + RAY_SHADOW];
    OTK_CHECK(optixSbtRecordPackHeader(s.pg_hit_sh, &rec_sh));
    rec_sh.data.kd = make_float3(0.f,0.f,0.f);
    rec_sh.data.ke = make_float3(0.f,0.f,0.f);

    HitgroupRecord& rec_rad_b = s.hg_rec_bayer[i * RAY_TYPE_COUNT + RAY_RADIANCE];
    OTK_CHECK(optixSbtRecordPackHeader(s.pg_hit_rad_bayer, &rec_rad_b));
    rec_rad_b.data.kd = s.kd[i];
    rec_rad_b.data.ke = s.ke[i];
    HitgroupRecord& rec_sh_b = s.hg_rec_bayer[i * RAY_TYPE_COUNT + RAY_SHADOW];
    OTK_CHECK(optixSbtRecordPackHeader(s.pg_hit_sh, &rec_sh_b));
    rec_sh_b.data.kd = make_float3(0.f,0.f,0.f);
    rec_sh_b.data.ke = make_float3(0.f,0.f,0.f);
  }

  CUdeviceptr d_rg_rgb=0, d_rg_bayer=0;
  CUdeviceptr d_ms_rgb=0, d_ms_bayer=0;
  CUdeviceptr d_hg_rgb=0, d_hg_bayer=0;
  CU_CHECK(cuMemAlloc(&d_rg_rgb, sizeof(RaygenRecord)));
  CU_CHECK(cuMemAlloc(&d_rg_bayer, sizeof(RaygenRecord)));
  CU_CHECK(cuMemAlloc(&d_ms_rgb, sizeof(MissRecord) * RAY_TYPE_COUNT));
  CU_CHECK(cuMemAlloc(&d_ms_bayer, sizeof(MissRecord) * RAY_TYPE_COUNT));
  CU_CHECK(cuMemAlloc(&d_hg_rgb, sizeof(HitgroupRecord) * s.hg_rec_rgb.size()));
  CU_CHECK(cuMemAlloc(&d_hg_bayer, sizeof(HitgroupRecord) * s.hg_rec_bayer.size()));
  CU_CHECK(cuMemcpyHtoD(d_rg_rgb, &s.rg_rec_rgb, sizeof(RaygenRecord)));
  CU_CHECK(cuMemcpyHtoD(d_rg_bayer, &s.rg_rec_bayer, sizeof(RaygenRecord)));
  CU_CHECK(cuMemcpyHtoD(d_ms_rgb, s.ms_rec_rgb.data(), sizeof(MissRecord) * RAY_TYPE_COUNT));
  CU_CHECK(cuMemcpyHtoD(d_ms_bayer, s.ms_rec_bayer.data(), sizeof(MissRecord) * RAY_TYPE_COUNT));
  CU_CHECK(cuMemcpyHtoD(d_hg_rgb, s.hg_rec_rgb.data(), sizeof(HitgroupRecord) * s.hg_rec_rgb.size()));
  CU_CHECK(cuMemcpyHtoD(d_hg_bayer, s.hg_rec_bayer.data(), sizeof(HitgroupRecord) * s.hg_rec_bayer.size()));

  s.d_sbt_rg_rgb = d_rg_rgb;
  s.d_sbt_rg_bayer = d_rg_bayer;
  s.d_sbt_ms_rgb = d_ms_rgb;
  s.d_sbt_ms_bayer = d_ms_bayer;
  s.d_sbt_hg_rgb = d_hg_rgb;
  s.d_sbt_hg_bayer = d_hg_bayer;

  s.sbt_rgb.raygenRecord                = d_rg_rgb;
  s.sbt_rgb.missRecordBase              = d_ms_rgb;
  s.sbt_rgb.missRecordStrideInBytes     = sizeof(MissRecord);
  s.sbt_rgb.missRecordCount             = RAY_TYPE_COUNT;
  s.sbt_rgb.hitgroupRecordBase          = d_hg_rgb;
  s.sbt_rgb.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
  s.sbt_rgb.hitgroupRecordCount         = total_records;

  s.sbt_bayer.raygenRecord                = d_rg_bayer;
  s.sbt_bayer.missRecordBase              = d_ms_bayer;
  s.sbt_bayer.missRecordStrideInBytes     = sizeof(MissRecord);
  s.sbt_bayer.missRecordCount             = RAY_TYPE_COUNT;
  s.sbt_bayer.hitgroupRecordBase          = d_hg_bayer;
  s.sbt_bayer.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
  s.sbt_bayer.hitgroupRecordCount         = total_records;
}

// ----- create/destroy --------------------------------------------------------
static State* make_state(uint32_t W, uint32_t H)
{
  CU_CHECK(cuInit(0));
  CUdevice dev = 0;
  CU_CHECK(cuDeviceGet(&dev, 0));
  CUcontext ctxCU = nullptr;
  CU_CHECK(cuDevicePrimaryCtxRetain(&ctxCU, dev));
  CU_CHECK(cuCtxSetCurrent(ctxCU));

  OTK_CHECK(optixInit());

  auto st = std::make_unique<State>();
  st->cuCtx = ctxCU;
  OptixDeviceContextOptions opts {};
#ifndef NDEBUG
  opts.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#else
  opts.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_NONE;
#endif
  opts.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
  opts.logCallbackFunction = [](unsigned int level, const char* tag, const char* msg, void*) {
    fprintf(stderr, "[OPTIX][%u][%s] %s\n", level, tag ? tag : "", msg ? msg : "");
    fflush(stderr);
  };
  opts.logCallbackLevel = 4;
  OTK_CHECK( optixDeviceContextCreate(ctxCU, &opts, &st->ctx) );

  st->width = W; st->height = H;

  // default cam looking into the box
  float3 eye    = make_float3(0.0f, 1.0f, 3.0f);
  float3 lookat = make_float3(0.0f, 1.0f, 0.0f);
  float3 up     = make_float3(0.0f, 1.0f, 0.0f);
  float fovY    = 45.0f * float(M_PI) / 180.0f;
  float3 Wv = normalize3( make_float3(lookat.x-eye.x, lookat.y-eye.y, lookat.z-eye.z) );
  float3 Uv = normalize3( cross3(Wv, up) );
  float3 Vv = cross3(Uv, Wv);
  float aspect = (H==0)?1.f : float(W)/float(H);
  float3 cw = Wv;
  float3 cuv = make_float3(std::tan(fovY*0.5f)*aspect*Uv.x,
                          std::tan(fovY*0.5f)*aspect*Uv.y,
                          std::tan(fovY*0.5f)*aspect*Uv.z);
  float3 cv = make_float3(std::tan(fovY*0.5f)*Vv.x,
                          std::tan(fovY*0.5f)*Vv.y,
                          std::tan(fovY*0.5f)*Vv.z);

  st->h_params.cam_eye = eye;
  st->h_params.cam_w   = cw;
  st->h_params.cam_u   = cuv;
  st->h_params.cam_v   = cv;

  // build scene
  buildCornell(*st);
  buildGAS(*st);
  st->h_params.handle     = st->gas;
  st->h_params.d_vertices = st->d_vertices;
  st->h_params.d_indices  = st->d_indices;
  st->h_params.d_normals  = st->d_normals;
  st->h_params.light_pos  = make_float3(0.f, 1.99f, 0.f);
  st->h_params.light_emit = make_float3(15.f, 15.f, 15.f); // match emissive ceiling light
  st->h_params.light_normal = normalize3(make_float3(0.f, -1.f, 0.f));
  st->h_params.light_half = make_float2(0.3f, 0.3f);

  // pipeline
  createPipeline(*st);

  // params/output buffers
  size_t rgbBytes    = size_t(W) * size_t(H) * sizeof(float3);
  size_t bayerBytes  = size_t(W) * size_t(H) * sizeof(float);

  CU_CHECK(cuMemAlloc(&st->d_out_rgb,   rgbBytes));
  CU_CHECK(cuMemAlloc(&st->d_out_bayer, bayerBytes));
  CU_CHECK(cuStreamCreate(&st->stream, CU_STREAM_NON_BLOCKING));

  st->h_params.out_rgb   = st->d_out_rgb;
  st->h_params.out_bayer = st->d_out_bayer;
  st->h_params.width     = W;
  st->h_params.height    = H;
  st->h_params.spp     = 1;
  st->h_params.frame     = 0;
  st->h_params.bayer_pattern = 0;

  CU_CHECK(cuMemAlloc(&st->d_params, sizeof(Params)));
  CU_CHECK(cuMemcpyHtoD(st->d_params, &st->h_params, sizeof(Params)));

  return st.release();
}

// ----- C API -----------------------------------------------------------------
static void* g_handle = nullptr;
static int g_last_w = 0;
static int g_last_h = 0;

extern "C" __declspec(dllexport)
void* optix_ctx_create(int width, int height)
{
  return (void*) make_state((uint32_t)width, (uint32_t)height);
}

extern "C" __declspec(dllexport)
void  optix_ctx_destroy(void* handle)
{
  delete reinterpret_cast<State*>(handle);
}

extern "C" __declspec(dllexport)
void  optix_ctx_set_bayer(void* handle, int pattern /*0..3*/)
{
  auto& s = *reinterpret_cast<State*>(handle);
  s.bayer_pattern = pattern;
  s.h_params.bayer_pattern = pattern;
  CU_CHECK(cuMemcpyHtoD(s.d_params, &s.h_params, sizeof(Params)));
}

extern "C" __declspec(dllexport)
void  optix_ctx_set_camera(void* handle,
                           float eye_x, float eye_y, float eye_z,
                           float look_x,float look_y,float look_z,
                           float up_x,  float up_y,  float up_z,
                           float vfov_deg)
{
  auto& s = *reinterpret_cast<State*>(handle);
  const float3 eye    = make_float3(eye_x,eye_y,eye_z);
  const float3 lookat = make_float3(look_x,look_y,look_z);
  const float3 up     = make_float3(up_x,up_y,up_z);

  float3 Wv = normalize3( make_float3(lookat.x-eye.x, lookat.y-eye.y, lookat.z-eye.z) );
  float3 Uv = normalize3( cross3(Wv, up) );
  float3 Vv = cross3(Uv, Wv);
  float fovY = vfov_deg * float(M_PI) / 180.0f;
  float aspect = (s.height==0)?1.f : float(s.width)/float(s.height);

  s.h_params.cam_eye = eye;
  s.h_params.cam_w   = Wv;
  s.h_params.cam_u   = make_float3(std::tan(fovY*0.5f)*aspect*Uv.x,
                                   std::tan(fovY*0.5f)*aspect*Uv.y,
                                   std::tan(fovY*0.5f)*aspect*Uv.z);
  s.h_params.cam_v   = make_float3(std::tan(fovY*0.5f)*Vv.x,
                                   std::tan(fovY*0.5f)*Vv.y,
                                   std::tan(fovY*0.5f)*Vv.z);

  CU_CHECK(cuMemcpyHtoD(s.d_params, &s.h_params, sizeof(Params)));
}

static int optix_ctx_render_rgb(void* handle, int spp, float* out_rgb)
{
    if (!out_rgb) return -1;
    // same pattern you use elsewhere: update Params on device, launch, copy back
    auto& s = *reinterpret_cast<State*>(handle);

    s.h_params.frame++;
	s.h_params.spp = spp;
    // If your renderer needs any per-frame fields set, do it here.
    // e.g., s.h_params.bayer_pattern = s.bayer_pattern;   // (RGB path usually not needed)

    CU_CHECK( cuMemcpyHtoD(s.d_params, &s.h_params, sizeof(Params)) );

    OTK_CHECK( optixLaunch(
        s.pipeline,
        /*stream*/ s.stream,
        /*params*/ s.d_params,
        /*paramsSize*/ sizeof(Params),
        /*SBT*/ &s.sbt_rgb,
        /*w,h,d*/ s.width, s.height, 1) );

    // interleaved RGB float32
    const size_t bytes = size_t(s.width) * size_t(s.height) * 3 * sizeof(float);
    // Host memory is typically not page-locked, so use synchronous copy
    // to avoid CUDA_ERROR_INVALID_VALUE from cuMemcpyDtoHAsync.
    CU_CHECK( cuMemcpyDtoH(out_rgb, s.d_out_rgb, bytes) );

    return 0;
}

static int optix_ctx_render_bayer_f32(void* handle, int spp, float* out_raw)
{
    if (!out_raw) return -1;
    auto& s = *reinterpret_cast<State*>(handle);

    s.h_params.frame++;
        s.h_params.spp = spp;
    s.h_params.bayer_pattern = s.bayer_pattern;  // keep whatever you already store in s.bayer_pattern
    CUdeviceptr saved_out_rgb = s.h_params.out_rgb;
    s.h_params.out_rgb = 0;
    CU_CHECK( cuMemcpyHtoD(s.d_params, &s.h_params, sizeof(Params)) );

    OTK_CHECK( optixLaunch(
        s.pipeline,
        /*stream*/ s.stream,
        /*params*/ s.d_params,
        /*paramsSize*/ sizeof(Params),
        /*SBT*/ &s.sbt_bayer,
        /*w,h,d*/ s.width, s.height, 1) );

    const size_t bytes = size_t(s.width) * size_t(s.height) * sizeof(float);
    // As above, perform a synchronous copy since the host buffer may not be pinned.
    CU_CHECK( cuMemcpyDtoH(out_raw, s.d_out_bayer, bytes) );

    s.h_params.out_rgb = saved_out_rgb;
    return 0;
}

// Helper to lazily create/destroy the renderer state.  The OptiX context
// has a number of expensive resources associated with it, so we keep it
// around between calls and recreate it only if the render resolution
// changes.  This function lives here so that the C API wrappers below can
// use it without needing forward declarations of the exported symbols.
static void ensure_handle(int w, int h)
{
    if (!g_handle || w != g_last_w || h != g_last_h) {
        if (g_handle) optix_ctx_destroy(g_handle);
        g_handle = optix_ctx_create(w, h);
        g_last_w = w; g_last_h = h;
        // Set a simple default camera once (tweak if you like)
        optix_ctx_set_camera(g_handle,
                             0.f, 1.f, 3.f,
                             0.f, 1.f, 0.f,
                             0.f, 1.f, 0.f,
                             45.f);
    }
}

extern "C" __declspec(dllexport)
int optix_render_rgb(int w, int h, int spp, float* out_rgb)
{
    ensure_handle(w, h);
    return optix_ctx_render_rgb(g_handle, spp, out_rgb);
}

extern "C" __declspec(dllexport)
int optix_render_bayer_f32(int w, int h, int spp, int pat, float* out_raw)
{
    ensure_handle(w, h);
    optix_ctx_set_bayer(g_handle, pat);
    return optix_ctx_render_bayer_f32(g_handle, spp, out_raw);
}
extern "C" __declspec(dllexport)
void optix_synchronize()
{
    if (g_handle) {
        auto& s = *reinterpret_cast<State*>(g_handle);
        CU_CHECK(cuStreamSynchronize(s.stream));
    }
}

// Pinned host allocations need the CUDA primary context to remain active while
// the memory is in use. Track a retained context and release it once the last
// allocation has been freed.
static CUcontext g_pinned_ctx = nullptr;
static size_t    g_pinned_allocs = 0;

extern "C" __declspec(dllexport)
void* optix_alloc_host(size_t bytes)
{
    CU_CHECK(cuInit(0));
    if (!g_pinned_ctx) {
        CUdevice dev = 0;
        CU_CHECK(cuDeviceGet(&dev, 0));
        CU_CHECK(cuDevicePrimaryCtxRetain(&g_pinned_ctx, dev));
    }

    CU_CHECK(cuCtxSetCurrent(g_pinned_ctx));
    void* ptr = nullptr;
    CU_CHECK(cuMemAllocHost(&ptr, bytes));
    ++g_pinned_allocs;
    CU_CHECK(cuCtxSetCurrent(nullptr));
    return ptr;
}

extern "C" __declspec(dllexport)
void optix_free_host(void* ptr)
{
    if (ptr) {
        CU_CHECK(cuCtxSetCurrent(g_pinned_ctx));
        CU_CHECK(cuMemFreeHost(ptr));
        CU_CHECK(cuCtxSetCurrent(nullptr));
        if (--g_pinned_allocs == 0) {
            CUdevice dev = 0;
            CU_CHECK(cuDeviceGet(&dev, 0));
            CU_CHECK(cuDevicePrimaryCtxRelease(dev));
            g_pinned_ctx = nullptr;
        }
    }
}


