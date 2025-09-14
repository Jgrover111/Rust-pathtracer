#include <vector_functions.h>
#pragma once
#include <cuda.h>
#include <vector_types.h>
#include <stdint.h>
#include <math_constants.h>

struct GuideRegion { float3 bmin,bmax; uint32_t lobe_ofs,lobe_num; };
struct GuideLobe   { float3 mu; float kappa; float weight; float3 rgb; };
struct GuideGPU {
  CUdeviceptr d_regions; uint32_t region_count;
  CUdeviceptr d_lobes;   uint32_t lobe_count;
  CUdeviceptr d_grid;    int3 grid_res; float3 grid_min,grid_max,cell_size;
};

__device__ __forceinline__ int guiding_region_id(const GuideGPU& G,const float3& P){
  float3 rel=make_float3((P.x-G.grid_min.x)/G.cell_size.x,
                         (P.y-G.grid_min.y)/G.cell_size.y,
                         (P.z-G.grid_min.z)/G.cell_size.z);
  int3 c;
  c.x=max(0,min(int(rel.x),G.grid_res.x-1));
  c.y=max(0,min(int(rel.y),G.grid_res.y-1));
  c.z=max(0,min(int(rel.z),G.grid_res.z-1));
  uint32_t id=((uint32_t*)G.d_grid)[(c.z*G.grid_res.y+c.y)*G.grid_res.x+c.x];
  return id==0xFFFFFFFFu?-1:int(id);
}

__device__ inline float3 vmf_sample(const float3& mu,float /*kappa*/,float2 /*u*/){
  return mu;
}
__device__ inline float vmf_pdf(const float3& /*mu*/,float /*kappa*/,const float3& /*wi*/){
  return 1.f/(4.f*CUDART_PI_F);
}
