#pragma once
#include <cuda.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <stdint.h>
#include <math_constants.h>

struct GuideRegion { float3 bmin,bmax; uint32_t lobe_ofs,lobe_num; };
struct GuideLobe   { float3 mu; float kappa; float weight; float3 rgb; };
struct GuideGPU {
  CUdeviceptr d_regions; uint32_t region_count;
  CUdeviceptr d_lobes;   uint32_t lobe_count;
  CUdeviceptr d_grid;    int3 grid_res; float3 grid_min,grid_max,cell_size;
};

struct TrainSample {
  float3 position;
  float3 dir_in;
  float3 contrib;
  uint32_t is_delta;
};

extern __device__ TrainSample* g_train_samples;
extern __device__ uint32_t*    g_train_write_idx;
extern __device__ uint32_t     g_train_sample_capacity;

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

__device__ inline float dot3(const float3& a,const float3& b){
  return a.x*b.x+a.y*b.y+a.z*b.z;
}
__device__ inline float3 cross3(const float3& a,const float3& b){
  return make_float3(a.y*b.z-a.z*b.y,
                     a.z*b.x-a.x*b.z,
                     a.x*b.y-a.y*b.x);
}
__device__ inline float3 normalize3(const float3& v){
  float len=sqrtf(dot3(v,v));
  return len>0.f?v*(1.f/len):make_float3(0.f,0.f,0.f);
}

__device__ inline float3 vmf_sample(const float3& mu,float kappa,float2 u){
  float cos_theta;
  if(kappa<=1e-4f){
    cos_theta=1.f-2.f*u.x;
  }else{
    float term=expf(-2.f*kappa);
    cos_theta=1.f+logf(term+(1.f-term)*u.x)/kappa;
  }
  float sin_theta=sqrtf(fmaxf(0.f,1.f-cos_theta*cos_theta));
  float phi=2.f*CUDART_PI_F*u.y;
  float3 local=make_float3(cosf(phi)*sin_theta,sinf(phi)*sin_theta,cos_theta);
  float3 w=normalize3(mu);
  float3 up=fabsf(w.z)<0.9999999f?make_float3(0.f,0.f,1.f):make_float3(1.f,0.f,0.f);
  float3 u_axis=normalize3(cross3(up,w));
  float3 v_axis=cross3(w,u_axis);
  return make_float3(u_axis.x*local.x+v_axis.x*local.y+w.x*local.z,
                     u_axis.y*local.x+v_axis.y*local.y+w.y*local.z,
                     u_axis.z*local.x+v_axis.z*local.y+w.z*local.z);
}

__device__ inline float vmf_pdf(const float3& mu,float kappa,const float3& wi){
  if(kappa<=1e-4f) return 1.f/(4.f*CUDART_PI_F);
  float norm=kappa/(4.f*CUDART_PI_F*sinhf(kappa));
  return norm*expf(kappa*dot3(mu,wi));
}

__device__ inline int guiding_choose_lobe(const GuideRegion& R,const GuideLobe* L,float u){
  float sum=0.f; for(uint32_t i=0;i<R.lobe_num;++i) sum+=L[i].weight;
  if(sum<=0.f) return -1;
  float r=u*sum,acc=0.f;
  for(uint32_t i=0;i<R.lobe_num;++i){acc+=L[i].weight; if(r<=acc) return int(i);}return int(R.lobe_num-1);
}

__device__ inline float3 guiding_sample_lobe(const GuideLobe& l,float2 u){
  return vmf_sample(l.mu,l.kappa,u);
}

__device__ inline float guiding_mixture_pdf(const GuideRegion& R,const GuideLobe* L,const float3& wi){
  float sum=0.f,pdf=0.f;
  for(uint32_t i=0;i<R.lobe_num;++i){
    float w=L[i].weight; sum+=w; pdf+=w*vmf_pdf(L[i].mu,L[i].kappa,wi);
  }
  return sum>0.f?pdf/sum:0.f;
}
