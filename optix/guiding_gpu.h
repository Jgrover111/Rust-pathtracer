#pragma once
#include "openpgl_bridge.h"
#include <cuda.h>
#include <vector_types.h>
#include <stdint.h>

struct GuideRegion { float3 bmin,bmax; uint32_t lobe_ofs,lobe_num; };
struct GuideLobe   { float3 mu; float kappa; float weight; float3 rgb; };

struct GuideGPU {
  CUdeviceptr d_regions; uint32_t region_count;
  CUdeviceptr d_lobes;   uint32_t lobe_count;
  CUdeviceptr d_grid;    int3 grid_res; float3 grid_min,grid_max,cell_size;
};

#ifdef __cplusplus
extern "C" {
#endif

void guiding_upload_snapshot(const struct pgl_region* regions,uint32_t n_regions,
                             const struct pgl_lobe* lobes,uint32_t n_lobes);
void guiding_set_enabled(int enabled);

#ifdef __cplusplus
}
#endif
