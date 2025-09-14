#pragma once
#include "openpgl_bridge.h"
#include <cuda.h>
#include <vector_types.h>
#include <stdint.h>

struct GuideRegion {
  float3 bmin, bmax;
  uint32_t lobe_ofs, lobe_num;
};
struct GuideLobe {
  float3 mu;
  float kappa;
  float weight;
  float3 rgb;  // accumulated color for this lobe
};

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

#ifdef __cplusplus
extern "C" {
#endif

void guiding_upload_snapshot(const struct pgl_region* regions,uint32_t n_regions,
                             const struct pgl_lobe* lobes,uint32_t n_lobes);
void guiding_set_enabled(int enabled);
void guiding_set_train_buffer_size(uint32_t n);
void guiding_map_train_samples(const TrainSample** samples, uint32_t* count);
void guiding_reset_train_write_idx();
void guiding_set_grid_res(int x,int y,int z);

#ifdef __cplusplus
}
#endif
