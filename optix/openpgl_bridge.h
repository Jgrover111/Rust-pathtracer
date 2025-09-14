#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef void* pgl_dev_t;
typedef void* pgl_field_t;
typedef void* pgl_samples_t;

pgl_dev_t   pgl_dev_create(int num_threads);
void        pgl_dev_destroy(pgl_dev_t);

pgl_field_t pgl_field_create(pgl_dev_t dev, const float world_min[3], const float world_max[3]);
void        pgl_field_destroy(pgl_field_t);

pgl_samples_t pgl_samples_create();
void          pgl_samples_destroy(pgl_samples_t);

void pgl_samples_add_surface(pgl_samples_t s,
   const float pos[3],
   const float dir_in[3],
   const float weight_rgb[3],
   int is_delta);

void pgl_field_update(pgl_field_t, pgl_samples_t);

struct pgl_region { float bmin[3]; float bmax[3]; uint32_t lobe_ofs; uint32_t lobe_num; };
struct pgl_lobe   { float mu[3]; float kappa; float weight; float rgb[3]; };

void pgl_field_snapshot(pgl_field_t,
   const struct pgl_region** regions, uint32_t* region_count,
   const struct pgl_lobe**   lobes,   uint32_t* lobe_count);

#ifdef __cplusplus
}
#endif
