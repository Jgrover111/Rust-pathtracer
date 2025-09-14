#include "openpgl_bridge.h"
#include <vector>
struct DummyStorage {
  std::vector<int> dummy;
};

pgl_dev_t pgl_dev_create(int){return nullptr;}
void pgl_dev_destroy(pgl_dev_t){}

pgl_field_t pgl_field_create(pgl_dev_t,const float[3],const float[3]){return nullptr;}
void pgl_field_destroy(pgl_field_t){}

pgl_samples_t pgl_samples_create(){return new DummyStorage;}
void pgl_samples_destroy(pgl_samples_t s){delete static_cast<DummyStorage*>(s);}

void pgl_samples_add_surface(pgl_samples_t,const float[3],const float[3],const float[3],int){}

void pgl_field_update(pgl_field_t,pgl_samples_t){}

void pgl_field_snapshot(pgl_field_t,const pgl_region** regions,uint32_t* region_count,const pgl_lobe** lobes,uint32_t* lobe_count){
  static pgl_region empty_region{};
  static pgl_lobe   empty_lobe{};
  if(regions) *regions=&empty_region;
  if(region_count) *region_count=0;
  if(lobes) *lobes=&empty_lobe;
  if(lobe_count) *lobe_count=0;
}
