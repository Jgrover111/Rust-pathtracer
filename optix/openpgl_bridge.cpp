#include "openpgl_bridge.h"

#include <algorithm>
#include <vector>
#include <limits>

#include <openpgl/cpp/Device.h>
#include <openpgl/cpp/Field.h>
#include <openpgl/cpp/FieldConfig.h>
#include <openpgl/cpp/Region.h>
#include <openpgl/cpp/Distribution.h>
#include <openpgl/cpp/SampleData.h>
#include <openpgl/cpp/SampleStorage.h>

using namespace openpgl::cpp;

struct SampleStorageRGB {
    SampleStorage storage;
    std::vector<pgl_vec3f> colors;
};

static std::vector<pgl_vec3f> g_lobe_rgb;

pgl_dev_t pgl_dev_create(int num_threads) {
    return new Device(PGL_DEVICE_TYPE_CPU_4, static_cast<size_t>(num_threads));
}

void pgl_dev_destroy(pgl_dev_t dev) { delete static_cast<Device *>(dev); }

pgl_field_t pgl_field_create(pgl_dev_t dev, const float world_min[3], const float world_max[3]) {
    Device *device = static_cast<Device *>(dev);
    FieldConfig cfg;
    cfg.Init(PGL_SPATIAL_STRUCTURE_KDTREE, PGL_DIRECTIONAL_DISTRIBUTION_VMM, true);
    Field *field = new Field(device, cfg);
    pgl_box3f bounds;
    pglBox3f(bounds, world_min[0], world_min[1], world_min[2], world_max[0], world_max[1], world_max[2]);
    field->SetSceneBounds(bounds);
    return field;
}

void pgl_field_destroy(pgl_field_t field) { delete static_cast<Field *>(field); }

pgl_samples_t pgl_samples_create() { return new SampleStorageRGB; }

void pgl_samples_destroy(pgl_samples_t s) { delete static_cast<SampleStorageRGB *>(s); }

static void add_sample(SampleStorageRGB *s, const float pos[3], const float dir_in[3], const float weight_rgb[3], uint32_t flags) {
    SampleData sample{};
    pglPoint3f(sample.position, pos[0], pos[1], pos[2]);
    sample.direction = {dir_in[0], dir_in[1], dir_in[2]};
    sample.weight = (weight_rgb[0] + weight_rgb[1] + weight_rgb[2]) / 3.0f;
    sample.pdf = 1.0f;
    sample.distance = 0.0f;
    sample.flags = flags;
    s->storage.AddSample(sample);
    s->colors.push_back({weight_rgb[0], weight_rgb[1], weight_rgb[2]});
}

void pgl_samples_add_surface(pgl_samples_t handle, const float pos[3], const float dir_in[3], const float weight_rgb[3], int is_delta) {
    auto *s = static_cast<SampleStorageRGB *>(handle);
    add_sample(s, pos, dir_in, weight_rgb, is_delta ? PGLSampleData::EDirectLight : 0u);
}

void pgl_samples_add_volume(pgl_samples_t handle, const float pos[3], const float dir_in[3], const float weight_rgb[3]) {
    auto *s = static_cast<SampleStorageRGB *>(handle);
    add_sample(s, pos, dir_in, weight_rgb, PGLSampleData::EInsideVolume);
}

void pgl_samples_add_direct(pgl_samples_t handle, const float pos[3], const float dir_in[3], const float weight_rgb[3]) {
    auto *s = static_cast<SampleStorageRGB *>(handle);
    add_sample(s, pos, dir_in, weight_rgb, PGLSampleData::EDirectLight);
}

void pgl_field_update(pgl_field_t field_handle, pgl_samples_t samples_handle) {
    auto *field = static_cast<Field *>(field_handle);
    auto *samples = static_cast<SampleStorageRGB *>(samples_handle);
    const size_t regionCount = field->GetRegionCount();
    std::vector<PGLRegion> regionHandles(regionCount);
    field->GetRegions(regionHandles.data());

    struct RegionInfo {
        pgl_box3f bounds;
        uint32_t lobe_ofs;
        std::vector<PGLVMMLobe> lobes;
    };

    std::vector<RegionInfo> regions(regionCount);
    uint32_t lobeOffset = 0;
    for (size_t i = 0; i < regionCount; ++i) {
        RegionInfo info;
        info.bounds = pglRegionGetBounds(regionHandles[i]);
        const size_t lobeCount = field->GetRegionLobeCount(regionHandles[i]);
        info.lobe_ofs = lobeOffset;
        info.lobes.resize(lobeCount);
        field->GetRegionLobes(regionHandles[i], info.lobes.data());
        regions[i] = info;
        lobeOffset += static_cast<uint32_t>(lobeCount);
    }

    g_lobe_rgb.assign(lobeOffset, {0.f, 0.f, 0.f});

    auto inside = [](const pgl_box3f &b, const pgl_point3f &p) {
        return p.v[0] >= b.lower.v[0] && p.v[0] <= b.upper.v[0] &&
               p.v[1] >= b.lower.v[1] && p.v[1] <= b.upper.v[1] &&
               p.v[2] >= b.lower.v[2] && p.v[2] <= b.upper.v[2];
    };

    size_t colorIdx = 0;
    const size_t sampleCountSurf = samples->storage.GetSizeSurface();
    for (size_t i = 0; i < sampleCountSurf; ++i) {
        SampleData sd = samples->storage.GetSampleSurface(static_cast<int>(i));
        const pgl_vec3f &col = samples->colors[colorIdx++];
        for (const auto &reg : regions) {
            if (!inside(reg.bounds, sd.position))
                continue;
            float best = -std::numeric_limits<float>::infinity();
            uint32_t bestIdx = 0;
            for (uint32_t j = 0; j < reg.lobes.size(); ++j) {
                const PGLVMMLobe &l = reg.lobes[j];
                float d = sd.direction.v[0] * l.mu.v[0] +
                          sd.direction.v[1] * l.mu.v[1] +
                          sd.direction.v[2] * l.mu.v[2];
                if (d > best) {
                    best = d;
                    bestIdx = j;
                }
            }
            pgl_vec3f &dst = g_lobe_rgb[reg.lobe_ofs + bestIdx];
            dst.v[0] += col.v[0];
            dst.v[1] += col.v[1];
            dst.v[2] += col.v[2];
            break;
        }
    }
    const size_t sampleCountVol = samples->storage.GetSizeVolume();
    for (size_t i = 0; i < sampleCountVol; ++i) {
        SampleData sd = samples->storage.GetSampleVolume(static_cast<int>(i));
        const pgl_vec3f &col = samples->colors[colorIdx++];
        for (const auto &reg : regions) {
            if (!inside(reg.bounds, sd.position))
                continue;
            float best = -std::numeric_limits<float>::infinity();
            uint32_t bestIdx = 0;
            for (uint32_t j = 0; j < reg.lobes.size(); ++j) {
                const PGLVMMLobe &l = reg.lobes[j];
                float d = sd.direction.v[0] * l.mu.v[0] +
                          sd.direction.v[1] * l.mu.v[1] +
                          sd.direction.v[2] * l.mu.v[2];
                if (d > best) {
                    best = d;
                    bestIdx = j;
                }
            }
            pgl_vec3f &dst = g_lobe_rgb[reg.lobe_ofs + bestIdx];
            dst.v[0] += col.v[0];
            dst.v[1] += col.v[1];
            dst.v[2] += col.v[2];
            break;
        }
    }

    field->Update(samples->storage);
    samples->storage.ClearSurface();
    samples->storage.ClearVolume();
    samples->colors.clear();
}

void pgl_field_snapshot(pgl_field_t field_handle,
                        const pgl_region **regions, uint32_t *region_count,
                        const pgl_lobe **lobes, uint32_t *lobe_count) {
    static std::vector<pgl_region> reg_vec;
    static std::vector<pgl_lobe> lobe_vec;

    reg_vec.clear();
    lobe_vec.clear();

    auto *field = static_cast<Field *>(field_handle);
    if (field) {
        const size_t regionCount = field->GetRegionCount();
        std::vector<PGLRegion> regionHandles(regionCount);
        field->GetRegions(regionHandles.data());

        uint32_t lobeOffset = 0;
        for (size_t i = 0; i < regionHandles.size(); ++i) {
            PGLRegion regHandle = regionHandles[i];
            pgl_box3f bounds = pglRegionGetBounds(regHandle);

            pgl_region reg{};
            reg.bmin[0] = bounds.lower.v[0];
            reg.bmin[1] = bounds.lower.v[1];
            reg.bmin[2] = bounds.lower.v[2];
            reg.bmax[0] = bounds.upper.v[0];
            reg.bmax[1] = bounds.upper.v[1];
            reg.bmax[2] = bounds.upper.v[2];
            reg.lobe_ofs = lobeOffset;

            const size_t lobeCountRegion = field->GetRegionLobeCount(regHandle);
            reg.lobe_num = static_cast<uint32_t>(lobeCountRegion);

            std::vector<PGLVMMLobe> lobesSrc(lobeCountRegion);
            field->GetRegionLobes(regHandle, lobesSrc.data());

            for (size_t j = 0; j < lobesSrc.size(); ++j) {
                const PGLVMMLobe &src = lobesSrc[j];
                pgl_lobe l{};
                l.mu[0] = src.mu.v[0];
                l.mu[1] = src.mu.v[1];
                l.mu[2] = src.mu.v[2];
                l.kappa = src.kappa;
                l.weight = src.weight;
                const size_t idx = lobeOffset + j;
                if (idx < g_lobe_rgb.size()) {
                    l.rgb[0] = g_lobe_rgb[idx].v[0];
                    l.rgb[1] = g_lobe_rgb[idx].v[1];
                    l.rgb[2] = g_lobe_rgb[idx].v[2];
                } else {
                    l.rgb[0] = l.rgb[1] = l.rgb[2] = src.weight;
                }
                lobe_vec.push_back(l);
            }

            lobeOffset += static_cast<uint32_t>(lobeCountRegion);
            reg_vec.push_back(reg);
        }
    }

    if (regions)
        *regions = reg_vec.data();
    if (region_count)
        *region_count = static_cast<uint32_t>(reg_vec.size());
    if (lobes)
        *lobes = lobe_vec.data();
    if (lobe_count)
        *lobe_count = static_cast<uint32_t>(lobe_vec.size());
}

