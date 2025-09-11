#include <stdint.h>

extern "C" __global__ void compact_active(const uint32_t* active_in,
                                           const uint32_t* flags,
                                           uint32_t* active_out,
                                           uint32_t* out_count,
                                           uint32_t count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    uint32_t px = active_in[idx];
    if (flags[px]) {
        uint32_t dst = atomicAdd(out_count, 1u);
        active_out[dst] = px;
    }
}

extern "C" __global__ void clear_noise(float* noise_map,
                                      const uint32_t* active,
                                      uint32_t count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    noise_map[ active[idx] ] = 0.f;
}