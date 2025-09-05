#pragma once

// Make sure we're compiling with NVCC for CUDA types.
#ifdef __CUDACC__
  #include <cuda_runtime.h>
  #include <vector_types.h>
#endif

// Provide C stdio symbols (fprintf, stderr) used by host-side CUDA error checks.
#include <stdio.h>

// ---- float3 helpers ----
// Add component-wise division overloads that NVCC/MSVC don't provide by default.
// These are 'inline' to avoid ODR/link conflicts and safe to include everywhere.

#ifndef CUDA_FIXUPS_FLOAT3_OPS
#define CUDA_FIXUPS_FLOAT3_OPS

// float3 / float3  -> component-wise division
__host__ __device__ inline float3 operator/(const float3& a, const float3& b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

// float / float3   -> component-wise reciprocal scaled by scalar
__host__ __device__ inline float3 operator/(float a, const float3& b) {
    return make_float3(a / b.x, a / b.y, a / b.z);
}

#endif // CUDA_FIXUPS_FLOAT3_OPS
