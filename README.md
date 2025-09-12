# Rust Path Tracer

This project is a GPU-accelerated path tracer written in Rust. It renders a simple scene and exports both tone-mapped PNG images and HDR AVIF variants.

## Prerequisites

- Rust toolchain (edition 2021).
- NVIDIA CUDA Toolkit and drivers. The build script uses `cmake` and `nvcc` to compile OptiX kernels.
- NVIDIA OptiX SDK.

## Building

```bash
cargo build
```

> **Note:** Building requires the CUDA toolkit. If the toolkit is not installed the build will fail during the `cmake` step.

## Running

Running the executable renders the scene and writes several images into the project root:

```bash
cargo run --release
```

## Outputs

| File | Description |
|------|-------------|
| `pt.exr` | Path traced RGB output (linear OpenEXR) |
| `pt_bayer.exr` | Result after demosaicing the Bayer render (linear OpenEXR) |

## License

This project is licensed under the terms specified in `LICENSE` (if present).
