# AGENTS Instructions

- Run `cargo fmt --all --check` before committing changes.
- Do **not** run `cargo build`. The build depends on NVIDIA's CUDA toolkit (`nvcc`) and the OptiX SDK, which are typically unavailable in this environment and costly to install.
- Instead, ask the user to verify that the project builds on their machine and note any issues in the PR.
