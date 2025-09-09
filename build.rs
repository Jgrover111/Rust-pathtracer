use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=optix/CMakeLists.txt");
    println!("cargo:rerun-if-changed=optix/kernels.cu");
    println!("cargo:rerun-if-changed=optix/optix_wrapper.cpp");
    println!("cargo:rerun-if-changed=optix/optix_device.cu");
    println!("cargo:rerun-if-changed=optix/cuda_fixups.cuh");

    let mut cfg = cmake::Config::new("optix");

    // Match cargo profile for clearer builds on Windows
    let build_type = if env::var("PROFILE").unwrap_or_default() == "release" {
        "Release"
    } else {
        "Debug"
    };
    cfg.define("CMAKE_BUILD_TYPE", build_type);

    let dst = cfg.build();
    let bin_dir = dst.join("bin");
    let lib_dir = dst.join("lib");

    // Tell rustc where to find the produced library
    println!("cargo:rustc-link-search=native={}", bin_dir.display());
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    // Link the import library if present (static link name without 'lib'/'dll' prefixes)
    println!("cargo:rustc-link-lib=dylib=optix_wrapper");

    // Copy the .dll next to our executable for runtime loading on Windows.
    #[cfg(windows)]
    {
        let profile = env::var("PROFILE").unwrap();
        let target_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("target")
            .join(&profile);

        let candidates = [
            bin_dir.join("optix_wrapper.dll"),
            lib_dir.join("optix_wrapper.dll"),
        ];

        if let Some(dll) = candidates.iter().find(|p| p.exists()) {
            let dst_dll = target_dir.join("optix_wrapper.dll");
            let _ = fs::create_dir_all(&target_dir);
            if fs::copy(dll, &dst_dll).is_ok() {
                println!(
                    "cargo:warning=my_pathtracer@{}: Copied {} -> {}",
                    env::var("CARGO_PKG_VERSION").unwrap_or_default(),
                    dll.display(),
                    dst_dll.display()
                );
            }
        }
    }
}
