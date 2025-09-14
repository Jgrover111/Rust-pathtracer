#[cfg(feature = "guiding")]
mod guiding;

use exr::prelude::write_rgb_file;
use std::ffi::c_int;
use std::time::Instant;

// ---- FFI ----------------------------------------------------------
extern "C" {
    fn optix_render_rgb(
        w: c_int,
        h: c_int,
        spp: c_int,
        max_depth: c_int,
        noise_threshold: f32,
        min_adaptive_samples: c_int,
        out_rgb: *mut f32,
    ) -> c_int;
    fn optix_render_bayer_f32(
        w: c_int,
        h: c_int,
        spp: c_int,
        max_depth: c_int,
        noise_threshold: f32,
        min_adaptive_samples: c_int,
        pattern: c_int,
        out_raw: *mut f32,
    ) -> c_int;
    fn optix_alloc_host(bytes: usize) -> *mut std::ffi::c_void;
    fn optix_free_host(ptr: *mut std::ffi::c_void);
    fn optix_synchronize();
}

// Stable shims used by the rest of main():
/// Render a full RGB image using the GPU backend.
///
/// # Parameters
/// - `w`: Image width in pixels. Must be positive.
/// - `h`: Image height in pixels. Must be positive.
/// - `spp`: Samples per pixel. Must be positive.
/// - `min_adaptive_samples`: Minimum samples per pixel before adaptive termination.
/// - `out`: Pointer to a buffer of at least `w * h * 3` `f32` values.
///
/// # Returns
/// Zero on success, non-zero on failure.
#[inline]
fn ffi_render_rgb(
    w: i32,
    h: i32,
    spp: i32,
    max_depth: i32,
    noise_threshold: f32,
    min_adaptive_samples: i32,
    out: *mut f32,
) -> c_int {
    unsafe {
        optix_render_rgb(
            w,
            h,
            spp,
            max_depth,
            noise_threshold,
            min_adaptive_samples,
            out,
        )
    }
}
/// Render a Bayer mosaic image using the GPU backend.
///
/// # Parameters
/// - `w`: Image width in pixels. Must be positive.
/// - `h`: Image height in pixels. Must be positive.
/// - `spp`: Samples per pixel. Must be positive.
/// - `pat`: Bayer pattern index.
/// - `min_adaptive_samples`: Minimum samples per pixel before adaptive termination.
/// - `out`: Pointer to a buffer of at least `w * h` `f32` values.
///
/// # Returns
/// Zero on success, non-zero on failure.
#[inline]
fn ffi_render_bayer_f32(
    w: i32,
    h: i32,
    spp: i32,
    max_depth: i32,
    noise_threshold: f32,
    min_adaptive_samples: i32,
    pat: i32,
    out: *mut f32,
) -> c_int {
    unsafe {
        optix_render_bayer_f32(
            w,
            h,
            spp,
            max_depth,
            noise_threshold,
            min_adaptive_samples,
            pat,
            out,
        )
    }
}
/// Allocate pinned host memory using the GPU API.
///
/// # Safety
/// The caller must ensure the returned pointer is freed with [`ffi_free_host`].
#[inline]
unsafe fn ffi_alloc_host(bytes: usize) -> *mut std::ffi::c_void {
    optix_alloc_host(bytes)
}
/// Free memory previously allocated with [`ffi_alloc_host`].
///
/// # Safety
/// `ptr` must originate from [`ffi_alloc_host`] and not be used afterwards.
#[inline]
unsafe fn ffi_free_host(ptr: *mut std::ffi::c_void) {
    optix_free_host(ptr)
}
/// Block until all GPU work on the default stream has completed.
#[inline]
unsafe fn ffi_stream_sync() {
    optix_synchronize();
}

// ---- utils, demosaic --------------------------------------------
/// Convert 2D coordinates into a linear index.
///
/// `w` is the image width in pixels. This function performs no bounds checking.
fn idx(x: i32, y: i32, w: i32) -> usize {
    (y as usize) * (w as usize) + (x as usize)
}
/// Clamp `v` into the inclusive range `[lo, hi]`.
#[inline]
fn clamp(v: i32, lo: i32, hi: i32) -> i32 {
    v.max(lo).min(hi)
}
/// Color in a Bayer color filter array.
#[derive(Copy, Clone, PartialEq)]
enum CFA {
    /// Red pixel
    R,
    /// Green pixel
    G,
    /// Blue pixel
    B,
}

/// Return the colour filter at a given pixel for the specified Bayer pattern.
///
/// Pattern indices map to 2×2 Bayer arrangements as follows:
///
/// * `0`: `RGGB`
/// * `1`: `BGGR`
/// * `2`: `GRBG`
/// * `3`: `GBRG`
fn cfa_at(x: i32, y: i32, pattern: i32) -> CFA {
    // 2×2 lookup tables for each Bayer configuration.
    // The table layout is [row][column].
    const PATTERNS: [[[CFA; 2]; 2]; 4] = [
        // 0: RGGB
        [[CFA::R, CFA::G], [CFA::G, CFA::B]],
        // 1: BGGR
        [[CFA::B, CFA::G], [CFA::G, CFA::R]],
        // 2: GRBG
        [[CFA::G, CFA::R], [CFA::B, CFA::G]],
        // 3: GBRG
        [[CFA::G, CFA::B], [CFA::R, CFA::G]],
    ];

    let p = pattern.max(0).min(3) as usize;
    PATTERNS[p][(y & 1) as usize][(x & 1) as usize]
}
/// Read a value from `img` clamping coordinates to the image bounds.
fn getf(img: &[f32], w: i32, h: i32, x: i32, y: i32) -> f32 {
    let xx = clamp(x, 0, w - 1);
    let yy = clamp(y, 0, h - 1);
    img[idx(xx, yy, w)]
}
/// Write `v` into `img` clamping coordinates to the image bounds.
fn setf(img: &mut [f32], w: i32, h: i32, x: i32, y: i32, v: f32) {
    let xx = clamp(x, 0, w - 1);
    let yy = clamp(y, 0, h - 1);
    img[idx(xx, yy, w)] = v;
}
/// Demosaic a Bayer image using the AMaZE algorithm.
///
/// `raw` must contain `w * h` samples in row-major order. `pattern` selects the Bayer pattern
/// (0-3).
fn demosaic_amaze(raw: &[f32], w: i32, h: i32, pattern: i32) -> Vec<f32> {
    let n = (w * h) as usize;
    let mut r = vec![0f32; n];
    let mut g = vec![0f32; n];
    let mut b = vec![0f32; n];
    for y in 0..h {
        for x in 0..w {
            let v = raw[idx(x, y, w)];
            match cfa_at(x, y, pattern) {
                CFA::R => r[idx(x, y, w)] = v,
                CFA::G => g[idx(x, y, w)] = v,
                CFA::B => b[idx(x, y, w)] = v,
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            let c = cfa_at(x, y, pattern);
            if c == CFA::R || c == CFA::B {
                let cx = if c == CFA::R { &r } else { &b };
                let c0 = getf(cx, w, h, x, y);
                let c_l2 = getf(cx, w, h, x - 2, y);
                let c_l1 = getf(cx, w, h, x - 1, y);
                let c_r1 = getf(cx, w, h, x + 1, y);
                let c_r2 = getf(cx, w, h, x + 2, y);
                let c_u2 = getf(cx, w, h, x, y - 2);
                let c_u1 = getf(cx, w, h, x, y - 1);
                let c_d1 = getf(cx, w, h, x, y + 1);
                let c_d2 = getf(cx, w, h, x, y + 2);
                let g_l1 = getf(&g, w, h, x - 1, y);
                let g_r1 = getf(&g, w, h, x + 1, y);
                let g_u1 = getf(&g, w, h, x, y - 1);
                let g_d1 = getf(&g, w, h, x, y + 1);
                let grad_h = (g_l1 - g_r1).abs()
                    + (2.0 * c0 - c_l2 - c_r2).abs() * 0.5
                    + (c_l1 - c_r1).abs();
                let grad_v = (g_u1 - g_d1).abs()
                    + (2.0 * c0 - c_u2 - c_d2).abs() * 0.5
                    + (c_u1 - c_d1).abs();
                let gh = 0.5 * (g_l1 + g_r1) + 0.25 * (2.0 * c0 - c_l2 - c_r2);
                let gv = 0.5 * (g_u1 + g_d1) + 0.25 * (2.0 * c0 - c_u2 - c_d2);
                let wh = 1.0 / (1e-6 + grad_h * grad_h);
                let wv = 1.0 / (1e-6 + grad_v * grad_v);
                let g_est = (wh * gh + wv * gv) / (wh + wv);
                setf(&mut g, w, h, x, y, g_est);
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            if matches!(cfa_at(x, y, pattern), CFA::G) {
                let g0 = getf(&g, w, h, x, y);
                let has_red_h = matches!(cfa_at(x - 1, y, pattern), CFA::R)
                    || matches!(cfa_at(x + 1, y, pattern), CFA::R);
                let has_red_v = matches!(cfa_at(x, y - 1, pattern), CFA::R)
                    || matches!(cfa_at(x, y + 1, pattern), CFA::R);
                let has_blue_h = matches!(cfa_at(x - 1, y, pattern), CFA::B)
                    || matches!(cfa_at(x + 1, y, pattern), CFA::B);
                let has_blue_v = matches!(cfa_at(x, y - 1, pattern), CFA::B)
                    || matches!(cfa_at(x, y + 1, pattern), CFA::B);
                let r_est = if has_red_h {
                    0.5 * (getf(&r, w, h, x - 1, y) + getf(&r, w, h, x + 1, y))
                        - 0.5 * (getf(&g, w, h, x - 1, y) + getf(&g, w, h, x + 1, y))
                        + g0
                } else if has_red_v {
                    0.5 * (getf(&r, w, h, x, y - 1) + getf(&r, w, h, x, y + 1))
                        - 0.5 * (getf(&g, w, h, x, y - 1) + getf(&g, w, h, x, y + 1))
                        + g0
                } else {
                    g0
                };
                let b_est = if has_blue_h {
                    0.5 * (getf(&b, w, h, x - 1, y) + getf(&b, w, h, x + 1, y))
                        - 0.5 * (getf(&g, w, h, x - 1, y) + getf(&g, w, h, x + 1, y))
                        + g0
                } else if has_blue_v {
                    0.5 * (getf(&b, w, h, x, y - 1) + getf(&b, w, h, x, y + 1))
                        - 0.5 * (getf(&g, w, h, x, y - 1) + getf(&g, w, h, x, y + 1))
                        + g0
                } else {
                    g0
                };
                setf(&mut r, w, h, x, y, r_est);
                setf(&mut b, w, h, x, y, b_est);
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            match cfa_at(x, y, pattern) {
                CFA::R => {
                    let g0 = getf(&g, w, h, x, y);
                    let b_est = 0.25
                        * ((getf(&b, w, h, x - 1, y - 1) - getf(&g, w, h, x - 1, y - 1))
                            + (getf(&b, w, h, x + 1, y - 1) - getf(&g, w, h, x + 1, y - 1))
                            + (getf(&b, w, h, x - 1, y + 1) - getf(&g, w, h, x - 1, y + 1))
                            + (getf(&b, w, h, x + 1, y + 1) - getf(&g, w, h, x + 1, y + 1)))
                        + g0;
                    setf(&mut b, w, h, x, y, b_est);
                }
                CFA::B => {
                    let g0 = getf(&g, w, h, x, y);
                    let r_est = 0.25
                        * ((getf(&r, w, h, x - 1, y - 1) - getf(&g, w, h, x - 1, y - 1))
                            + (getf(&r, w, h, x + 1, y - 1) - getf(&g, w, h, x + 1, y - 1))
                            + (getf(&r, w, h, x - 1, y + 1) - getf(&g, w, h, x - 1, y + 1))
                            + (getf(&r, w, h, x + 1, y + 1) - getf(&g, w, h, x + 1, y + 1)))
                        + g0;
                    setf(&mut r, w, h, x, y, r_est);
                }
                _ => {}
            }
        }
    }
    let mut rg = vec![0.0; n];
    let mut bg = vec![0.0; n];
    for i in 0..n {
        rg[i] = r[i] - g[i];
        bg[i] = b[i] - g[i];
    }
    // Allocate chroma difference buffers without cloning to avoid extra copies
    let mut rg2 = vec![0.0; n];
    let mut bg2 = vec![0.0; n];
    for y in 0..h {
        for x in 0..w {
            let gh = (getf(&g, w, h, x - 1, y) - getf(&g, w, h, x + 1, y)).abs();
            let gv = (getf(&g, w, h, x, y - 1) - getf(&g, w, h, x, y + 1)).abs();
            let w_h = 1.0 / (1e-6 + gh);
            let w_v = 1.0 / (1e-6 + gv);
            let rgh = 0.5 * (getf(&rg, w, h, x - 1, y) + getf(&rg, w, h, x + 1, y));
            let rgv = 0.5 * (getf(&rg, w, h, x, y - 1) + getf(&rg, w, h, x, y + 1));
            let bgh = 0.5 * (getf(&bg, w, h, x - 1, y) + getf(&bg, w, h, x + 1, y));
            let bgv = 0.5 * (getf(&bg, w, h, x, y - 1) + getf(&bg, w, h, x, y + 1));
            setf(&mut rg2, w, h, x, y, (w_h * rgh + w_v * rgv) / (w_h + w_v));
            setf(&mut bg2, w, h, x, y, (w_h * bgh + w_v * bgv) / (w_h + w_v));
        }
    }
    for i in 0..n {
        // Preserve highlights by avoiding upper clipping during reconstruction
        r[i] = (g[i] + rg2[i]).max(0.0);
        b[i] = (g[i] + bg2[i]).max(0.0);
    }
    let mut out = vec![0f32; n * 3];
    for y in 0..h {
        for x in 0..w {
            let i = idx(x, y, w);
            out[i * 3 + 0] = r[i].max(0.0);
            out[i * 3 + 1] = g[i].max(0.0);
            out[i * 3 + 2] = b[i].max(0.0);
        }
    }
    out
}
/// Demosaic a Bayer image using simple bilinear interpolation.
fn demosaic_bilinear(raw: &[f32], w: i32, h: i32, pattern: i32) -> Vec<f32> {
    let mut out = vec![0f32; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = idx(x, y, w);
            let v = raw[i];
            match cfa_at(x, y, pattern) {
                CFA::R => {
                    let g = (getf(raw, w, h, x - 1, y)
                        + getf(raw, w, h, x + 1, y)
                        + getf(raw, w, h, x, y - 1)
                        + getf(raw, w, h, x, y + 1))
                        * 0.25;
                    let b = (getf(raw, w, h, x - 1, y - 1)
                        + getf(raw, w, h, x + 1, y - 1)
                        + getf(raw, w, h, x - 1, y + 1)
                        + getf(raw, w, h, x + 1, y + 1))
                        * 0.25;
                    out[i * 3] = v;
                    out[i * 3 + 1] = g;
                    out[i * 3 + 2] = b;
                }
                CFA::B => {
                    let g = (getf(raw, w, h, x - 1, y)
                        + getf(raw, w, h, x + 1, y)
                        + getf(raw, w, h, x, y - 1)
                        + getf(raw, w, h, x, y + 1))
                        * 0.25;
                    let r = (getf(raw, w, h, x - 1, y - 1)
                        + getf(raw, w, h, x + 1, y - 1)
                        + getf(raw, w, h, x - 1, y + 1)
                        + getf(raw, w, h, x + 1, y + 1))
                        * 0.25;
                    out[i * 3] = r;
                    out[i * 3 + 1] = g;
                    out[i * 3 + 2] = v;
                }
                CFA::G => {
                    let r;
                    let b;
                    if matches!(cfa_at(x - 1, y, pattern), CFA::R)
                        || matches!(cfa_at(x + 1, y, pattern), CFA::R)
                    {
                        r = (getf(raw, w, h, x - 1, y) + getf(raw, w, h, x + 1, y)) * 0.5;
                        b = (getf(raw, w, h, x, y - 1) + getf(raw, w, h, x, y + 1)) * 0.5;
                    } else {
                        r = (getf(raw, w, h, x, y - 1) + getf(raw, w, h, x, y + 1)) * 0.5;
                        b = (getf(raw, w, h, x - 1, y) + getf(raw, w, h, x + 1, y)) * 0.5;
                    }
                    out[i * 3] = r;
                    out[i * 3 + 1] = v;
                    out[i * 3 + 2] = b;
                }
            }
        }
    }
    out
}

#[cfg(feature = "fast-demosaic")]
/// Select the demosaic algorithm based on compile-time features.
/// Uses bilinear interpolation when `fast-demosaic` is enabled.
fn demosaic(raw: &[f32], w: i32, h: i32, pattern: i32) -> Vec<f32> {
    demosaic_bilinear(raw, w, h, pattern)
}

#[cfg(not(feature = "fast-demosaic"))]
/// Select the demosaic algorithm based on compile-time features.
/// Uses the AMaZE algorithm when `fast-demosaic` is not enabled.
fn demosaic(raw: &[f32], w: i32, h: i32, pattern: i32) -> Vec<f32> {
    demosaic_amaze(raw, w, h, pattern)
}
/// Save a linear RGB image to an OpenEXR file.
///
/// `img` must contain `w * h * 3` values in row-major order. `exposure` scales the
/// linear pixel values before writing.
fn save_exr_linear(
    path: &str,
    w: i32,
    h: i32,
    img: &[f32],
    exposure: f32,
) -> Result<(), exr::error::Error> {
    write_rgb_file(path, w as usize, h as usize, |x, y| {
        let i = idx(x as i32, y as i32, w) * 3;
        (
            img[i] * exposure,
            img[i + 1] * exposure,
            img[i + 2] * exposure,
        )
    })
}

// ---- main -------------------------------------------------------------------
fn main() {
    #[cfg(feature = "guiding")]
    let _guiding_state = {
        let enabled = std::env::args().any(|a| a == "--guiding");
        guiding::GuidingState::new(enabled)
    };
    let w = 1920;
    let h = 1440;
    let spp = 512;
    let max_depth = 32;
    let pattern = 0;
    let noise_threshold = 0.01_f32; // 0 disables adaptive sampling
    let min_adaptive_samples = 32;

    let rgb_bytes = (w * h * 3) as usize * std::mem::size_of::<f32>();
    let bayer_bytes = (w * h) as usize * std::mem::size_of::<f32>();

    let rgb_ptr = unsafe { ffi_alloc_host(rgb_bytes) as *mut f32 };
    let bayer_ptr = unsafe { ffi_alloc_host(bayer_bytes) as *mut f32 };

    let rgb = unsafe { std::slice::from_raw_parts_mut(rgb_ptr, (w * h * 3) as usize) };
    let bayer = unsafe { std::slice::from_raw_parts_mut(bayer_ptr, (w * h) as usize) };

    let t0 = Instant::now();
    assert_eq!(
        ffi_render_rgb(
            w,
            h,
            spp,
            max_depth,
            noise_threshold,
            min_adaptive_samples,
            rgb_ptr,
        ),
        0
    );
    let t_rgb = t0.elapsed();

    let t1 = Instant::now();
    assert_eq!(
        ffi_render_bayer_f32(
            w,
            h,
            spp,
            max_depth,
            noise_threshold,
            min_adaptive_samples,
            pattern,
            bayer_ptr,
        ),
        0
    );
    let t_raw = t1.elapsed();
    // Synchronize only when the CPU needs to read the GPU output
    unsafe {
        ffi_stream_sync();
    }
    let exposure = 1.0f32;
    println!("Exposure multiplier={:.6}", exposure);

    match save_exr_linear("pt.exr", w, h, &rgb, exposure) {
        Ok(_) => println!("✅ Saved pt.exr (render {:?})", t_rgb),
        Err(e) => eprintln!("Failed to save pt.exr: {}", e),
    }

    let t2 = Instant::now();
    let demosaiced = demosaic(&bayer, w, h, pattern);
    let t_demosaic = t2.elapsed();
    println!("Bayer render: {:?}", t_raw);
    println!(
        "Demosaic ({}): {:?}",
        if cfg!(feature = "fast-demosaic") {
            "bilinear"
        } else {
            "AMaZE"
        },
        t_demosaic
    );

    match save_exr_linear("pt_bayer.exr", w, h, &demosaiced, exposure) {
        Ok(_) => println!("✅ Saved pt_bayer.exr"),
        Err(e) => eprintln!("Failed to save pt_bayer.exr: {}", e),
    }

    unsafe {
        ffi_free_host(rgb_ptr as *mut std::ffi::c_void);
        ffi_free_host(bayer_ptr as *mut std::ffi::c_void);
    }
}

#[cfg(test)]
mod tests {
    use super::{cfa_at, CFA};

    #[test]
    fn pattern_rggb() {
        assert_eq!(cfa_at(0, 0, 0), CFA::R);
        assert_eq!(cfa_at(1, 0, 0), CFA::G);
        assert_eq!(cfa_at(0, 1, 0), CFA::G);
        assert_eq!(cfa_at(1, 1, 0), CFA::B);
    }

    #[test]
    fn pattern_bggr() {
        assert_eq!(cfa_at(0, 0, 1), CFA::B);
        assert_eq!(cfa_at(1, 0, 1), CFA::G);
        assert_eq!(cfa_at(0, 1, 1), CFA::G);
        assert_eq!(cfa_at(1, 1, 1), CFA::R);
    }

    #[test]
    fn pattern_grbg() {
        assert_eq!(cfa_at(0, 0, 2), CFA::G);
        assert_eq!(cfa_at(1, 0, 2), CFA::R);
        assert_eq!(cfa_at(0, 1, 2), CFA::B);
        assert_eq!(cfa_at(1, 1, 2), CFA::G);
    }

    #[test]
    fn pattern_gbrg() {
        assert_eq!(cfa_at(0, 0, 3), CFA::G);
        assert_eq!(cfa_at(1, 0, 3), CFA::B);
        assert_eq!(cfa_at(0, 1, 3), CFA::R);
        assert_eq!(cfa_at(1, 1, 3), CFA::G);
    }
}
