use avif_parse::read_avif;
use avif_serialize::constants::{
    ColorPrimaries, MatrixCoefficients as SerializeMatrixCoefficients, TransferCharacteristics,
};
use rav1e::color::PixelRange;
use ravif::{BitDepth, MatrixCoefficients};
use std::ffi::c_int;
use std::fs::File;
use std::io;
use std::time::Instant;

// ---- FFI ----------------------------------------------------------
extern "C" {
    fn optix_render_rgb(
        w: c_int,
        h: c_int,
        spp: c_int,
        max_depth: c_int,
        out_rgb: *mut f32,
    ) -> c_int;
    fn optix_render_bayer_f32(
        w: c_int,
        h: c_int,
        spp: c_int,
        max_depth: c_int,
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
/// - `out`: Pointer to a buffer of at least `w * h * 3` `f32` values.
///
/// # Returns
/// Zero on success, non-zero on failure.
#[inline]
fn ffi_render_rgb(w: i32, h: i32, spp: i32, max_depth: i32, out: *mut f32) -> c_int {
    unsafe { optix_render_rgb(w, h, spp, max_depth, out) }
}
/// Render a Bayer mosaic image using the GPU backend.
///
/// # Parameters
/// - `w`: Image width in pixels. Must be positive.
/// - `h`: Image height in pixels. Must be positive.
/// - `spp`: Samples per pixel. Must be positive.
/// - `pat`: Bayer pattern index.
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
    pat: i32,
    out: *mut f32,
) -> c_int {
    unsafe { optix_render_bayer_f32(w, h, spp, max_depth, pat, out) }
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
/// Return the `p`th percentile of `arr`.
///
/// `p` should be in `[0, 1]`. Empty arrays yield `1.0` to avoid division by zero.
fn percentile(arr: &[f32], p: f32) -> f32 {
    let mut v: Vec<f32> = arr.to_vec();
    let n = v.len();
    if n == 0 {
        return 1.0;
    }
    let k = ((n as f32 - 1.0) * p).round().max(0.0) as usize;
    let _ = v.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    v[k].max(1e-6)
}
/// Multiply a 3×3 matrix by a 3-component vector.
fn mul3(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}
/// ACES filmic tone mapping curve.
fn aces_film(x: f32) -> f32 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
}
/// Apply [`aces_film`] to each channel of `v`.
fn tonemap_aces(v: [f32; 3]) -> [f32; 3] {
    [aces_film(v[0]), aces_film(v[1]), aces_film(v[2])]
}
/// Standard sRGB OETF converting linear light to gamma-encoded value.
fn oetf_srgb(x: f32) -> f32 {
    if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

const M_ACESCG_TO_SRGB: [[f32; 3]; 3] = [
    [1.4514393, -0.23651075, -0.21492857],
    [-0.07655377, 1.1762297, -0.09967593],
    [0.00831615, -0.00603245, 0.99771631],
];
const M_ACESCG_TO_REC2020: [[f32; 3]; 3] = [
    [1.705051, -0.621792, -0.083258],
    [-0.130257, 1.140802, -0.010547],
    [-0.024003, -0.128968, 1.152971],
];
/// Save an ACEScg image to an 8-bit sRGB PNG file.
///
/// `img_aces` must have `w * h * 3` elements. `exposure` scales the pixel values before
/// tonemapping.
fn save_png_srgb_from_acescg(
    path: &str,
    w: i32,
    h: i32,
    img_aces: &[f32],
    exposure: f32,
) -> Result<(), io::Error> {
    let mut buf = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = idx(x, y, w) * 3;
            let a = [
                img_aces[i + 0] * exposure,
                img_aces[i + 1] * exposure,
                img_aces[i + 2] * exposure,
            ];
            let t = tonemap_aces(a);
            let srgb_lin = mul3(M_ACESCG_TO_SRGB, t);
            let r = oetf_srgb(srgb_lin[0].clamp(0.0, 1.0));
            let g = oetf_srgb(srgb_lin[1].clamp(0.0, 1.0));
            let b = oetf_srgb(srgb_lin[2].clamp(0.0, 1.0));
            buf[i + 0] = (r * 255.0 + 0.5) as u8;
            buf[i + 1] = (g * 255.0 + 0.5) as u8;
            buf[i + 2] = (b * 255.0 + 0.5) as u8;
        }
    }
    let file = File::create(path)?;
    let mut enc = png::Encoder::new(file, w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    let mut writer = enc.write_header()?;
    writer.write_image_data(&buf)?;
    Ok(())
}
/// Convert absolute luminance in nits to a PQ-encoded value.
fn pq_oetf_from_nits(nits: f32) -> f32 {
    let m1 = 2610.0 / 16384.0;
    let m2 = 2523.0 / 32.0;
    let c1 = 3424.0 / 4096.0;
    let c2 = 2413.0 / 128.0;
    let c3 = 2392.0 / 128.0;
    let l = (nits / 10000.0).max(0.0);
    let l_m1 = l.powf(m1);
    ((c1 + c2 * l_m1) / (1.0 + c3 * l_m1))
        .powf(m2)
        .clamp(0.0, 1.0)
}
/// Save an ACEScg image as a 10-bit Rec.2100 PQ AVIF file.
///
/// `img_aces` must contain `w * h * 3` values. `exposure` scales values before tonemapping.
fn save_avif_rec2100_pq_from_acescg(
    path: &str,
    w: i32,
    h: i32,
    img_aces: &[f32],
    exposure: f32,
) -> Result<(), io::Error> {
    let n = (w * h) as usize;
    let mut rgb10: Vec<rgb::RGB<u16>> = Vec::with_capacity(n);
    for y in 0..h {
        for x in 0..w {
            let i = idx(x, y, w) * 3;
            let a = [
                img_aces[i] * exposure,
                img_aces[i + 1] * exposure,
                img_aces[i + 2] * exposure,
            ];
            let t = tonemap_aces(a);
            let rec2020 = mul3(M_ACESCG_TO_REC2020, t);
            // ACES tonemapping is defined for a 100 nit reference display.
            let r = pq_oetf_from_nits(rec2020[0].max(0.0) * 200.0);
            let g = pq_oetf_from_nits(rec2020[1].max(0.0) * 200.0);
            let b = pq_oetf_from_nits(rec2020[2].max(0.0) * 200.0);
            let r10 = (r * 1023.0 + 0.5) as u16;
            let g10 = (g * 1023.0 + 0.5) as u16;
            let b10 = (b * 1023.0 + 0.5) as u16;
            rgb10.push(rgb::RGB {
                r: r10,
                g: g10,
                b: b10,
            });
        }
    }

    const BT2020: [f32; 3] = [0.2627, 0.6780, 0.0593];
    let planes = rgb10.iter().map(|&px| {
        let r = px.r as f32;
        let g = px.g as f32;
        let b = px.b as f32;
        let y = BT2020[0] * r + BT2020[1] * g + BT2020[2] * b;
        let cb = (b - y) * (0.5 / (1.0 - BT2020[2])) + 512.0;
        let cr = (r - y) * (0.5 / (1.0 - BT2020[0])) + 512.0;
        [y.round() as u16, cb.round() as u16, cr.round() as u16]
    });

    let enc = ravif::Encoder::new()
        .with_quality(90.0)
        .with_speed(6)
        .with_bit_depth(BitDepth::Ten);
    let avif = enc
        .encode_raw_planes_10_bit(
            w as usize,
            h as usize,
            planes,
            None::<[_; 0]>,
            PixelRange::Full,
            MatrixCoefficients::BT2020NCL,
        )
        .expect("avif encode");

    let parsed = read_avif(&mut avif.avif_file.as_slice()).expect("parse avif");
    let avif_bytes = avif_serialize::Aviffy::new()
        .set_color_primaries(ColorPrimaries::Bt2020)
        .set_transfer_characteristics(TransferCharacteristics::Smpte2084)
        .set_matrix_coefficients(SerializeMatrixCoefficients::Bt2020Ncl)
        .set_full_color_range(true)
        .to_vec(
            &parsed.primary_item,
            parsed.alpha_item.as_deref(),
            w as u32,
            h as u32,
            10,
        );

    std::fs::write(path, &avif_bytes)?;
    Ok(())
}

// ---- main -------------------------------------------------------------------
fn main() {
    let w = 1920;
    let h = 1440;
    let spp = 128;
    let max_depth = 32;
    let pattern = 0;

    let rgb_bytes = (w * h * 3) as usize * std::mem::size_of::<f32>();
    let bayer_bytes = (w * h) as usize * std::mem::size_of::<f32>();

    let rgb_ptr = unsafe { ffi_alloc_host(rgb_bytes) as *mut f32 };
    let bayer_ptr = unsafe { ffi_alloc_host(bayer_bytes) as *mut f32 };

    let rgb = unsafe { std::slice::from_raw_parts_mut(rgb_ptr, (w * h * 3) as usize) };
    let bayer = unsafe { std::slice::from_raw_parts_mut(bayer_ptr, (w * h) as usize) };

    let t0 = Instant::now();
    assert_eq!(ffi_render_rgb(w, h, spp, max_depth, rgb_ptr), 0);
    let t_rgb = t0.elapsed();

    let t1 = Instant::now();
    assert_eq!(
        ffi_render_bayer_f32(w, h, spp, max_depth, pattern, bayer_ptr),
        0
    );
    let t_raw = t1.elapsed();
    let mut lums = Vec::with_capacity((w * h) as usize);

    // Synchronize only when the CPU needs to read the GPU output
    unsafe {
        ffi_stream_sync();
    }
    for i in 0..(w * h) as usize {
        let r = rgb[i * 3];
        let g = rgb[i * 3 + 1];
        let b = rgb[i * 3 + 2];
        lums.push(0.2126 * r + 0.7152 * g + 0.0722 * b);
    }
    let target_percentile = 0.85;
    let exp_rgb = target_percentile / percentile(&lums, target_percentile);
    println!(
        "Auto-exposure: target percentile={:.2}, exposure multiplier={:.6}",
        target_percentile, exp_rgb
    );

    match save_png_srgb_from_acescg("pt.png", w, h, &rgb, exp_rgb) {
        Ok(_) => println!("✅ Saved pt.png (render {:?})", t_rgb),
        Err(e) => eprintln!("Failed to save pt.png: {}", e),
    }

    let t2 = Instant::now();
    let demosaiced = demosaic(&bayer, w, h, pattern);
    let t_demosaic = t2.elapsed();
    let mut l2 = Vec::with_capacity((w * h) as usize);
    for i in 0..(w * h) as usize {
        let r = demosaiced[i * 3];
        let g = demosaiced[i * 3 + 1];
        let b = demosaiced[i * 3 + 2];
        l2.push(0.2126 * r + 0.7152 * g + 0.0722 * b);
    }
    let exp_raw = target_percentile / percentile(&l2, target_percentile);

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

    match save_png_srgb_from_acescg("pt_bayer.png", w, h, &demosaiced, exp_raw) {
        Ok(_) => println!("✅ Saved pt_bayer.png"),
        Err(e) => eprintln!("Failed to save pt_bayer.png: {}", e),
    }
    if let Err(e) = save_avif_rec2100_pq_from_acescg("pt_pq.avif", w, h, &rgb, exp_rgb) {
        eprintln!("Failed to save pt_pq.avif: {}", e);
    }
    if let Err(e) = save_avif_rec2100_pq_from_acescg("pt_bayer_pq.avif", w, h, &demosaiced, exp_raw)
    {
        eprintln!("Failed to save pt_bayer_pq.avif: {}", e);
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
