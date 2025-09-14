#[cfg(feature = "guiding")]
use std::ffi::{c_int, c_void};

#[cfg(feature = "guiding")]
#[repr(C)]
pub struct PglRegion {
    pub bmin: [f32; 3],
    pub bmax: [f32; 3],
    pub lobe_ofs: u32,
    pub lobe_num: u32,
}
#[cfg(feature = "guiding")]
#[repr(C)]
pub struct PglLobe {
    pub mu: [f32; 3],
    pub kappa: f32,
    pub weight: f32,
    pub rgb: [f32; 3],
}

#[cfg(feature = "guiding")]
extern "C" {
    pub fn pgl_dev_create(num_threads: c_int) -> *mut c_void;
    pub fn pgl_dev_destroy(dev: *mut c_void);
    pub fn pgl_field_create(
        dev: *mut c_void,
        world_min: *const f32,
        world_max: *const f32,
    ) -> *mut c_void;
    pub fn pgl_field_destroy(field: *mut c_void);
    pub fn pgl_samples_create() -> *mut c_void;
    pub fn pgl_samples_destroy(s: *mut c_void);
    pub fn pgl_samples_add_surface(
        s: *mut c_void,
        pos: *const f32,
        dir_in: *const f32,
        weight_rgb: *const f32,
        is_delta: c_int,
    );
    pub fn pgl_field_update(field: *mut c_void, samples: *mut c_void);
    pub fn pgl_field_snapshot(
        field: *mut c_void,
        regions: *mut *const PglRegion,
        region_count: *mut u32,
        lobes: *mut *const PglLobe,
        lobe_count: *mut u32,
    );
    pub fn guiding_upload_snapshot(
        regions: *const PglRegion,
        n_regions: u32,
        lobes: *const PglLobe,
        n_lobes: u32,
    );
    pub fn guiding_set_enabled(enabled: c_int);
}

#[cfg(feature = "guiding")]
pub struct GuidingState {
    pub enabled: bool,
}
#[cfg(feature = "guiding")]
impl GuidingState {
    pub fn new(enabled: bool) -> Self {
        unsafe {
            guiding_set_enabled(if enabled { 1 } else { 0 });
        }
        Self { enabled }
    }
}
