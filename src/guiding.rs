#[cfg(feature = "guiding")]
use std::ffi::{c_int, c_void};

#[cfg(feature = "guiding")]
#[repr(C)]
pub struct TrainSample {
    pub position: [f32; 3],
    pub dir_in: [f32; 3],
    pub contrib: [f32; 3],
    pub is_delta: u32,
}

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
    pub fn guiding_set_train_buffer_size(n: u32);
    pub fn guiding_set_grid_res(x: c_int, y: c_int, z: c_int);
    pub fn guiding_map_train_samples(samples: *mut *const TrainSample, count: *mut u32);
    pub fn guiding_reset_train_write_idx();
}

#[cfg(feature = "guiding")]
pub struct Device(*mut c_void);
#[cfg(feature = "guiding")]
impl Device {
    pub fn new(num_threads: i32) -> Option<Self> {
        let ptr = unsafe { pgl_dev_create(num_threads as c_int) };
        if ptr.is_null() {
            None
        } else {
            Some(Self(ptr))
        }
    }
    pub fn as_ptr(&self) -> *mut c_void {
        self.0
    }
}
#[cfg(feature = "guiding")]
impl Drop for Device {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { pgl_dev_destroy(self.0) };
        }
    }
}

#[cfg(feature = "guiding")]
pub struct Field(*mut c_void);
#[cfg(feature = "guiding")]
impl Field {
    pub fn new(dev: &Device, world_min: [f32; 3], world_max: [f32; 3]) -> Option<Self> {
        let ptr = unsafe { pgl_field_create(dev.as_ptr(), world_min.as_ptr(), world_max.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self(ptr))
        }
    }
    pub fn as_ptr(&self) -> *mut c_void {
        self.0
    }
}
#[cfg(feature = "guiding")]
impl Drop for Field {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { pgl_field_destroy(self.0) };
        }
    }
}

#[cfg(feature = "guiding")]
pub struct Samples(*mut c_void);
#[cfg(feature = "guiding")]
impl Samples {
    pub fn new() -> Option<Self> {
        let ptr = unsafe { pgl_samples_create() };
        if ptr.is_null() {
            None
        } else {
            Some(Self(ptr))
        }
    }
    pub fn as_ptr(&self) -> *mut c_void {
        self.0
    }
}
#[cfg(feature = "guiding")]
impl Drop for Samples {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { pgl_samples_destroy(self.0) };
        }
    }
}

#[cfg(feature = "guiding")]
pub struct GuidingState {
    pub enabled: bool,
    dev: Option<Device>,
    field: Option<Field>,
    samples: Option<Samples>,
    grid_res: [i32; 3],
    update_interval: u32,
    frame_idx: u32,
}

#[cfg(feature = "guiding")]
impl GuidingState {
    pub fn new(
        enabled: bool,
        scene_min: [f32; 3],
        scene_max: [f32; 3],
        grid_res: [i32; 3],
        update_interval: u32,
        train_capacity: u32,
    ) -> Self {
        unsafe {
            guiding_set_enabled(if enabled { 1 } else { 0 });
            guiding_set_train_buffer_size(train_capacity);
            guiding_set_grid_res(
                grid_res[0] as c_int,
                grid_res[1] as c_int,
                grid_res[2] as c_int,
            );
        }
        let dev = if enabled { Device::new(0) } else { None };
        let field = if enabled {
            dev.as_ref()
                .and_then(|d| Field::new(d, scene_min, scene_max))
        } else {
            None
        };
        let samples = if enabled { Samples::new() } else { None };
        Self {
            enabled,
            dev,
            field,
            samples,
            grid_res,
            update_interval,
            frame_idx: 0,
        }
    }

    fn collect_train_samples(&mut self) {
        if !(self.enabled) {
            return;
        }
        let samples_ptr = match self.samples {
            Some(ref s) => s.as_ptr(),
            None => return,
        };
        unsafe {
            let mut ptr: *const TrainSample = std::ptr::null();
            let mut count: u32 = 0;
            guiding_map_train_samples(&mut ptr, &mut count);
            if !ptr.is_null() && count > 0 {
                let slice = std::slice::from_raw_parts(ptr, count as usize);
                for ts in slice {
                    pgl_samples_add_surface(
                        samples_ptr,
                        ts.position.as_ptr(),
                        ts.dir_in.as_ptr(),
                        ts.contrib.as_ptr(),
                        ts.is_delta as c_int,
                    );
                }
            }
            guiding_reset_train_write_idx();
        }
    }

    pub fn process_batch(&mut self) {
        if !self.enabled {
            return;
        }
        self.collect_train_samples();
        self.frame_idx += 1;
        if self.frame_idx % self.update_interval == 0 {
            if let (Some(ref f), Some(ref s)) = (&self.field, &self.samples) {
                unsafe {
                    pgl_field_update(f.as_ptr(), s.as_ptr());
                    let mut regions: *const PglRegion = std::ptr::null();
                    let mut region_count: u32 = 0;
                    let mut lobes: *const PglLobe = std::ptr::null();
                    let mut lobe_count: u32 = 0;
                    pgl_field_snapshot(
                        f.as_ptr(),
                        &mut regions,
                        &mut region_count,
                        &mut lobes,
                        &mut lobe_count,
                    );
                    guiding_upload_snapshot(regions, region_count, lobes, lobe_count);
                }
            }
        }
    }
}
