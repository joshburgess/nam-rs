use wasm_bindgen::prelude::*;

// ── wasm-bindgen API (used by main thread via ES module import) ────────────

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct NamModel {
    inner: Box<dyn nam_core::Dsp>,
}

#[wasm_bindgen]
impl NamModel {
    /// Load a NAM model from raw .nam file bytes.
    #[wasm_bindgen(constructor)]
    pub fn load(data: &[u8]) -> Result<NamModel, JsValue> {
        let json_str = std::str::from_utf8(data)
            .map_err(|e| JsValue::from_str(&format!("Invalid UTF-8: {e}")))?;
        let inner = nam_core::get_dsp_from_json(json_str)
            .map_err(|e| JsValue::from_str(&format!("Failed to load model: {e}")))?;
        Ok(NamModel { inner })
    }

    /// Reset the model state for the given sample rate and buffer size.
    pub fn reset(&mut self, sample_rate: f64, buffer_size: usize) {
        self.inner.reset(sample_rate, buffer_size);
    }

    /// Prewarm the model to stabilize internal state.
    pub fn prewarm(&mut self) {
        self.inner.prewarm();
    }

    /// Process a block of audio. Input and output must be the same length.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.inner.process(input, output);
    }

    /// Return the model's expected sample rate, if known.
    pub fn sample_rate(&self) -> Option<f32> {
        self.inner.metadata().expected_sample_rate.map(|r| r as f32)
    }
}

// ── C-style FFI (used by AudioWorklet — zero JS import dependencies) ───────
//
// These functions are exported as plain WASM exports with no wasm-bindgen
// glue, so the worklet can instantiate the module with an empty import object.
// The worklet manages memory pointers directly.

static mut MODEL: Option<Box<dyn nam_core::Dsp>> = None;

/// Allocate `size` bytes aligned to `align` in WASM linear memory.
/// Returns a pointer. The caller must free with `nam_dealloc`.
#[no_mangle]
pub extern "C" fn nam_alloc(size: usize, align: usize) -> *mut u8 {
    let layout = std::alloc::Layout::from_size_align(size, align).unwrap();
    unsafe { std::alloc::alloc(layout) }
}

/// Free a previous allocation.
#[no_mangle]
pub extern "C" fn nam_dealloc(ptr: *mut u8, size: usize, align: usize) {
    let layout = std::alloc::Layout::from_size_align(size, align).unwrap();
    unsafe { std::alloc::dealloc(ptr, layout) }
}

/// Load a model from JSON bytes already in WASM memory.
/// Returns 0 on success, 1 on error.
#[no_mangle]
pub extern "C" fn nam_load(json_ptr: *const u8, json_len: usize) -> i32 {
    let json_bytes = unsafe { std::slice::from_raw_parts(json_ptr, json_len) };
    let json_str = match std::str::from_utf8(json_bytes) {
        Ok(s) => s,
        Err(_) => return 1,
    };
    match nam_core::get_dsp_from_json(json_str) {
        Ok(dsp) => {
            unsafe { MODEL = Some(dsp) };
            0
        }
        Err(_) => 1,
    }
}

/// Reset the loaded model.
#[no_mangle]
pub extern "C" fn nam_reset(sample_rate: f64, buffer_size: usize) {
    unsafe {
        if let Some(ref mut m) = MODEL {
            m.reset(sample_rate, buffer_size);
        }
    }
}

/// Prewarm the loaded model.
#[no_mangle]
pub extern "C" fn nam_prewarm() {
    unsafe {
        if let Some(ref mut m) = MODEL {
            m.prewarm();
        }
    }
}

/// Process audio. Input and output pointers must point to `len` f32 values
/// already allocated in WASM linear memory.
#[no_mangle]
pub extern "C" fn nam_process(input_ptr: *const f32, output_ptr: *mut f32, len: usize) {
    unsafe {
        if let Some(ref mut m) = MODEL {
            let input = std::slice::from_raw_parts(input_ptr, len);
            let output = std::slice::from_raw_parts_mut(output_ptr, len);
            m.process(input, output);
        }
    }
}

/// Get the model's expected sample rate. Returns 0.0 if no model loaded or unknown.
#[no_mangle]
pub extern "C" fn nam_sample_rate() -> f32 {
    unsafe {
        MODEL
            .as_ref()
            .and_then(|m| m.metadata().expected_sample_rate)
            .map(|r| r as f32)
            .unwrap_or(0.0)
    }
}
