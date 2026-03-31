use wasm_bindgen::prelude::*;

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
