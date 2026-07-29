#[cfg(feature = "float_io")]
pub type Sample = f32;
#[cfg(not(feature = "float_io"))]
pub type Sample = f64;

use crate::error::NamError;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ActivationMode {
    #[default]
    Accurate,
    Fast,
}

impl ActivationMode {
    #[inline]
    pub(crate) fn use_fast_tanh(self) -> bool {
        matches!(self, Self::Fast)
    }
}

#[cfg(feature = "float_io")]
pub fn sample_to_f64(sample: Sample) -> f64 {
    f64::from(sample)
}

#[cfg(not(feature = "float_io"))]
pub fn sample_to_f64(sample: Sample) -> f64 {
    sample
}

#[cfg(feature = "float_io")]
pub fn sample_to_f32(sample: Sample) -> f32 {
    sample
}

#[cfg(not(feature = "float_io"))]
pub fn sample_to_f32(sample: Sample) -> f32 {
    sample as f32
}

#[cfg(feature = "float_io")]
pub fn sample_from_f32(sample: f32) -> Sample {
    sample
}

#[cfg(not(feature = "float_io"))]
pub fn sample_from_f32(sample: f32) -> Sample {
    f64::from(sample)
}

#[cfg(feature = "float_io")]
pub fn sample_from_f64(sample: f64) -> Sample {
    sample as f32
}

#[cfg(not(feature = "float_io"))]
pub fn sample_from_f64(sample: f64) -> Sample {
    sample
}

/// Metadata parsed from the .nam file.
#[derive(Debug, Clone, Default)]
pub struct DspMetadata {
    pub raw: Option<serde_json::Value>,
    pub loudness: Option<f64>,
    pub gain: Option<f64>,
    pub expected_sample_rate: Option<f64>,
    pub name: Option<String>,
    pub modeled_by: Option<String>,
    pub gear_type: Option<String>,
    pub gear_make: Option<String>,
    pub gear_model: Option<String>,
    pub tone_type: Option<String>,
    pub input_level_dbu: Option<f64>,
    pub output_level_dbu: Option<f64>,
    pub validation_esr: Option<f64>,
}

pub trait Dsp: Send {
    /// Process audio. `input` and `output` must be the same length.
    fn process(&mut self, input: &[Sample], output: &mut [Sample]);

    /// Reset state for a new sample rate and buffer size.
    fn reset(&mut self, sample_rate: f64, max_buffer_size: usize);

    /// Number of silence samples needed for prewarm.
    fn prewarm_samples(&self) -> usize {
        0
    }

    /// Warm up by processing silence.
    fn prewarm(&mut self) {
        let mut remaining = self.prewarm_samples();
        let silence = [Sample::default(); 64];
        let mut discard = [Sample::default(); 64];
        while remaining > 0 {
            let chunk_size = remaining.min(silence.len());
            self.process(&silence[..chunk_size], &mut discard[..chunk_size]);
            remaining -= chunk_size;
        }
    }

    fn metadata(&self) -> &DspMetadata;

    fn set_activation_mode(&mut self, _mode: ActivationMode) {}

    /// Select a slimmable model width, where 0.0 chooses the smallest width and 1.0 the largest.
    fn set_slimming(&mut self, _value: f64) -> Result<(), NamError> {
        Err(NamError::UnsupportedOperation {
            operation: "slimmable width selection",
        })
    }

    /// Return sorted internal size-selection breakpoints. The bounds 0.0 and 1.0 are implied.
    fn slimming_breakpoints(&self) -> Vec<f64> {
        Vec::new()
    }

    /// Number of output channels. Default is 1 (mono).
    /// Overridden by models that produce multi-channel output (e.g. WaveNet used as condition_dsp).
    fn num_output_channels(&self) -> usize {
        1
    }

    /// Process one sample and write multi-channel output.
    /// Default implementation calls process() and writes the single output to out[0].
    fn process_sample_multi_channel(&mut self, input_sample: Sample, out: &mut [f32]) {
        let input = [input_sample];
        let mut output = [Sample::default()];
        self.process(&input, &mut output);
        if !out.is_empty() {
            out[0] = sample_to_f32(output[0]);
        }
    }

    /// Process a block of samples and write multi-channel output in column-major format.
    /// output_data[f * output_stride + c] = channel c of frame f.
    /// Default falls back to per-sample processing.
    fn process_block_multi_channel(
        &mut self,
        input: &[Sample],
        output_data: &mut [f32],
        output_stride: usize,
        out_channels: usize,
        num_frames: usize,
    ) {
        for (f, &sample) in input.iter().enumerate().take(num_frames) {
            let col_start = f * output_stride;
            self.process_sample_multi_channel(
                sample,
                &mut output_data[col_start..col_start + out_channels],
            );
        }
    }

    fn set_max_buffer_size(&mut self, _max_buffer_size: usize) {}
}
