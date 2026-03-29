#[cfg(feature = "float_io")]
pub type Sample = f32;
#[cfg(not(feature = "float_io"))]
pub type Sample = f64;

/// Metadata parsed from the .nam file.
#[derive(Debug, Clone, Default)]
pub struct DspMetadata {
    pub loudness: Option<f64>,
    pub expected_sample_rate: Option<f64>,
    pub input_level_dbu: Option<f64>,
    pub output_level_dbu: Option<f64>,
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
        let n = self.prewarm_samples();
        if n == 0 {
            return;
        }
        let silence = vec![Sample::default(); n];
        let mut discard = vec![Sample::default(); n];
        self.process(&silence, &mut discard);
    }

    fn metadata(&self) -> &DspMetadata;

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
            out[0] = output[0] as f32;
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
        for f in 0..num_frames {
            let col_start = f * output_stride;
            self.process_sample_multi_channel(
                input[f],
                &mut output_data[col_start..col_start + out_channels],
            );
        }
    }

    fn set_max_buffer_size(&mut self, _max_buffer_size: usize) {}
}
