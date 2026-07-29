use crate::dsp::{Dsp, DspMetadata, Sample};
use crate::error::NamError;
use crate::util::{positive_config_usize, WeightIter};
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

const AUTO_DIRECT_MAX_TAPS: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LinearImplementation {
    Auto,
    Direct,
    Fft,
}

impl LinearImplementation {
    fn from_config(config: &serde_json::Value) -> Result<Self, NamError> {
        let Some(value) = config.get("implementation") else {
            return Ok(Self::Auto);
        };
        let implementation = value.as_str().ok_or_else(|| NamError::InvalidConfigType {
            field: "implementation".into(),
            expected: "a string",
        })?;
        match implementation.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "direct" | "legacy" | "old" => Ok(Self::Direct),
            "fft" | "partitioned_fft" | "partitioned-fft" => Ok(Self::Fft),
            _ => Err(NamError::UnsupportedConfigValue {
                field: "implementation".into(),
                value: implementation.into(),
            }),
        }
    }
}

struct LinearFftState {
    block_size: usize,
    fft_size: usize,
    direct_taps: usize,
    kernel_spectra: Vec<Vec<Complex32>>,
    input_spectra: Vec<Vec<Complex32>>,
    input_time: Vec<Complex32>,
    input_pos: usize,
    spectrum_write_index: usize,
    output_ring: Vec<f32>,
    sample_index: usize,
    accumulator: Vec<Complex32>,
    scratch: Vec<Complex32>,
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
}

impl LinearFftState {
    fn new(weights: &[f32]) -> Self {
        let block_size = if weights.len() <= 2_048 {
            256
        } else if weights.len() <= 8_192 {
            512
        } else {
            1_024
        };
        let fft_size = 2 * block_size;
        let direct_taps = weights.len().min(block_size);
        let tail_len = weights.len() - direct_taps;
        let num_partitions = tail_len.div_ceil(block_size);

        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(fft_size);
        let inverse = planner.plan_fft_inverse(fft_size);
        let scratch_len = forward
            .get_inplace_scratch_len()
            .max(inverse.get_inplace_scratch_len());
        let mut scratch = vec![Complex32::default(); scratch_len];
        let mut kernel_spectra = vec![vec![Complex32::default(); fft_size]; num_partitions];
        for (partition, spectrum) in kernel_spectra.iter_mut().enumerate() {
            let start = direct_taps + partition * block_size;
            let count = block_size.min(weights.len() - start);
            for (bin, weight) in spectrum[..count]
                .iter_mut()
                .zip(&weights[start..start + count])
            {
                bin.re = *weight;
            }
            forward.process_with_scratch(spectrum, &mut scratch);
        }

        Self {
            block_size,
            fft_size,
            direct_taps,
            kernel_spectra,
            input_spectra: vec![vec![Complex32::default(); fft_size]; num_partitions],
            input_time: vec![Complex32::default(); fft_size],
            input_pos: 0,
            spectrum_write_index: 0,
            output_ring: vec![0.0; 4 * block_size],
            sample_index: 0,
            accumulator: vec![Complex32::default(); fft_size],
            scratch,
            forward,
            inverse,
        }
    }

    fn reset(&mut self) {
        for spectrum in &mut self.input_spectra {
            spectrum.fill(Complex32::default());
        }
        self.input_time.fill(Complex32::default());
        self.input_pos = 0;
        self.spectrum_write_index = 0;
        self.output_ring.fill(0.0);
        self.sample_index = 0;
        self.accumulator.fill(Complex32::default());
        self.scratch.fill(Complex32::default());
    }

    fn push_sample(&mut self, sample: f32) {
        if self.kernel_spectra.is_empty() {
            self.sample_index += 1;
            return;
        }
        self.input_time[self.input_pos].re = sample;
        self.input_pos += 1;
        if self.input_pos == self.block_size {
            self.run_block();
        }
        self.sample_index += 1;
    }

    fn take_tail(&mut self) -> f32 {
        let index = self.sample_index % self.output_ring.len();
        let value = self.output_ring[index];
        self.output_ring[index] = 0.0;
        value
    }

    fn run_block(&mut self) {
        self.forward
            .process_with_scratch(&mut self.input_time, &mut self.scratch);
        self.input_spectra[self.spectrum_write_index].copy_from_slice(&self.input_time);
        self.accumulator.fill(Complex32::default());

        for (partition, kernel) in self.kernel_spectra.iter().enumerate() {
            let input_index = (self.spectrum_write_index + self.input_spectra.len() - partition)
                % self.input_spectra.len();
            for ((sum, input), kernel) in self
                .accumulator
                .iter_mut()
                .zip(&self.input_spectra[input_index])
                .zip(kernel)
            {
                *sum += *input * *kernel;
            }
        }
        self.inverse
            .process_with_scratch(&mut self.accumulator, &mut self.scratch);

        let output_start = self.sample_index + 1 - self.block_size + self.direct_taps;
        let scale = 1.0 / self.fft_size as f32;
        for (offset, value) in self.accumulator[..self.fft_size - 1].iter().enumerate() {
            let index = (output_start + offset) % self.output_ring.len();
            self.output_ring[index] += value.re * scale;
        }

        self.input_time.fill(Complex32::default());
        self.input_pos = 0;
        self.spectrum_write_index = (self.spectrum_write_index + 1) % self.input_spectra.len();
    }
}

pub struct Linear {
    weights: Vec<f32>,
    bias: f32,
    history: Vec<f32>,
    history_pos: usize,
    metadata: DspMetadata,
    fft: Option<LinearFftState>,
}

impl Linear {
    pub fn from_config(
        config: &serde_json::Value,
        weights: &[f32],
        metadata: DspMetadata,
    ) -> Result<Self, NamError> {
        let receptive_field = positive_config_usize(&config["receptive_field"], "receptive_field")?;
        let has_bias = config["bias"].as_bool().unwrap_or(false);
        let requested_implementation = LinearImplementation::from_config(config)?;

        let mut iter = WeightIter::new(weights);
        let w = iter.take(receptive_field)?;
        let w = w.to_vec();
        let bias = if has_bias { iter.take(1)?[0] } else { 0.0 };
        iter.assert_exhausted()?;

        let implementation = match requested_implementation {
            LinearImplementation::Auto if receptive_field > AUTO_DIRECT_MAX_TAPS => {
                LinearImplementation::Fft
            }
            LinearImplementation::Auto => LinearImplementation::Direct,
            implementation => implementation,
        };
        let fft = (implementation == LinearImplementation::Fft).then(|| LinearFftState::new(&w));

        Ok(Self {
            weights: w,
            bias,
            history: vec![0.0; receptive_field],
            history_pos: 0,
            metadata,
            fft,
        })
    }
}

impl Dsp for Linear {
    fn process(&mut self, input: &[Sample], output: &mut [Sample]) {
        let len = self.weights.len();
        let direct_taps = self.fft.as_ref().map_or(len, |fft| fft.direct_taps);
        for (i, &sample) in input.iter().enumerate() {
            let sample = crate::dsp::sample_to_f32(sample);
            self.history[self.history_pos] = sample;

            let mut sum = 0.0f32;
            for (j, &w) in self.weights[..direct_taps].iter().enumerate() {
                let idx = (self.history_pos + len - j) % len;
                sum += w * self.history[idx];
            }
            if let Some(fft) = self.fft.as_mut() {
                sum += fft.take_tail();
                fft.push_sample(sample);
            }
            sum += self.bias;
            output[i] = sum as Sample;

            self.history_pos = (self.history_pos + 1) % len;
        }
    }

    fn reset(&mut self, _sample_rate: f64, _max_buffer_size: usize) {
        self.history.fill(0.0);
        self.history_pos = 0;
        if let Some(fft) = self.fft.as_mut() {
            fft.reset();
        }
    }

    fn metadata(&self) -> &DspMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linear(weights: &[f32]) -> Linear {
        let config = serde_json::json!({
            "receptive_field": weights.len()
        });
        Linear::from_config(&config, weights, DspMetadata::default()).unwrap()
    }

    fn make_linear_with_implementation(weights: &[f32], implementation: &str) -> Linear {
        let config = serde_json::json!({
            "receptive_field": weights.len(),
            "implementation": implementation
        });
        Linear::from_config(&config, weights, DspMetadata::default()).unwrap()
    }

    fn render_in_chunks(model: &mut Linear, input: &[Sample], chunks: &[usize]) -> Vec<Sample> {
        let mut output = vec![Sample::default(); input.len()];
        let mut offset = 0;
        let mut chunk_index = 0;
        while offset < input.len() {
            let count = chunks[chunk_index % chunks.len()].min(input.len() - offset);
            model.process(
                &input[offset..offset + count],
                &mut output[offset..offset + count],
            );
            offset += count;
            chunk_index += 1;
        }
        output
    }

    #[test]
    fn test_identity_single_tap() {
        let mut model = make_linear(&[1.0]);
        let input = vec![0.5 as Sample, 0.25 as Sample];
        let mut output = vec![0.0 as Sample; 2];
        model.process(&input, &mut output);
        assert!((output[0] - 0.5).abs() < 1e-6);
        assert!((output[1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_bias() {
        let config = serde_json::json!({
            "receptive_field": 1,
            "bias": true
        });
        let mut model = Linear::from_config(&config, &[2.0, 0.25], DspMetadata::default()).unwrap();
        let input = vec![0.5 as Sample, -0.5 as Sample];
        let mut output = vec![0.0 as Sample; 2];
        model.process(&input, &mut output);
        assert!((output[0] - 1.25).abs() < 1e-6);
        assert!((output[1] + 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_bias_weight_required_when_enabled() {
        let config = serde_json::json!({
            "receptive_field": 1,
            "bias": true
        });
        assert!(Linear::from_config(&config, &[1.0], DspMetadata::default()).is_err());
    }

    #[test]
    fn test_scaling() {
        let mut model = make_linear(&[2.0]);
        let input = vec![1.0 as Sample];
        let mut output = vec![0.0 as Sample; 1];
        model.process(&input, &mut output);
        assert!((output[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_fir_two_taps() {
        let mut model = make_linear(&[0.5, 0.5]);
        let input = vec![1.0 as Sample, 0.0, 0.0, 0.0];
        let mut output = vec![0.0 as Sample; 4];
        model.process(&input, &mut output);
        assert!((output[0] - 0.5).abs() < 1e-6);
        assert!((output[1] - 0.5).abs() < 1e-6);
        assert!((output[2]).abs() < 1e-6);
    }

    #[test]
    fn test_fft_matches_direct_for_irregular_chunks() {
        let weight_count = if cfg!(miri) { 257 } else { 1_536 };
        let input_count = if cfg!(miri) { 300 } else { 4_096 };
        let weights = (0..weight_count)
            .map(|index| (-0.001 * index as f32).exp() * (0.037 * (index + 1) as f32).sin() * 0.01)
            .collect::<Vec<_>>();
        let input = (0..input_count)
            .map(|index| {
                (0.2 * (0.013 * index as f64).sin() + 0.05 * (0.071 * index as f64).cos()) as Sample
            })
            .collect::<Vec<_>>();
        let mut direct = make_linear_with_implementation(&weights, "direct");
        let mut fft = make_linear_with_implementation(&weights, "fft");

        let direct_output = render_in_chunks(&mut direct, &input, &[1, 17, 64, 255, 3, 512, 31]);
        let fft_output = render_in_chunks(&mut fft, &input, &[1, 17, 64, 255, 3, 512, 31]);
        let max_difference = direct_output
            .iter()
            .zip(&fft_output)
            .map(|(direct, fft)| (direct - fft).abs())
            .fold(0.0 as Sample, Sample::max);

        assert!(max_difference < 5.0e-5, "max difference: {max_difference}");
    }

    #[test]
    fn test_fft_is_block_partition_invariant() {
        let weight_count = if cfg!(miri) { 257 } else { 1_024 };
        let input_count = if cfg!(miri) { 300 } else { 3_000 };
        let weights = (0..weight_count)
            .map(|index| ((index + 1) as f32 * 0.017).sin() * 0.005)
            .collect::<Vec<_>>();
        let input = (0..input_count)
            .map(|index| ((index as f64 * 0.031).sin() * 0.2) as Sample)
            .collect::<Vec<_>>();
        let mut single = make_linear_with_implementation(&weights, "fft");
        let mut irregular = make_linear_with_implementation(&weights, "fft");

        let single_output = render_in_chunks(&mut single, &input, &[input.len()]);
        let irregular_output = render_in_chunks(&mut irregular, &input, &[1, 7, 64, 257, 3]);

        assert_eq!(single_output, irregular_output);
    }

    #[test]
    fn test_fft_reset_restores_fresh_state() {
        let weight_count = if cfg!(miri) { 257 } else { 1_024 };
        let input_count = if cfg!(miri) { 300 } else { 1_000 };
        let weights = vec![0.001; weight_count];
        let input = vec![0.25 as Sample; input_count];
        let mut model = make_linear_with_implementation(&weights, "fft");
        let first = render_in_chunks(&mut model, &input, &[63, 128, 7]);

        model.reset(48_000.0, 128);
        let reset = render_in_chunks(&mut model, &input, &[63, 128, 7]);

        assert_eq!(reset, first);
    }

    #[test]
    fn test_auto_selects_fft_only_for_long_filters() {
        assert!(make_linear(&vec![0.0; 256]).fft.is_none());
        assert!(make_linear(&vec![0.0; 257]).fft.is_some());
    }

    #[test]
    fn test_linear_rejects_unknown_implementation() {
        let config = serde_json::json!({
            "receptive_field": 1,
            "implementation": "gpu"
        });

        assert!(matches!(
            Linear::from_config(&config, &[1.0], DspMetadata::default()),
            Err(NamError::UnsupportedConfigValue { field, .. })
                if field == "implementation"
        ));
    }

    #[test]
    fn test_reset_clears_state() {
        let mut model = make_linear(&[0.5, 0.5]);
        let input = vec![1.0 as Sample];
        let mut output = vec![0.0 as Sample; 1];
        model.process(&input, &mut output);

        model.reset(48000.0, 1024);

        let input2 = vec![0.0 as Sample];
        let mut output2 = vec![0.0 as Sample; 1];
        model.process(&input2, &mut output2);
        assert!((output2[0]).abs() < 1e-6);
    }

    #[test]
    fn test_weight_mismatch_too_few() {
        let config = serde_json::json!({ "receptive_field": 5 });
        let result = Linear::from_config(&config, &[1.0, 2.0, 3.0], DspMetadata::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_weight_mismatch_too_many() {
        let config = serde_json::json!({ "receptive_field": 2 });
        let result =
            Linear::from_config(&config, &[1.0, 2.0, 3.0, 4.0, 5.0], DspMetadata::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_receptive_field_is_rejected_before_processing() {
        let config = serde_json::json!({ "receptive_field": 0 });

        assert!(matches!(
            Linear::from_config(&config, &[], DspMetadata::default()),
            Err(NamError::InvalidConfigField { field, .. })
                if field == "receptive_field"
        ));
    }

    #[test]
    fn test_process_empty_buffer() {
        let mut model = make_linear(&[1.0]);
        let input: Vec<Sample> = vec![];
        let mut output: Vec<Sample> = vec![];
        model.process(&input, &mut output);
    }

    #[test]
    fn test_prewarm_is_noop() {
        let mut model = make_linear(&[1.0]);
        assert_eq!(model.prewarm_samples(), 0);
        model.prewarm();
        let input = vec![1.0 as Sample];
        let mut output = vec![0.0 as Sample; 1];
        model.process(&input, &mut output);
        assert!((output[0] - 1.0).abs() < 1e-6);
    }
}
