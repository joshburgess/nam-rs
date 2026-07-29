use crate::dsp::{Dsp, DspMetadata, Sample};
use crate::error::NamError;
use crate::util::{positive_config_usize, WeightIter};

pub struct Linear {
    weights: Vec<f32>,
    bias: f32,
    history: Vec<f32>,
    history_pos: usize,
    metadata: DspMetadata,
}

impl Linear {
    pub fn from_config(
        config: &serde_json::Value,
        weights: &[f32],
        metadata: DspMetadata,
    ) -> Result<Self, NamError> {
        let receptive_field = positive_config_usize(&config["receptive_field"], "receptive_field")?;
        let has_bias = config["bias"].as_bool().unwrap_or(false);

        let mut iter = WeightIter::new(weights);
        let w = iter.take(receptive_field)?;
        let w = w.to_vec();
        let bias = if has_bias { iter.take(1)?[0] } else { 0.0 };
        iter.assert_exhausted()?;

        Ok(Self {
            weights: w,
            bias,
            history: vec![0.0; receptive_field],
            history_pos: 0,
            metadata,
        })
    }
}

impl Dsp for Linear {
    fn process(&mut self, input: &[Sample], output: &mut [Sample]) {
        let len = self.weights.len();
        for (i, &sample) in input.iter().enumerate() {
            self.history[self.history_pos] = crate::dsp::sample_to_f32(sample);

            let mut sum = 0.0f32;
            for (j, &w) in self.weights.iter().enumerate() {
                let idx = (self.history_pos + len - j) % len;
                sum += w * self.history[idx];
            }
            sum += self.bias;
            output[i] = sum as Sample;

            self.history_pos = (self.history_pos + 1) % len;
        }
    }

    fn reset(&mut self, _sample_rate: f64, _max_buffer_size: usize) {
        self.history.fill(0.0);
        self.history_pos = 0;
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
