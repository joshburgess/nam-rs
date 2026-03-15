use crate::dsp::{Dsp, DspMetadata, Sample};
use crate::error::NamError;
use crate::util::WeightIter;

pub struct Linear {
    weights: Vec<f32>,
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
        let receptive_field = config["receptive_field"]
            .as_u64()
            .ok_or_else(|| NamError::MissingField("receptive_field".into()))?
            as usize;

        let mut iter = WeightIter::new(weights);
        let w = iter.take(receptive_field)?;
        let w = w.to_vec();
        iter.assert_exhausted()?;

        Ok(Self {
            weights: w,
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
            self.history[self.history_pos] = sample as f32;

            let mut sum = 0.0f32;
            for (j, &w) in self.weights.iter().enumerate() {
                let idx = (self.history_pos + len - j) % len;
                sum += w * self.history[idx];
            }
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
