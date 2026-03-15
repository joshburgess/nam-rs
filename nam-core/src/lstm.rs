use ndarray::{Array1, Array2};

use crate::dsp::{Dsp, DspMetadata, Sample};
use crate::error::NamError;
use crate::util::{sigmoid, WeightIter};

struct LstmCell {
    /// Combined weight matrix [4*hidden, input_size + hidden_size], row-major.
    /// Gate order: IFGO (Input, Forget, Gate/cell, Output).
    w: Array2<f32>,
    /// Combined bias [4*hidden].
    b: Array1<f32>,
    /// Concatenated [input | hidden_state].
    xh: Array1<f32>,
    /// Cell state.
    c: Array1<f32>,
    /// Scratch for gate pre-activations.
    ifgo: Array1<f32>,
    input_size: usize,
    hidden_size: usize,
}

impl LstmCell {
    fn new(
        w: Array2<f32>,
        b: Array1<f32>,
        initial_hidden: Array1<f32>,
        initial_cell: Array1<f32>,
        input_size: usize,
        hidden_size: usize,
    ) -> Self {
        let mut xh = Array1::zeros(input_size + hidden_size);
        xh.slice_mut(ndarray::s![input_size..])
            .assign(&initial_hidden);
        Self {
            w,
            b,
            xh,
            c: initial_cell,
            ifgo: Array1::zeros(4 * hidden_size),
            input_size,
            hidden_size,
        }
    }

    /// Process one sample through this cell.
    #[inline]
    fn process(&mut self, input: &Array1<f32>) {
        let h = self.hidden_size;

        // Copy input into xh
        self.xh
            .slice_mut(ndarray::s![..self.input_size])
            .assign(input);

        // ifgo = W @ xh + b
        ndarray::linalg::general_mat_vec_mul(1.0, &self.w, &self.xh, 0.0, &mut self.ifgo);
        self.ifgo += &self.b;

        // Apply gate activations and update cell/hidden state
        for i in 0..h {
            let ig = sigmoid(self.ifgo[i]); // input gate
            let fg = sigmoid(self.ifgo[i + h]); // forget gate
            let gg = self.ifgo[i + 2 * h].tanh(); // cell gate
            let og = sigmoid(self.ifgo[i + 3 * h]); // output gate

            self.c[i] = fg * self.c[i] + ig * gg;
            self.xh[self.input_size + i] = og * self.c[i].tanh();
        }
    }

    fn hidden_state(&self) -> ndarray::ArrayView1<'_, f32> {
        self.xh.slice(ndarray::s![self.input_size..])
    }
}

pub struct Lstm {
    cells: Vec<LstmCell>,
    head_weight: Array2<f32>,
    head_bias: Array1<f32>,
    /// Scratch buffer for single-sample input.
    input_buf: Array1<f32>,
    metadata: DspMetadata,
    expected_sample_rate: f64,
}

impl Lstm {
    pub fn from_config(
        config: &serde_json::Value,
        weights: &[f32],
        metadata: DspMetadata,
    ) -> Result<Self, NamError> {
        let num_layers = config["num_layers"]
            .as_u64()
            .ok_or_else(|| NamError::MissingField("num_layers".into()))?
            as usize;
        let input_size = config["input_size"]
            .as_u64()
            .ok_or_else(|| NamError::MissingField("input_size".into()))?
            as usize;
        let hidden_size = config["hidden_size"]
            .as_u64()
            .ok_or_else(|| NamError::MissingField("hidden_size".into()))?
            as usize;

        let mut iter = WeightIter::new(weights);
        let mut cells = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let layer_input_size = if i == 0 { input_size } else { hidden_size };
            let w = iter.take_matrix(4 * hidden_size, layer_input_size + hidden_size)?;
            let b = iter.take_vector(4 * hidden_size)?;
            let initial_hidden = iter.take_vector(hidden_size)?;
            let initial_cell = iter.take_vector(hidden_size)?;

            cells.push(LstmCell::new(
                w,
                b,
                initial_hidden,
                initial_cell,
                layer_input_size,
                hidden_size,
            ));
        }

        // Head: out_channels is typically 1
        let out_channels = 1;
        let head_weight = iter.take_matrix(out_channels, hidden_size)?;
        let head_bias = iter.take_vector(out_channels)?;

        iter.assert_exhausted()?;

        let expected_sample_rate = metadata.expected_sample_rate.unwrap_or(48000.0);

        Ok(Self {
            cells,
            head_weight,
            head_bias,
            input_buf: Array1::zeros(input_size),
            metadata,
            expected_sample_rate,
        })
    }
}

impl Dsp for Lstm {
    fn process(&mut self, input: &[Sample], output: &mut [Sample]) {
        for (i, &sample) in input.iter().enumerate() {
            self.input_buf[0] = sample as f32;

            // Forward through LSTM layers
            self.cells[0].process(&self.input_buf);
            for layer in 1..self.cells.len() {
                let prev_hidden = self.cells[layer - 1].hidden_state().to_owned();
                self.cells[layer].process(&prev_hidden);
            }

            // Head linear layer
            let final_hidden = self.cells.last().unwrap().hidden_state();
            let out = self.head_weight.row(0).dot(&final_hidden) + self.head_bias[0];
            output[i] = out as Sample;
        }
    }

    fn reset(&mut self, _sample_rate: f64, _max_buffer_size: usize) {
        // LSTM state persists (initial state was set at construction).
        // A full reset would require re-loading the model.
    }

    fn prewarm_samples(&self) -> usize {
        let n = (0.5 * self.expected_sample_rate) as usize;
        if n == 0 {
            1
        } else {
            n
        }
    }

    fn metadata(&self) -> &DspMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_lstm_loads() {
        let path = Path::new("test_fixtures/models/lstm.nam");
        if !path.exists() {
            eprintln!("Skipping test: {:?} not found", path);
            return;
        }
        let content = std::fs::read_to_string(path).unwrap();
        let root: serde_json::Value = serde_json::from_str(&content).unwrap();
        let weights: Vec<f32> = root["weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let metadata = DspMetadata::default();
        let config = &root["config"];

        let mut model = Lstm::from_config(config, &weights, metadata).unwrap();

        // Process some silence — should not panic
        let input = vec![0.0 as Sample; 64];
        let mut output = vec![0.0 as Sample; 64];
        model.process(&input, &mut output);

        // Process a simple impulse
        let mut impulse = vec![0.0 as Sample; 64];
        impulse[0] = 1.0 as Sample;
        let mut out2 = vec![0.0 as Sample; 64];
        model.process(&impulse, &mut out2);

        // Output should not be all zeros after impulse
        let has_nonzero = out2.iter().any(|&x| x != 0.0);
        assert!(has_nonzero, "LSTM output was all zeros after impulse");
    }

    #[test]
    fn test_lstm_single_step_hand_computed() {
        // Tiny LSTM: input_size=1, hidden_size=1, num_layers=1
        // Gate order: IFGO. All biases=0, initial h=0, initial c=0.
        let w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; // [4, 2] row-major
        let b = [0.0f32; 4];
        let h0 = [0.0f32];
        let c0 = [0.0f32];
        let head_w = [1.0f32];
        let head_b = [0.0f32];

        let mut weights: Vec<f32> = Vec::new();
        weights.extend_from_slice(&w);
        weights.extend_from_slice(&b);
        weights.extend_from_slice(&h0);
        weights.extend_from_slice(&c0);
        weights.extend_from_slice(&head_w);
        weights.extend_from_slice(&head_b);

        let config = serde_json::json!({
            "input_size": 1, "hidden_size": 1, "num_layers": 1
        });
        let mut model = Lstm::from_config(&config, &weights, DspMetadata::default()).unwrap();

        let input = vec![1.0 as Sample];
        let mut output = vec![0.0 as Sample; 1];
        model.process(&input, &mut output);

        let expected = {
            let i = 1.0f32 / (1.0 + (-0.1f32).exp());
            let f = 1.0 / (1.0 + (-0.3f32).exp());
            let g = 0.5f32.tanh();
            let o = 1.0 / (1.0 + (-0.7f32).exp());
            let c = f * 0.0 + i * g;
            o * c.tanh()
        };

        assert!(
            (output[0] as f32 - expected).abs() < 1e-5,
            "LSTM output {} != expected {}",
            output[0],
            expected
        );
    }

    #[test]
    fn test_lstm_state_persists_across_calls() {
        let path = Path::new("test_fixtures/models/lstm.nam");
        if !path.exists() {
            return;
        }

        let mut model_a = crate::get_dsp(path).unwrap();
        let input1 = vec![1.0 as Sample; 32];
        let mut output1 = vec![0.0 as Sample; 32];
        model_a.process(&input1, &mut output1);
        let input2 = vec![0.0 as Sample; 32];
        let mut output2a = vec![0.0 as Sample; 32];
        model_a.process(&input2, &mut output2a);

        let mut model_b = crate::get_dsp(path).unwrap();
        let mut full_input = vec![1.0 as Sample; 32];
        full_input.extend(vec![0.0 as Sample; 32]);
        let mut full_output = vec![0.0 as Sample; 64];
        model_b.process(&full_input, &mut full_output);

        for i in 0..32 {
            assert!(
                (output2a[i] - full_output[32 + i]).abs() < 1e-5,
                "Sample {} mismatch: split={}, full={}",
                i,
                output2a[i],
                full_output[32 + i]
            );
        }
    }

    #[test]
    fn test_lstm_silence_stabilizes() {
        let path = Path::new("test_fixtures/models/lstm.nam");
        if !path.exists() {
            return;
        }
        let mut model = crate::get_dsp(path).unwrap();
        model.prewarm();

        let silence = vec![0.0 as Sample; 256];
        let mut output = vec![0.0 as Sample; 256];
        model.process(&silence, &mut output);

        assert!(
            (output[255] - output[254]).abs() < 1e-4,
            "Output should stabilize: {} vs {}",
            output[255],
            output[254]
        );
    }

    #[test]
    fn test_lstm_prewarm_samples_positive() {
        let path = Path::new("test_fixtures/models/lstm.nam");
        if !path.exists() {
            return;
        }
        let model = crate::get_dsp(path).unwrap();
        assert!(model.prewarm_samples() > 0);
    }

    #[test]
    fn test_lstm_process_empty_buffer() {
        let path = Path::new("test_fixtures/models/lstm.nam");
        if !path.exists() {
            return;
        }
        let mut model = crate::get_dsp(path).unwrap();
        let input: Vec<Sample> = vec![];
        let mut output: Vec<Sample> = vec![];
        model.process(&input, &mut output);
    }

    #[test]
    fn test_lstm_zero_input() {
        // Build synthetic LSTM with known weights, feed zeros
        let config = serde_json::json!({
            "input_size": 1, "hidden_size": 2, "num_layers": 1
        });
        // Weight count: W[4*2, 1+2]=24, b[8], h0[2], c0[2], head_w[1*2]=2, head_b[1] = 39
        let weights = vec![0.1f32; 39];
        let mut model = Lstm::from_config(&config, &weights, DspMetadata::default()).unwrap();

        let input = vec![0.0 as Sample; 16];
        let mut output = vec![0.0 as Sample; 16];
        model.process(&input, &mut output);

        assert!(output.iter().all(|&x| (x as f64).is_finite()));
    }

    #[test]
    fn test_lstm_multiple_layers() {
        let config = serde_json::json!({
            "input_size": 1, "hidden_size": 2, "num_layers": 2
        });
        // Layer 0: W[8, 3]=24, b[8], h0[2], c0[2] = 36
        // Layer 1: W[8, 4]=32, b[8], h0[2], c0[2] = 44
        // Head: w[2], b[1] = 3
        // Total = 36 + 44 + 3 = 83
        let weights = vec![0.05f32; 83];
        let mut model = Lstm::from_config(&config, &weights, DspMetadata::default()).unwrap();

        let input = vec![0.5 as Sample; 8];
        let mut output = vec![0.0 as Sample; 8];
        model.process(&input, &mut output);

        assert!(output.iter().all(|&x| (x as f64).is_finite()));
        assert!(output.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_lstm_two_steps_hand_computed() {
        // Verify second step uses updated hidden/cell state from first step
        let w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let b = [0.0f32; 4];
        let h0 = [0.0f32];
        let c0 = [0.0f32];
        let head_w = [1.0f32];
        let head_b = [0.0f32];

        let mut weights: Vec<f32> = Vec::new();
        weights.extend_from_slice(&w);
        weights.extend_from_slice(&b);
        weights.extend_from_slice(&h0);
        weights.extend_from_slice(&c0);
        weights.extend_from_slice(&head_w);
        weights.extend_from_slice(&head_b);

        let config = serde_json::json!({
            "input_size": 1, "hidden_size": 1, "num_layers": 1
        });
        let mut model = Lstm::from_config(&config, &weights, DspMetadata::default()).unwrap();

        let input = vec![1.0 as Sample, 0.5 as Sample];
        let mut output = vec![0.0 as Sample; 2];
        model.process(&input, &mut output);

        // Step 1: x=1.0, h_prev=0
        let i1 = 1.0f32 / (1.0 + (-0.1f32).exp());
        let f1 = 1.0 / (1.0 + (-0.3f32).exp());
        let g1 = 0.5f32.tanh();
        let o1 = 1.0 / (1.0 + (-0.7f32).exp());
        let c1 = f1 * 0.0 + i1 * g1;
        let h1 = o1 * c1.tanh();

        assert!((output[0] as f32 - h1).abs() < 1e-5);

        // Step 2: x=0.5, h_prev=h1, c_prev=c1
        let i_pre = 0.1 * 0.5 + 0.2 * h1;
        let f_pre = 0.3 * 0.5 + 0.4 * h1;
        let g_pre = 0.5 * 0.5 + 0.6 * h1;
        let o_pre = 0.7 * 0.5 + 0.8 * h1;
        let i2 = 1.0 / (1.0 + (-i_pre).exp());
        let f2 = 1.0 / (1.0 + (-f_pre).exp());
        let g2 = g_pre.tanh();
        let o2 = 1.0 / (1.0 + (-o_pre).exp());
        let c2 = f2 * c1 + i2 * g2;
        let h2 = o2 * c2.tanh();

        assert!(
            (output[1] as f32 - h2).abs() < 1e-5,
            "Step 2: output {} != expected {}",
            output[1],
            h2
        );
    }

    #[test]
    fn test_lstm_state_evolution() {
        // Sine wave input should produce evolving output
        let path = Path::new("test_fixtures/models/lstm.nam");
        if !path.exists() {
            return;
        }
        let mut model = crate::get_dsp(path).unwrap();

        let input: Vec<Sample> = (0..32).map(|i| (i as f64 * 0.3).sin() as Sample).collect();
        let mut output = vec![0.0 as Sample; 32];
        model.process(&input, &mut output);

        assert!(output.iter().all(|&x| (x as f64).is_finite()));
        // Output should not be constant
        let min = output
            .iter()
            .copied()
            .fold(f64::INFINITY, |a, b| a.min(b as f64));
        let max = output
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |a, b| a.max(b as f64));
        assert!(max - min > 1e-6, "LSTM output should vary with sine input");
    }

    #[test]
    fn test_lstm_different_buffer_sizes() {
        let path = Path::new("test_fixtures/models/lstm.nam");
        if !path.exists() {
            return;
        }
        let mut model = crate::get_dsp(path).unwrap();

        // Process with various buffer sizes
        for &size in &[1, 7, 16, 64, 128] {
            let input = vec![0.1 as Sample; size];
            let mut output = vec![0.0 as Sample; size];
            model.process(&input, &mut output);
            assert!(
                output.iter().all(|&x| (x as f64).is_finite()),
                "Non-finite output at buffer size {}",
                size
            );
        }
    }
}
