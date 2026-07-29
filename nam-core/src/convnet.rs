use ndarray::{Array1, Array2, ArrayView1};

use crate::activations::Activation;
use crate::dsp::{ActivationMode, Dsp, DspMetadata, Sample};
use crate::error::NamError;
use crate::util::{
    checked_dimension_add, checked_dimension_mul, positive_config_usize, WeightIter,
};

/// Batch normalization (inference mode only): y = scale * x + loc.
struct BatchNorm {
    scale: Array1<f32>,
    loc: Array1<f32>,
}

impl BatchNorm {
    fn from_weights(dim: usize, iter: &mut WeightIter) -> Result<Self, NamError> {
        let running_mean = iter.take_vector(dim)?;
        let running_var = iter.take_vector(dim)?;
        let gamma = iter.take_vector(dim)?;
        let beta = iter.take_vector(dim)?;
        let eps = iter.take(1)?[0];

        let scale: Array1<f32> = &gamma / &running_var.mapv(|v| (eps + v).sqrt());
        let loc: Array1<f32> = &beta - &(&scale * &running_mean);

        Ok(Self { scale, loc })
    }

    #[inline]
    fn apply(&self, data: &mut [f32], channels: usize) {
        // data is [channels] for a single frame
        debug_assert_eq!(data.len(), channels);
        for (i, x) in data.iter_mut().enumerate() {
            *x = *x * self.scale[i] + self.loc[i];
        }
    }
}

/// 1D convolution with kernel_size=2 and given dilation.
struct Conv1d {
    /// Weight per kernel tap. Each is [out_channels, in_channels].
    /// For kernel_size=2: weights[0] is the older tap, weights[1] is the newer tap.
    weights: [Array2<f32>; 2],
    bias: Option<Array1<f32>>,
    #[allow(dead_code)] // retained for future block-based processing
    dilation: usize,
}

impl Conv1d {
    fn from_weights(
        in_channels: usize,
        out_channels: usize,
        dilation: usize,
        has_bias: bool,
        iter: &mut WeightIter,
    ) -> Result<Self, NamError> {
        checked_dimension_mul("ConvNet convolution weight", out_channels, in_channels)?;
        // Weight layout: for group 0 (groups=1): for out_ch, for in_ch, for kernel_tap
        let kernel_size = 2;
        let mut w0 = Array2::<f32>::zeros((out_channels, in_channels));
        let mut w1 = Array2::<f32>::zeros((out_channels, in_channels));

        for i in 0..out_channels {
            for j in 0..in_channels {
                let taps = iter.take(kernel_size)?;
                w0[[i, j]] = taps[0];
                w1[[i, j]] = taps[1];
            }
        }

        let bias = if has_bias {
            Some(iter.take_vector(out_channels)?)
        } else {
            None
        };

        Ok(Self {
            weights: [w0, w1],
            dilation,
            bias,
        })
    }

    /// Compute output for a single frame given the two input frames
    /// (the older one at `dilation` steps back and the current one).
    #[inline]
    fn forward_frame(&self, older: ArrayView1<f32>, newer: ArrayView1<f32>, out: &mut [f32]) {
        let out_ch = self.weights[0].nrows();
        for i in 0..out_ch {
            let mut sum = 0.0f32;
            for j in 0..older.len() {
                sum += self.weights[0][[i, j]] * older[j] + self.weights[1][[i, j]] * newer[j];
            }
            if let Some(ref bias) = self.bias {
                sum += bias[i];
            }
            out[i] = sum;
        }
    }
}

struct ConvNetBlock {
    conv: Conv1d,
    batchnorm: Option<BatchNorm>,
    activation: Activation,
    out_channels: usize,
}

impl ConvNetBlock {
    fn from_weights(
        in_channels: usize,
        out_channels: usize,
        dilation: usize,
        use_batchnorm: bool,
        activation: &Activation,
        iter: &mut WeightIter,
    ) -> Result<Self, NamError> {
        // Conv has bias only when batchnorm is NOT used
        let conv = Conv1d::from_weights(in_channels, out_channels, dilation, !use_batchnorm, iter)?;
        let batchnorm = if use_batchnorm {
            Some(BatchNorm::from_weights(out_channels, iter)?)
        } else {
            None
        };

        Ok(Self {
            conv,
            batchnorm,
            activation: activation.clone(),
            out_channels,
        })
    }
}

struct Head {
    weight: Array2<f32>, // [out_channels, in_channels]
    bias: Array1<f32>,   // [out_channels]
}

impl Head {
    fn from_weights(
        in_channels: usize,
        out_channels: usize,
        iter: &mut WeightIter,
    ) -> Result<Self, NamError> {
        let weight = iter.take_matrix(out_channels, in_channels)?;
        let bias = iter.take_vector(out_channels)?;
        Ok(Self { weight, bias })
    }

    #[inline]
    fn forward(&self, input: &[f32]) -> f32 {
        // For mono output (out_channels=1)
        let mut sum = self.bias[0];
        for (j, &w) in self.weight.row(0).iter().enumerate() {
            sum += w * input[j];
        }
        sum
    }
}

pub struct ConvNet {
    blocks: Vec<ConvNetBlock>,
    head: Head,
    /// Ring buffer: [max_channels, buffer_size]. Stores per-block output history.
    /// We keep separate buffers per block level for simplicity.
    /// Actually, we process sample-by-sample and keep a history buffer.
    history: Vec<Vec<Vec<f32>>>, // [block_level][frame_index][channels]
    #[allow(dead_code)]
    channels: usize,
    dilations: Vec<usize>,
    prewarm_samples: usize,
    metadata: DspMetadata,
    activation_mode: ActivationMode,
}

impl ConvNet {
    pub fn from_config(
        config: &serde_json::Value,
        weights: &[f32],
        metadata: DspMetadata,
    ) -> Result<Self, NamError> {
        let channels = positive_config_usize(&config["channels"], "channels")?;
        let dilations: Vec<usize> = config["dilations"]
            .as_array()
            .ok_or_else(|| NamError::MissingField("dilations".into()))?
            .iter()
            .enumerate()
            .map(|(index, value)| positive_config_usize(value, &format!("dilations[{index}]")))
            .collect::<Result<_, _>>()?;
        let use_batchnorm = config["batchnorm"].as_bool().unwrap_or(false);

        let activation_val = &config["activation"];
        let activation = if let Some(name) = activation_val.as_str() {
            Activation::from_name(name)?
        } else {
            Activation::from_name("Relu")?
        };

        let in_channels = config
            .get("in_channels")
            .map(|value| positive_config_usize(value, "in_channels"))
            .transpose()?
            .unwrap_or(1);
        let out_channels = config
            .get("out_channels")
            .map(|value| positive_config_usize(value, "out_channels"))
            .transpose()?
            .unwrap_or(1);
        let prewarm_samples = dilations.iter().try_fold(1usize, |total, &dilation| {
            checked_dimension_add("ConvNet prewarm", total, dilation)
        })?;
        let num_levels = checked_dimension_add("ConvNet history levels", dilations.len(), 1)?;
        for level in 0..num_levels {
            let level_channels = if level == 0 { in_channels } else { channels };
            let needed = if level < dilations.len() {
                checked_dimension_add("ConvNet history length", dilations[level], 1)?
            } else {
                1
            };
            checked_dimension_mul("ConvNet history", needed, level_channels)?;
        }

        let mut iter = WeightIter::new(weights);
        let mut blocks = Vec::with_capacity(dilations.len());

        for (i, &dilation) in dilations.iter().enumerate() {
            let block_in = if i == 0 { in_channels } else { channels };
            blocks.push(ConvNetBlock::from_weights(
                block_in,
                channels,
                dilation,
                use_batchnorm,
                &activation,
                &mut iter,
            )?);
        }

        let head = Head::from_weights(channels, out_channels, &mut iter)?;
        iter.assert_exhausted()?;

        // Compute max history needed: max dilation
        // We need history for each "level" (block input). The history length needed
        // at each level is the dilation of that block + 1.
        // But since blocks feed into each other sample-by-sample, we need a different approach.
        // We'll use a simple approach: process one sample at a time, maintaining ring buffers.

        // History buffers: for each block level (0 = input, 1..N = after block i-1)
        // we need dilation[i] + 1 frames of history.
        let mut history = Vec::with_capacity(num_levels);
        for level in 0..num_levels {
            let level_channels = if level == 0 { in_channels } else { channels };
            // Need enough history for the max dilation that reads from this level
            let needed = if level < dilations.len() {
                checked_dimension_add("ConvNet history length", dilations[level], 1)?
            } else {
                1 // last level is only read by head, needs 1 frame
            };
            checked_dimension_mul("ConvNet history", needed, level_channels)?;
            let buf: Vec<Vec<f32>> = (0..needed).map(|_| vec![0.0; level_channels]).collect();
            history.push(buf);
        }

        Ok(Self {
            blocks,
            head,
            history,
            channels,
            dilations: dilations.to_vec(),
            prewarm_samples,
            metadata,
            activation_mode: ActivationMode::Accurate,
        })
    }

    /// Process a single sample through the network.
    fn process_sample(&mut self, input_sample: f32) -> f32 {
        // Write input to level 0 history (rotate)
        self.history[0].rotate_right(1);
        self.history[0][0].fill(0.0);
        self.history[0][0][0] = input_sample;

        // Process through each block
        for block_idx in 0..self.blocks.len() {
            let dilation = self.dilations[block_idx];
            let out_ch = self.blocks[block_idx].out_channels;

            // Get older and newer frames from input history
            let older_idx = dilation; // dilation steps back
            let newer_idx = 0; // current

            // Conv1d forward
            let mut frame_out = vec![0.0f32; out_ch];
            {
                let history_level = &self.history[block_idx];
                let older = ArrayView1::from(&history_level[older_idx]);
                let newer = ArrayView1::from(&history_level[newer_idx]);
                self.blocks[block_idx]
                    .conv
                    .forward_frame(older, newer, &mut frame_out);
            }

            // Batchnorm
            if let Some(ref bn) = self.blocks[block_idx].batchnorm {
                bn.apply(&mut frame_out, out_ch);
            }

            // Activation
            self.blocks[block_idx]
                .activation
                .apply_slice_with_mode(&mut frame_out, self.activation_mode);

            // Write to next level's history
            let next_level = block_idx + 1;
            self.history[next_level].rotate_right(1);
            self.history[next_level][0] = frame_out;
        }

        // Head: reads current frame from last level
        let last_level = self.blocks.len();
        self.head.forward(&self.history[last_level][0])
    }
}

impl Dsp for ConvNet {
    fn process(&mut self, input: &[Sample], output: &mut [Sample]) {
        for (i, &sample) in input.iter().enumerate() {
            output[i] =
                crate::dsp::sample_from_f32(self.process_sample(crate::dsp::sample_to_f32(sample)));
        }
    }

    fn reset(&mut self, _sample_rate: f64, _max_buffer_size: usize) {
        for level in &mut self.history {
            for frame in level.iter_mut() {
                frame.fill(0.0);
            }
        }
    }

    fn prewarm_samples(&self) -> usize {
        self.prewarm_samples
    }

    fn metadata(&self) -> &DspMetadata {
        &self.metadata
    }

    fn set_activation_mode(&mut self, mode: ActivationMode) {
        self.activation_mode = mode;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal ConvNet from JSON config and weight vector.
    fn make_convnet(
        channels: usize,
        dilations: &[usize],
        batchnorm: bool,
        activation: &str,
        weights: &[f32],
    ) -> ConvNet {
        let config = serde_json::json!({
            "channels": channels,
            "dilations": dilations,
            "batchnorm": batchnorm,
            "activation": activation,
        });
        ConvNet::from_config(&config, weights, DspMetadata::default()).unwrap()
    }

    /// Compute expected weight count for a ConvNet.
    /// kernel_size=2, groups=1, in_channels=1 for first block.
    fn expected_weight_count(channels: usize, dilations: &[usize], batchnorm: bool) -> usize {
        let kernel_size = 2;
        let mut total = 0;
        for (i, _) in dilations.iter().enumerate() {
            let in_ch = if i == 0 { 1 } else { channels };
            // Conv weights: in_ch * channels * kernel_size
            total += in_ch * channels * kernel_size;
            if !batchnorm {
                // Conv bias
                total += channels;
            } else {
                // BatchNorm: running_mean + running_var + gamma + beta + eps
                total += channels * 4 + 1;
            }
        }
        // Head: channels * 1 (out_channels=1) + 1 (bias)
        total += channels + 1;
        total
    }

    #[test]
    fn test_convnet_basic() {
        let channels = 2;
        let dilations = [1];
        let n = expected_weight_count(channels, &dilations, false);
        let weights: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let mut model = make_convnet(channels, &dilations, false, "Relu", &weights);

        let input = vec![1.0 as Sample; 8];
        let mut output = vec![0.0 as Sample; 8];
        model.process(&input, &mut output);

        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_convnet_multiple_blocks() {
        let channels = 2;
        let dilations = [1, 2, 4];
        let n = expected_weight_count(channels, &dilations, false);
        let weights: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin()).collect();
        let mut model = make_convnet(channels, &dilations, false, "Tanh", &weights);

        let input = vec![0.5 as Sample; 16];
        let mut output = vec![0.0 as Sample; 16];
        model.process(&input, &mut output);

        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_convnet_with_batchnorm() {
        let channels = 2;
        let dilations = [1];
        let n = expected_weight_count(channels, &dilations, true);

        // Construct weights with sensible batchnorm params:
        // Conv weights (no bias when batchnorm): 1*2*2 = 4 weights
        // BN: running_mean(2), running_var(2), gamma(2), beta(2), eps(1) = 9
        // Head: 2+1 = 3
        // Total = 4 + 9 + 3 = 16
        assert_eq!(n, 16);
        let mut weights = vec![0.1f32; 4]; // conv weights
        weights.extend_from_slice(&[0.0, 0.0]); // running_mean
        weights.extend_from_slice(&[1.0, 1.0]); // running_var
        weights.extend_from_slice(&[1.0, 1.0]); // gamma
        weights.extend_from_slice(&[0.0, 0.0]); // beta
        weights.push(1e-5); // eps
        weights.extend_from_slice(&[0.5, 0.5]); // head weight
        weights.push(0.0); // head bias

        let mut model = make_convnet(channels, &dilations, true, "Relu", &weights);

        let input = vec![1.0 as Sample; 8];
        let mut output = vec![0.0 as Sample; 8];
        model.process(&input, &mut output);

        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_convnet_zero_input() {
        let channels = 2;
        let dilations = [1, 2];
        let n = expected_weight_count(channels, &dilations, false);
        let weights: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let mut model = make_convnet(channels, &dilations, false, "Relu", &weights);

        let input = vec![0.0 as Sample; 16];
        let mut output = vec![0.0 as Sample; 16];
        model.process(&input, &mut output);

        // All outputs should be finite (bias-only contribution through ReLU)
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_convnet_prewarm() {
        let channels = 2;
        let dilations = [1, 2];
        let n = expected_weight_count(channels, &dilations, false);
        let weights: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let mut model = make_convnet(channels, &dilations, false, "Relu", &weights);

        assert_eq!(model.prewarm_samples(), 1 + 1 + 2);
        model.prewarm();

        let input = vec![0.5 as Sample; 8];
        let mut output = vec![0.0 as Sample; 8];
        model.process(&input, &mut output);

        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_convnet_state_persistence() {
        let channels = 2;
        let dilations = [1, 2];
        let n = expected_weight_count(channels, &dilations, false);
        let weights: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).sin()).collect();

        // Split processing
        let config = serde_json::json!({
            "channels": channels,
            "dilations": dilations,
            "batchnorm": false,
            "activation": "Tanh",
        });
        let mut model_split =
            ConvNet::from_config(&config, &weights, DspMetadata::default()).unwrap();
        let in1 = vec![1.0 as Sample; 4];
        let mut out1 = vec![0.0 as Sample; 4];
        model_split.process(&in1, &mut out1);
        let in2 = vec![0.0 as Sample; 4];
        let mut out2 = vec![0.0 as Sample; 4];
        model_split.process(&in2, &mut out2);

        // Full processing
        let mut model_full =
            ConvNet::from_config(&config, &weights, DspMetadata::default()).unwrap();
        let mut full_in = vec![1.0 as Sample; 4];
        full_in.extend(vec![0.0 as Sample; 4]);
        let mut full_out = vec![0.0 as Sample; 8];
        model_full.process(&full_in, &mut full_out);

        for i in 0..4 {
            assert!(
                (out2[i] - full_out[4 + i]).abs() < 1e-5,
                "ConvNet state mismatch at {}: split={}, full={}",
                i,
                out2[i],
                full_out[4 + i]
            );
        }
    }

    #[test]
    fn test_convnet_reset() {
        let channels = 2;
        let dilations = [1];
        let n = expected_weight_count(channels, &dilations, false);
        let weights: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).sin()).collect();
        let config = serde_json::json!({
            "channels": channels, "dilations": dilations, "batchnorm": false, "activation": "Tanh",
        });

        let mut model = ConvNet::from_config(&config, &weights, DspMetadata::default()).unwrap();
        let input = vec![1.0 as Sample; 8];
        let mut output = vec![0.0 as Sample; 8];
        model.process(&input, &mut output);

        model.reset(48000.0, 4096);

        let mut model_fresh =
            ConvNet::from_config(&config, &weights, DspMetadata::default()).unwrap();
        let mut out_reset = vec![0.0 as Sample; 8];
        let mut out_fresh = vec![0.0 as Sample; 8];
        model.process(&input, &mut out_reset);
        model_fresh.process(&input, &mut out_fresh);

        for i in 0..8 {
            assert!(
                (out_reset[i] - out_fresh[i]).abs() < 1e-5,
                "ConvNet reset mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_convnet_process_empty_buffer() {
        let channels = 2;
        let dilations = [1];
        let n = expected_weight_count(channels, &dilations, false);
        let weights: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let mut model = make_convnet(channels, &dilations, false, "Relu", &weights);
        let input: Vec<Sample> = vec![];
        let mut output: Vec<Sample> = vec![];
        model.process(&input, &mut output);
    }

    #[test]
    fn test_convnet_rejects_prewarm_overflow_before_reading_weights() {
        let config = serde_json::json!({
            "channels": 1,
            "dilations": [u64::MAX],
            "batchnorm": false,
            "activation": "Tanh",
        });

        let result = ConvNet::from_config(&config, &[], DspMetadata::default());

        #[cfg(target_pointer_width = "64")]
        assert!(matches!(
            result,
            Err(NamError::DimensionOverflow {
                context: "ConvNet prewarm",
                ..
            })
        ));
        #[cfg(not(target_pointer_width = "64"))]
        assert!(matches!(
            result,
            Err(NamError::ConfigIntegerOutOfRange { .. })
        ));
    }
}
