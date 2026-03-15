use crate::activations::Activation;
use crate::dsp::{Dsp, DspMetadata, Sample};
use crate::error::NamError;
use crate::util::WeightIter;

// ── Gating mode ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GatingMode {
    None,
    Gated,
    Blended,
}

// ── FiLM parameters (parsed from JSON) ──────────────────────────────────────

#[derive(Debug, Clone)]
struct FiLMParams {
    active: bool,
    shift: bool,
    groups: usize,
}

impl FiLMParams {
    fn inactive() -> Self {
        Self {
            active: false,
            shift: false,
            groups: 1,
        }
    }

    fn from_json(val: &serde_json::Value) -> Self {
        if val.is_boolean() && !val.as_bool().unwrap_or(false) {
            return Self::inactive();
        }
        if val.is_null() {
            return Self::inactive();
        }
        if let Some(obj) = val.as_object() {
            let active = obj.get("active").and_then(|v| v.as_bool()).unwrap_or(true);
            let shift = obj.get("shift").and_then(|v| v.as_bool()).unwrap_or(true);
            let groups = obj.get("groups").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
            Self {
                active,
                shift,
                groups,
            }
        } else {
            Self::inactive()
        }
    }
}

// ── Conv1x1 (with groups support) ───────────────────────────────────────────

/// 1x1 convolution (pointwise linear layer) with optional grouped convolution.
/// Weight matrix is stored as full [out_channels, in_channels] with block-diagonal
/// structure for grouped conv (zeros off-diagonal).
struct Conv1x1 {
    weight: Vec<f32>, // [out_channels * in_channels] row-major (or block-diagonal)
    bias: Option<Vec<f32>>,
    out_channels: usize,
    in_channels: usize,
    groups: usize,
}

impl Conv1x1 {
    fn from_weights(
        in_channels: usize,
        out_channels: usize,
        has_bias: bool,
        groups: usize,
        iter: &mut WeightIter,
    ) -> Result<Self, NamError> {
        let out_per_group = out_channels / groups;
        let in_per_group = in_channels / groups;

        // Store as full matrix with block-diagonal structure
        let mut weight = vec![0.0f32; out_channels * in_channels];

        // C++ weight order: for group, for out_per_group, for in_per_group
        for g in 0..groups {
            for i in 0..out_per_group {
                for j in 0..in_per_group {
                    let val = iter.take(1)?[0];
                    let row = g * out_per_group + i;
                    let col = g * in_per_group + j;
                    weight[row * in_channels + col] = val;
                }
            }
        }

        let bias = if has_bias {
            Some(iter.take(out_channels)?.to_vec())
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            out_channels,
            in_channels,
            groups,
        })
    }

    /// out = W @ in (+ bias). Single frame.
    #[inline]
    #[allow(clippy::needless_range_loop)]
    fn forward(&self, input: &[f32], output: &mut [f32]) {
        let out_ch = self.out_channels;
        let in_ch = self.in_channels;

        if self.groups == 1 {
            for i in 0..out_ch {
                let mut sum = 0.0f32;
                let row_start = i * in_ch;
                for j in 0..in_ch {
                    sum += self.weight[row_start + j] * input[j];
                }
                if let Some(ref b) = self.bias {
                    sum += b[i];
                }
                output[i] = sum;
            }
        } else {
            let out_per_group = out_ch / self.groups;
            let in_per_group = in_ch / self.groups;
            for g in 0..self.groups {
                for i in 0..out_per_group {
                    let out_idx = g * out_per_group + i;
                    let mut sum = 0.0f32;
                    let row_start = out_idx * in_ch;
                    let in_start = g * in_per_group;
                    for j in 0..in_per_group {
                        sum += self.weight[row_start + in_start + j] * input[in_start + j];
                    }
                    if let Some(ref b) = self.bias {
                        sum += b[out_idx];
                    }
                    output[out_idx] = sum;
                }
            }
        }
    }
}

// ── Conv1D (dilated, with groups support) ───────────────────────────────────

/// Dilated 1D convolution with ring buffer. Supports grouped convolution.
/// Weight stored per kernel tap as full [out_channels, in_channels] block-diagonal.
struct Conv1d {
    /// Weight per kernel tap. weights[k] is [out_channels * in_channels] row-major.
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    kernel_size: usize,
    dilation: usize,
    out_channels: usize,
    in_channels: usize,
    #[allow(dead_code)] // retained for future block-based processing
    groups: usize,
}

impl Conv1d {
    fn from_weights(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        groups: usize,
        iter: &mut WeightIter,
    ) -> Result<Self, NamError> {
        let out_per_group = out_channels / groups;
        let in_per_group = in_channels / groups;

        let mut tap_weights: Vec<Vec<f32>> = (0..kernel_size)
            .map(|_| vec![0.0f32; out_channels * in_channels])
            .collect();

        // C++ weight order: for group, for out_per_group, for in_per_group, for kernel_tap
        for g in 0..groups {
            for i in 0..out_per_group {
                for j in 0..in_per_group {
                    let taps = iter.take(kernel_size)?;
                    let row = g * out_per_group + i;
                    let col = g * in_per_group + j;
                    for k in 0..kernel_size {
                        tap_weights[k][row * in_channels + col] = taps[k];
                    }
                }
            }
        }

        let bias = iter.take(out_channels)?.to_vec();

        Ok(Self {
            weights: tap_weights,
            bias,
            kernel_size,
            dilation,
            out_channels,
            in_channels,
            groups,
        })
    }

    /// Receptive field (zero-indexed): dilation * (kernel_size - 1).
    fn receptive_field(&self) -> usize {
        self.dilation * (self.kernel_size - 1)
    }
}

// ── FiLM (Feature-wise Linear Modulation) ───────────────────────────────────

struct FiLM {
    cond_to_scale_shift: Conv1x1,
    do_shift: bool,
    input_dim: usize,
}

impl FiLM {
    fn from_weights(
        condition_dim: usize,
        input_dim: usize,
        shift: bool,
        groups: usize,
        iter: &mut WeightIter,
    ) -> Result<Self, NamError> {
        let out_channels = if shift { 2 * input_dim } else { input_dim };
        let cond_to_scale_shift =
            Conv1x1::from_weights(condition_dim, out_channels, true, groups, iter)?;
        Ok(Self {
            cond_to_scale_shift,
            do_shift: shift,
            input_dim,
        })
    }

    /// Apply FiLM: output = input * scale (+ shift)
    /// scale_shift_buf must be at least cond_to_scale_shift.out_channels long.
    #[inline]
    fn forward(
        &self,
        input: &[f32],
        condition: &[f32],
        output: &mut [f32],
        scale_shift_buf: &mut [f32],
    ) {
        let ss_len = self.cond_to_scale_shift.out_channels;
        self.cond_to_scale_shift
            .forward(condition, &mut scale_shift_buf[..ss_len]);

        let dim = self.input_dim;
        if self.do_shift {
            for i in 0..dim {
                output[i] = input[i] * scale_shift_buf[i] + scale_shift_buf[dim + i];
            }
        } else {
            for i in 0..dim {
                output[i] = input[i] * scale_shift_buf[i];
            }
        }
    }
}

// ── WaveNet Layer ────────────────────────────────────────────────────────────

struct WaveNetLayer {
    conv: Conv1d,
    input_mixin: Conv1x1,
    layer1x1: Option<Conv1x1>,
    head1x1: Option<Conv1x1>,
    activation: Activation,
    secondary_activation: Activation,
    gating_mode: GatingMode,
    #[allow(dead_code)]
    channels: usize,
    bottleneck: usize,
    #[allow(dead_code)]
    head_output_size: usize, // head1x1 out_channels if active, else bottleneck

    // FiLM modules (optional)
    conv_pre_film: Option<FiLM>,
    conv_post_film: Option<FiLM>,
    input_mixin_pre_film: Option<FiLM>,
    input_mixin_post_film: Option<FiLM>,
    activation_pre_film: Option<FiLM>,
    activation_post_film: Option<FiLM>,
    layer1x1_post_film: Option<FiLM>,
    head1x1_post_film: Option<FiLM>,
}

impl WaveNetLayer {
    #[allow(clippy::too_many_arguments)]
    fn from_weights(
        channels: usize,
        bottleneck: usize,
        condition_size: usize,
        kernel_size: usize,
        dilation: usize,
        activation: &Activation,
        gating_mode: GatingMode,
        groups_input: usize,
        groups_input_mixin: usize,
        has_layer1x1: bool,
        layer1x1_groups: usize,
        head1x1_params: &Head1x1Params,
        secondary_activation: &Activation,
        film_params: &LayerFiLMParams,
        iter: &mut WeightIter,
    ) -> Result<Self, NamError> {
        let conv_out = if gating_mode != GatingMode::None {
            2 * bottleneck
        } else {
            bottleneck
        };

        // 1. Conv weights
        let conv = Conv1d::from_weights(
            channels,
            conv_out,
            kernel_size,
            dilation,
            groups_input,
            iter,
        )?;

        // 2. Input mixin weights
        let input_mixin =
            Conv1x1::from_weights(condition_size, conv_out, false, groups_input_mixin, iter)?;

        // 3. Layer1x1 weights (if active)
        let layer1x1 = if has_layer1x1 {
            Some(Conv1x1::from_weights(
                bottleneck,
                channels,
                true,
                layer1x1_groups,
                iter,
            )?)
        } else {
            None
        };

        // 4. Head1x1 weights (if active)
        let head1x1 = if head1x1_params.active {
            Some(Conv1x1::from_weights(
                bottleneck,
                head1x1_params.out_channels,
                true,
                head1x1_params.groups,
                iter,
            )?)
        } else {
            None
        };

        let head_output_size = if head1x1_params.active {
            head1x1_params.out_channels
        } else {
            bottleneck
        };

        // 5. FiLM weights in order: conv_pre, conv_post, input_mixin_pre, input_mixin_post,
        //    activation_pre, activation_post, layer1x1_post, head1x1_post
        let conv_pre_film = if film_params.conv_pre.active {
            Some(FiLM::from_weights(
                condition_size,
                channels,
                film_params.conv_pre.shift,
                film_params.conv_pre.groups,
                iter,
            )?)
        } else {
            None
        };

        let conv_post_film = if film_params.conv_post.active {
            Some(FiLM::from_weights(
                condition_size,
                conv_out,
                film_params.conv_post.shift,
                film_params.conv_post.groups,
                iter,
            )?)
        } else {
            None
        };

        let input_mixin_pre_film = if film_params.input_mixin_pre.active {
            Some(FiLM::from_weights(
                condition_size,
                condition_size,
                film_params.input_mixin_pre.shift,
                film_params.input_mixin_pre.groups,
                iter,
            )?)
        } else {
            None
        };

        let input_mixin_post_film = if film_params.input_mixin_post.active {
            Some(FiLM::from_weights(
                condition_size,
                conv_out,
                film_params.input_mixin_post.shift,
                film_params.input_mixin_post.groups,
                iter,
            )?)
        } else {
            None
        };

        let activation_pre_film = if film_params.activation_pre.active {
            let z_channels = conv_out;
            Some(FiLM::from_weights(
                condition_size,
                z_channels,
                film_params.activation_pre.shift,
                film_params.activation_pre.groups,
                iter,
            )?)
        } else {
            None
        };

        let activation_post_film = if film_params.activation_post.active {
            Some(FiLM::from_weights(
                condition_size,
                bottleneck,
                film_params.activation_post.shift,
                film_params.activation_post.groups,
                iter,
            )?)
        } else {
            None
        };

        let layer1x1_post_film = if film_params.layer1x1_post.active && has_layer1x1 {
            Some(FiLM::from_weights(
                condition_size,
                channels,
                film_params.layer1x1_post.shift,
                film_params.layer1x1_post.groups,
                iter,
            )?)
        } else {
            None
        };

        let head1x1_post_film = if film_params.head1x1_post.active && head1x1_params.active {
            Some(FiLM::from_weights(
                condition_size,
                head1x1_params.out_channels,
                film_params.head1x1_post.shift,
                film_params.head1x1_post.groups,
                iter,
            )?)
        } else {
            None
        };

        Ok(Self {
            conv,
            input_mixin,
            layer1x1,
            head1x1,
            activation: activation.clone(),
            secondary_activation: secondary_activation.clone(),
            gating_mode,
            channels,
            bottleneck,
            head_output_size,
            conv_pre_film,
            conv_post_film,
            input_mixin_pre_film,
            input_mixin_post_film,
            activation_pre_film,
            activation_post_film,
            layer1x1_post_film,
            head1x1_post_film,
        })
    }
}

// ── Head1x1 params ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Head1x1Params {
    active: bool,
    out_channels: usize,
    groups: usize,
}

// ── FiLM params for all 8 positions in a layer ──────────────────────────────

#[derive(Debug, Clone)]
struct LayerFiLMParams {
    conv_pre: FiLMParams,
    conv_post: FiLMParams,
    input_mixin_pre: FiLMParams,
    input_mixin_post: FiLMParams,
    activation_pre: FiLMParams,
    activation_post: FiLMParams,
    layer1x1_post: FiLMParams,
    head1x1_post: FiLMParams,
}

// ── WaveNet LayerArray ──────────────────────────────────────────────────────

struct WaveNetLayerArray {
    rechannel: Conv1x1,
    layers: Vec<WaveNetLayer>,
    head_rechannel: Conv1x1,
    channels: usize,
    head_output_size: usize, // head1x1.out_channels if active, else bottleneck
}

impl WaveNetLayerArray {
    fn receptive_field(&self) -> usize {
        self.layers.iter().map(|l| l.conv.receptive_field()).sum()
    }
}

// ── Ring buffer for Conv1D history ──────────────────────────────────────────

struct RingBuffer {
    data: Vec<Vec<f32>>, // [buffer_size][channels]
    pos: usize,
    size: usize,
}

impl RingBuffer {
    fn new(size: usize, channels: usize) -> Self {
        Self {
            data: vec![vec![0.0; channels]; size],
            pos: 0,
            size,
        }
    }

    fn push(&mut self, frame: &[f32]) {
        self.data[self.pos].copy_from_slice(frame);
        self.pos = (self.pos + 1) % self.size;
    }

    /// Get frame at `offset` steps before the current write position.
    /// offset=0 means the most recently pushed frame.
    #[inline]
    fn get(&self, offset: usize) -> &[f32] {
        let idx = (self.pos + self.size - 1 - offset) % self.size;
        &self.data[idx]
    }

    fn reset(&mut self) {
        for frame in &mut self.data {
            frame.fill(0.0);
        }
        self.pos = 0;
    }
}

// ── Top-level WaveNet ───────────────────────────────────────────────────────

pub struct WaveNet {
    layer_arrays: Vec<WaveNetLayerArray>,
    head_scale: f32,
    prewarm_samples_count: usize,
    metadata: DspMetadata,

    // Optional condition DSP
    condition_dsp: Option<Box<dyn Dsp>>,

    // Per-layer-array ring buffers for the dilated conv history.
    ring_buffers: Vec<Vec<RingBuffer>>,

    // Scratch buffers (pre-allocated, reused each sample)
    condition_input_buf: Vec<f32>,  // [in_channels] (raw input)
    condition_output_buf: Vec<f32>, // [condition_size] (after condition_dsp)
    rechannel_buf: Vec<f32>,
    conv_out_buf: Vec<f32>,
    mixin_out_buf: Vec<f32>,
    z_buf: Vec<f32>,
    activated_buf: Vec<f32>,
    layer1x1_buf: Vec<f32>,
    head1x1_buf: Vec<f32>,
    residual_buf: Vec<f32>,
    head_accum_buf: Vec<f32>,
    head_rechannel_buf: Vec<f32>,
    prev_layer_output: Vec<f32>,
    prev_head_output: Vec<f32>,
    film_scratch_buf: Vec<f32>, // scratch for FiLM scale/shift computation
    film_input_buf: Vec<f32>,   // scratch for FiLM input modulation
    film_condition_buf: Vec<f32>, // scratch for condition modulation via FiLM
}

impl WaveNet {
    pub fn from_config(
        config: &serde_json::Value,
        weights: &[f32],
        metadata: DspMetadata,
    ) -> Result<Self, NamError> {
        Self::from_config_with_condition_dsp(config, weights, metadata, None)
    }

    pub fn from_config_with_condition_dsp(
        config: &serde_json::Value,
        weights: &[f32],
        metadata: DspMetadata,
        condition_dsp: Option<Box<dyn Dsp>>,
    ) -> Result<Self, NamError> {
        let layers_json = config["layers"]
            .as_array()
            .ok_or_else(|| NamError::MissingField("layers".into()))?;

        let mut iter = WeightIter::new(weights);
        let mut layer_arrays = Vec::new();
        let mut ring_buffers_all = Vec::new();
        let mut max_conv_out = 0usize;
        let mut max_channels = 0usize;
        let mut max_bottleneck = 0usize;
        let mut max_head_output_size = 0usize;
        let mut max_head_size = 0usize;
        let mut condition_size = 0usize;
        let mut max_film_scratch = 0usize;

        for la_json in layers_json {
            let input_size = la_json["input_size"].as_u64().unwrap() as usize;
            let cond_size = la_json["condition_size"].as_u64().unwrap() as usize;
            let head_size = la_json["head_size"].as_u64().unwrap() as usize;
            let channels = la_json["channels"].as_u64().unwrap() as usize;
            let bottleneck = la_json
                .get("bottleneck")
                .and_then(|v| v.as_u64())
                .unwrap_or(channels as u64) as usize;
            let kernel_size = la_json["kernel_size"].as_u64().unwrap() as usize;
            let dilations: Vec<usize> = la_json["dilations"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();

            let num_layers = dilations.len();

            // Parse activation configs (per-layer or single)
            let activation_configs: Vec<Activation> = {
                let act_val = &la_json["activation"];
                if let Some(arr) = act_val.as_array() {
                    arr.iter()
                        .map(Activation::from_json)
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    let act = Activation::from_json(act_val)
                        .unwrap_or_else(|_| Activation::from_name("Tanh").unwrap());
                    vec![act; num_layers]
                }
            };

            // Parse gating modes (per-layer or single or old bool)
            let (gating_modes, secondary_activations) =
                parse_gating_and_secondary(la_json, num_layers)?;

            let head_bias = la_json["head_bias"].as_bool().unwrap_or(false);

            // Groups
            let groups_input = la_json
                .get("groups_input")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize;
            let groups_input_mixin = la_json
                .get("groups_input_mixin")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize;

            // Layer1x1 config
            let (has_layer1x1, layer1x1_groups) = if let Some(l1x1) = la_json.get("layer1x1") {
                let active = l1x1.get("active").and_then(|v| v.as_bool()).unwrap_or(true);
                let groups = l1x1.get("groups").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
                (active, groups)
            } else {
                (true, 1) // default: active with groups=1
            };

            // Head1x1 config
            let head1x1_params = if let Some(h1x1) = la_json.get("head1x1") {
                let active = h1x1
                    .get("active")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let out_channels = h1x1
                    .get("out_channels")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(channels as u64) as usize;
                let groups = h1x1.get("groups").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
                Head1x1Params {
                    active,
                    out_channels,
                    groups,
                }
            } else {
                Head1x1Params {
                    active: false,
                    out_channels: channels,
                    groups: 1,
                }
            };

            // FiLM params
            let film_params = LayerFiLMParams {
                conv_pre: la_json
                    .get("conv_pre_film")
                    .map(FiLMParams::from_json)
                    .unwrap_or_else(FiLMParams::inactive),
                conv_post: la_json
                    .get("conv_post_film")
                    .map(FiLMParams::from_json)
                    .unwrap_or_else(FiLMParams::inactive),
                input_mixin_pre: la_json
                    .get("input_mixin_pre_film")
                    .map(FiLMParams::from_json)
                    .unwrap_or_else(FiLMParams::inactive),
                input_mixin_post: la_json
                    .get("input_mixin_post_film")
                    .map(FiLMParams::from_json)
                    .unwrap_or_else(FiLMParams::inactive),
                activation_pre: la_json
                    .get("activation_pre_film")
                    .map(FiLMParams::from_json)
                    .unwrap_or_else(FiLMParams::inactive),
                activation_post: la_json
                    .get("activation_post_film")
                    .map(FiLMParams::from_json)
                    .unwrap_or_else(FiLMParams::inactive),
                layer1x1_post: la_json
                    .get("layer1x1_post_film")
                    .map(FiLMParams::from_json)
                    .unwrap_or_else(FiLMParams::inactive),
                head1x1_post: la_json
                    .get("head1x1_post_film")
                    .map(FiLMParams::from_json)
                    .unwrap_or_else(FiLMParams::inactive),
            };

            let head_out_size = if head1x1_params.active {
                head1x1_params.out_channels
            } else {
                bottleneck
            };

            condition_size = cond_size;
            let conv_out = if gating_modes.iter().any(|g| *g != GatingMode::None) {
                2 * bottleneck
            } else {
                bottleneck
            };
            max_conv_out = max_conv_out.max(conv_out);
            max_channels = max_channels.max(channels);
            max_bottleneck = max_bottleneck.max(bottleneck);
            max_head_output_size = max_head_output_size.max(head_out_size);
            max_head_size = max_head_size.max(head_size);

            // Estimate max FiLM scratch needed
            let max_film_dim = [
                channels * 2,
                conv_out * 2,
                cond_size * 2,
                bottleneck * 2,
                head_out_size * 2,
            ]
            .into_iter()
            .max()
            .unwrap_or(0);
            max_film_scratch = max_film_scratch.max(max_film_dim);

            // Build layer array
            // Per C++ weight order: rechannel, then each layer, then head_rechannel
            let rechannel = Conv1x1::from_weights(input_size, channels, false, 1, &mut iter)?;

            let mut layers = Vec::new();
            let mut ring_bufs = Vec::new();

            for (layer_idx, &dil) in dilations.iter().enumerate() {
                let layer_gating = gating_modes[layer_idx];
                let layer_conv_out = if layer_gating != GatingMode::None {
                    2 * bottleneck
                } else {
                    bottleneck
                };
                max_conv_out = max_conv_out.max(layer_conv_out);

                let layer = WaveNetLayer::from_weights(
                    channels,
                    bottleneck,
                    cond_size,
                    kernel_size,
                    dil,
                    &activation_configs[layer_idx],
                    layer_gating,
                    groups_input,
                    groups_input_mixin,
                    has_layer1x1,
                    layer1x1_groups,
                    &head1x1_params,
                    &secondary_activations[layer_idx],
                    &film_params,
                    &mut iter,
                )?;

                let rf = layer.conv.receptive_field();
                ring_bufs.push(RingBuffer::new(rf + 1, channels));
                layers.push(layer);
            }

            let head_rechannel =
                Conv1x1::from_weights(head_out_size, head_size, head_bias, 1, &mut iter)?;

            layer_arrays.push(WaveNetLayerArray {
                rechannel,
                layers,
                head_rechannel,
                channels,
                head_output_size: head_out_size,
            });
            ring_buffers_all.push(ring_bufs);
        }

        let head_scale_from_weights = iter.take(1)?[0];
        let head_scale = head_scale_from_weights;

        iter.assert_exhausted()?;

        // Compute prewarm
        let condition_prewarm = condition_dsp
            .as_ref()
            .map(|d| d.prewarm_samples())
            .unwrap_or(1);
        let prewarm_samples_count = condition_prewarm
            + layer_arrays
                .iter()
                .map(|la| la.receptive_field())
                .sum::<usize>();

        let in_channels = 1; // NAM standard

        Ok(Self {
            layer_arrays,
            head_scale,
            prewarm_samples_count,
            metadata,
            condition_dsp,
            ring_buffers: ring_buffers_all,
            condition_input_buf: vec![0.0; in_channels.max(1)],
            condition_output_buf: vec![0.0; condition_size.max(1)],
            rechannel_buf: vec![0.0; max_channels],
            conv_out_buf: vec![0.0; max_conv_out],
            mixin_out_buf: vec![0.0; max_conv_out],
            z_buf: vec![0.0; max_conv_out],
            activated_buf: vec![0.0; max_bottleneck],
            layer1x1_buf: vec![0.0; max_channels],
            head1x1_buf: vec![0.0; max_head_output_size],
            residual_buf: vec![0.0; max_channels],
            head_accum_buf: vec![0.0; max_head_output_size],
            head_rechannel_buf: vec![0.0; max_head_size],
            prev_layer_output: vec![0.0; max_channels],
            prev_head_output: vec![0.0; max_head_size],
            film_scratch_buf: vec![0.0; max_film_scratch + 64],
            film_input_buf: vec![0.0; max_conv_out.max(max_channels).max(condition_size) + 64],
            film_condition_buf: vec![0.0; condition_size.max(1) + 64],
        })
    }

    fn process_sample(&mut self, input_sample: f32) -> f32 {
        // Condition input
        self.condition_input_buf[0] = input_sample;

        // Process condition DSP if present
        if let Some(ref mut cdsp) = self.condition_dsp {
            let cond_len = self.condition_output_buf.len();
            cdsp.process_sample_multi_channel(
                input_sample as Sample,
                &mut self.condition_output_buf[..cond_len],
            );
        } else {
            // No condition DSP: condition = raw input
            let cond_len = self.condition_output_buf.len();
            self.condition_output_buf[..cond_len.min(self.condition_input_buf.len())]
                .copy_from_slice(
                    &self.condition_input_buf[..cond_len.min(self.condition_input_buf.len())],
                );
        }

        let mut _final_head_size = 0usize;

        for (arr_idx, la) in self.layer_arrays.iter().enumerate() {
            let channels = la.channels;
            let head_output_size = la.head_output_size;
            let head_size = la.head_rechannel.out_channels;
            _final_head_size = head_size;

            // Rechannel: project input to channels
            // C++: first array uses condition_input; subsequent use prev layer output
            let rechannel_input = if arr_idx == 0 {
                &self.condition_input_buf[..]
            } else {
                &self.prev_layer_output[..self.layer_arrays[arr_idx - 1].channels]
            };

            la.rechannel
                .forward(rechannel_input, &mut self.rechannel_buf[..channels]);

            // Zero head accumulator
            self.head_accum_buf[..head_output_size].fill(0.0);

            // If subsequent array, copy previous head output
            if arr_idx > 0 {
                let prev_head_size = self.layer_arrays[arr_idx - 1].head_rechannel.out_channels;
                let copy_size = head_output_size.min(prev_head_size);
                self.head_accum_buf[..copy_size]
                    .copy_from_slice(&self.prev_head_output[..copy_size]);
            }

            // Process each layer
            for (layer_idx, layer) in la.layers.iter().enumerate() {
                let conv = &layer.conv;
                let dilation = conv.dilation;
                let kernel_size = conv.kernel_size;
                let conv_out_ch = conv.out_channels;
                let bottleneck = layer.bottleneck;
                let gating_mode = layer.gating_mode;

                // Push current input into ring buffer
                if layer_idx == 0 {
                    self.ring_buffers[arr_idx][layer_idx].push(&self.rechannel_buf[..channels]);
                } else {
                    self.ring_buffers[arr_idx][layer_idx].push(&self.residual_buf[..channels]);
                }

                // ── Step 1: Dilated convolution ──
                // Optional conv_pre_film: modulate input before conv
                if let Some(ref film) = layer.conv_pre_film {
                    // We need to apply FiLM to each input frame read from ring buffer.
                    // For the per-sample processing, we modulate the current frame that was just pushed.
                    // Actually in the C++ block processing, conv_pre_film modulates the whole input
                    // before the conv reads it. In per-sample mode, we need to modulate the ring buffer
                    // content. Since the conv reads multiple taps from the ring buffer, we'd need
                    // to modulate each tap separately, which is expensive and different from the
                    // block-processing C++ approach.
                    //
                    // A simpler approach: for per-sample mode, we can modulate the most recently
                    // pushed frame in the ring buffer (the only one that changes between samples).
                    // This is correct for causal processing because each frame only gets modulated once
                    // when it enters the buffer.
                    let rb = &self.ring_buffers[arr_idx][layer_idx];
                    let frame = rb.get(0);
                    self.film_input_buf[..channels].copy_from_slice(&frame[..channels]);
                    film.forward(
                        &self.film_input_buf[..channels],
                        &self.condition_output_buf,
                        &mut self.film_condition_buf[..channels],
                        &mut self.film_scratch_buf,
                    );
                    // Overwrite the ring buffer frame
                    // Safety: we just pushed this frame, so we can overwrite it
                    let rb = &mut self.ring_buffers[arr_idx][layer_idx];
                    let pos = (rb.pos + rb.size - 1) % rb.size;
                    rb.data[pos][..channels].copy_from_slice(&self.film_condition_buf[..channels]);
                }

                // Dilated convolution: read from ring buffer
                self.conv_out_buf[..conv_out_ch].copy_from_slice(&conv.bias[..conv_out_ch]);

                for k in 0..kernel_size {
                    let offset = dilation * (kernel_size - 1 - k);
                    let frame = self.ring_buffers[arr_idx][layer_idx].get(offset);

                    let w = &conv.weights[k];
                    let in_ch = conv.in_channels;
                    for i in 0..conv_out_ch {
                        let mut sum = 0.0f32;
                        let row_start = i * in_ch;
                        for j in 0..in_ch {
                            sum += w[row_start + j] * frame[j];
                        }
                        self.conv_out_buf[i] += sum;
                    }
                }

                // Optional conv_post_film
                if let Some(ref film) = layer.conv_post_film {
                    self.film_input_buf[..conv_out_ch]
                        .copy_from_slice(&self.conv_out_buf[..conv_out_ch]);
                    film.forward(
                        &self.film_input_buf[..conv_out_ch],
                        &self.condition_output_buf,
                        &mut self.conv_out_buf[..conv_out_ch],
                        &mut self.film_scratch_buf,
                    );
                }

                // ── Step 2: Input mixin ──
                // Optional input_mixin_pre_film
                let mixin_condition = if let Some(ref film) = layer.input_mixin_pre_film {
                    let cond_len = self.condition_output_buf.len();
                    self.film_condition_buf[..cond_len]
                        .copy_from_slice(&self.condition_output_buf[..cond_len]);
                    film.forward(
                        &self.film_condition_buf[..cond_len],
                        &self.condition_output_buf,
                        &mut self.film_input_buf[..cond_len],
                        &mut self.film_scratch_buf,
                    );
                    &self.film_input_buf[..cond_len]
                } else {
                    &self.condition_output_buf[..]
                };

                layer
                    .input_mixin
                    .forward(mixin_condition, &mut self.mixin_out_buf[..conv_out_ch]);

                // Optional input_mixin_post_film
                if let Some(ref film) = layer.input_mixin_post_film {
                    self.film_input_buf[..conv_out_ch]
                        .copy_from_slice(&self.mixin_out_buf[..conv_out_ch]);
                    film.forward(
                        &self.film_input_buf[..conv_out_ch],
                        &self.condition_output_buf,
                        &mut self.mixin_out_buf[..conv_out_ch],
                        &mut self.film_scratch_buf,
                    );
                }

                // z = conv_out + mixin_out
                for i in 0..conv_out_ch {
                    self.z_buf[i] = self.conv_out_buf[i] + self.mixin_out_buf[i];
                }

                // Optional activation_pre_film
                if let Some(ref film) = layer.activation_pre_film {
                    self.film_input_buf[..conv_out_ch].copy_from_slice(&self.z_buf[..conv_out_ch]);
                    film.forward(
                        &self.film_input_buf[..conv_out_ch],
                        &self.condition_output_buf,
                        &mut self.z_buf[..conv_out_ch],
                        &mut self.film_scratch_buf,
                    );
                }

                // ── Step 3: Activation (with optional gating/blending) ──
                // Use apply_scalar_channel for PReLU per-channel support
                match gating_mode {
                    GatingMode::None => {
                        for i in 0..bottleneck {
                            self.activated_buf[i] =
                                layer.activation.apply_scalar_channel(self.z_buf[i], i);
                        }
                    }
                    GatingMode::Gated => {
                        for i in 0..bottleneck {
                            let primary = layer.activation.apply_scalar_channel(self.z_buf[i], i);
                            let gate = layer
                                .secondary_activation
                                .apply_scalar_channel(self.z_buf[i + bottleneck], i);
                            self.activated_buf[i] = primary * gate;
                        }
                    }
                    GatingMode::Blended => {
                        for i in 0..bottleneck {
                            let pre_activation = self.z_buf[i];
                            let activated =
                                layer.activation.apply_scalar_channel(pre_activation, i);
                            let alpha = layer
                                .secondary_activation
                                .apply_scalar_channel(self.z_buf[i + bottleneck], i);
                            self.activated_buf[i] =
                                alpha * activated + (1.0 - alpha) * pre_activation;
                        }
                    }
                }

                // Optional activation_post_film
                if let Some(ref film) = layer.activation_post_film {
                    self.film_input_buf[..bottleneck]
                        .copy_from_slice(&self.activated_buf[..bottleneck]);
                    film.forward(
                        &self.film_input_buf[..bottleneck],
                        &self.condition_output_buf,
                        &mut self.activated_buf[..bottleneck],
                        &mut self.film_scratch_buf,
                    );
                }

                // ── Step 4: layer1x1 for residual ──
                let rb_input = self.ring_buffers[arr_idx][layer_idx].get(0);
                if let Some(ref l1x1) = layer.layer1x1 {
                    l1x1.forward(
                        &self.activated_buf[..bottleneck],
                        &mut self.layer1x1_buf[..channels],
                    );

                    // Optional layer1x1_post_film — C++ only applies this in BLENDED mode
                    if gating_mode == GatingMode::Blended {
                        if let Some(ref film) = layer.layer1x1_post_film {
                            self.film_input_buf[..channels]
                                .copy_from_slice(&self.layer1x1_buf[..channels]);
                            film.forward(
                                &self.film_input_buf[..channels],
                                &self.condition_output_buf,
                                &mut self.layer1x1_buf[..channels],
                                &mut self.film_scratch_buf,
                            );
                        }
                    }

                    for (i, &rb_val) in rb_input[..channels].iter().enumerate() {
                        self.residual_buf[i] = rb_val + self.layer1x1_buf[i];
                    }
                } else {
                    self.residual_buf[..channels].copy_from_slice(&rb_input[..channels]);
                }

                // ── Step 5: head output (skip connection) ──
                if let Some(ref h1x1) = layer.head1x1 {
                    let h_out = h1x1.out_channels;
                    h1x1.forward(
                        &self.activated_buf[..bottleneck],
                        &mut self.head1x1_buf[..h_out],
                    );

                    // Optional head1x1_post_film
                    if let Some(ref film) = layer.head1x1_post_film {
                        self.film_input_buf[..h_out].copy_from_slice(&self.head1x1_buf[..h_out]);
                        film.forward(
                            &self.film_input_buf[..h_out],
                            &self.condition_output_buf,
                            &mut self.head1x1_buf[..h_out],
                            &mut self.film_scratch_buf,
                        );
                    }

                    for i in 0..h_out.min(head_output_size) {
                        self.head_accum_buf[i] += self.head1x1_buf[i];
                    }
                } else {
                    // No head1x1: accumulate activated output (or z for GatingMode::None)
                    if gating_mode == GatingMode::None && layer.activation_post_film.is_none() {
                        // For no gating, no post-film: head output is z (same as activated)
                        for i in 0..bottleneck.min(head_output_size) {
                            self.head_accum_buf[i] += self.activated_buf[i];
                        }
                    } else {
                        for i in 0..bottleneck.min(head_output_size) {
                            self.head_accum_buf[i] += self.activated_buf[i];
                        }
                    }
                }
            }

            // Store layer output for next array's input
            self.prev_layer_output[..channels].copy_from_slice(&self.residual_buf[..channels]);

            // Head rechannel: head_accum -> head_size output
            la.head_rechannel.forward(
                &self.head_accum_buf[..head_output_size],
                &mut self.head_rechannel_buf[..head_size],
            );

            // Store for next array
            self.prev_head_output[..head_size]
                .copy_from_slice(&self.head_rechannel_buf[..head_size]);
        }

        // Output = head_scale * last head_rechannel output
        self.head_scale * self.prev_head_output[0]
    }
}

impl Dsp for WaveNet {
    fn process(&mut self, input: &[Sample], output: &mut [Sample]) {
        for (i, &sample) in input.iter().enumerate() {
            output[i] = self.process_sample(sample as f32) as Sample;
        }
    }

    fn reset(&mut self, sample_rate: f64, max_buffer_size: usize) {
        for arr_bufs in &mut self.ring_buffers {
            for rb in arr_bufs.iter_mut() {
                rb.reset();
            }
        }
        if let Some(ref mut cdsp) = self.condition_dsp {
            cdsp.reset(sample_rate, max_buffer_size);
        }
    }

    fn num_output_channels(&self) -> usize {
        self.layer_arrays
            .last()
            .map(|la| la.head_rechannel.out_channels)
            .unwrap_or(1)
    }

    fn process_sample_multi_channel(&mut self, input_sample: Sample, out: &mut [f32]) {
        self.process_sample(input_sample as f32);
        let head_size = self.num_output_channels();
        for (o, &h) in out
            .iter_mut()
            .zip(self.prev_head_output[..head_size].iter())
        {
            *o = self.head_scale * h;
        }
    }

    fn prewarm_samples(&self) -> usize {
        self.prewarm_samples_count
    }

    fn metadata(&self) -> &DspMetadata {
        &self.metadata
    }
}

// ── Helpers for parsing gating mode and secondary activations ───────────────

fn parse_gating_and_secondary(
    la_json: &serde_json::Value,
    num_layers: usize,
) -> Result<(Vec<GatingMode>, Vec<Activation>), NamError> {
    let parse_gating_str = |s: &str| -> Result<GatingMode, NamError> {
        match s {
            "gated" => Ok(GatingMode::Gated),
            "blended" => Ok(GatingMode::Blended),
            "none" => Ok(GatingMode::None),
            other => Err(NamError::InvalidConfig(format!(
                "Invalid gating_mode: {}",
                other
            ))),
        }
    };

    let default_secondary = || Activation::Sigmoid;

    if let Some(gm_val) = la_json.get("gating_mode") {
        if let Some(arr) = gm_val.as_array() {
            // Per-layer gating modes
            let mut modes = Vec::new();
            let mut sec_acts = Vec::new();

            let sec_val = la_json.get("secondary_activation");

            for (idx, gm_json) in arr.iter().enumerate() {
                let mode_str = gm_json.as_str().ok_or_else(|| {
                    NamError::InvalidConfig("gating_mode element not string".into())
                })?;
                let mode = parse_gating_str(mode_str)?;
                modes.push(mode);

                if mode != GatingMode::None {
                    if let Some(sv) = sec_val {
                        if let Some(sa_arr) = sv.as_array() {
                            sec_acts.push(Activation::from_json(&sa_arr[idx])?);
                        } else {
                            sec_acts.push(Activation::from_json(sv)?);
                        }
                    } else {
                        sec_acts.push(default_secondary());
                    }
                } else {
                    sec_acts.push(default_secondary()); // placeholder
                }
            }
            Ok((modes, sec_acts))
        } else if let Some(mode_str) = gm_val.as_str() {
            let mode = parse_gating_str(mode_str)?;
            let sec_act = if mode != GatingMode::None {
                if let Some(sv) = la_json.get("secondary_activation") {
                    Activation::from_json(sv)?
                } else {
                    default_secondary()
                }
            } else {
                default_secondary()
            };
            Ok((vec![mode; num_layers], vec![sec_act; num_layers]))
        } else {
            Ok((
                vec![GatingMode::None; num_layers],
                vec![default_secondary(); num_layers],
            ))
        }
    } else if let Some(gated) = la_json.get("gated").and_then(|v| v.as_bool()) {
        // Backward compatibility: bool "gated"
        let mode = if gated {
            GatingMode::Gated
        } else {
            GatingMode::None
        };
        let sec_act = default_secondary();
        Ok((vec![mode; num_layers], vec![sec_act; num_layers]))
    } else {
        Ok((
            vec![GatingMode::None; num_layers],
            vec![default_secondary(); num_layers],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn load_wavenet(filename: &str) -> Option<WaveNet> {
        let path = Path::new("test_fixtures/models").join(filename);
        if !path.exists() {
            eprintln!("Skipping test: {:?} not found", path);
            return None;
        }
        let content = std::fs::read_to_string(&path).unwrap();
        let root: serde_json::Value = serde_json::from_str(&content).unwrap();
        let weights: Vec<f32> = root["weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let metadata = DspMetadata::default();
        let config = &root["config"];

        // Check for condition_dsp
        let condition_dsp = if let Some(cd) = config.get("condition_dsp") {
            if !cd.is_null() {
                match crate::get_dsp::get_dsp_from_value(cd) {
                    Ok(dsp) => Some(dsp),
                    Err(e) => {
                        eprintln!("Failed to load condition_dsp for {}: {}", filename, e);
                        return None;
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        match WaveNet::from_config_with_condition_dsp(config, &weights, metadata, condition_dsp) {
            Ok(wn) => Some(wn),
            Err(e) => {
                eprintln!("Failed to load {}: {}", filename, e);
                None
            }
        }
    }

    #[test]
    fn test_wavenet_loads() {
        let model = load_wavenet("wavenet.nam");
        assert!(model.is_some(), "wavenet.nam should load");
    }

    #[test]
    fn test_wavenet_processes() {
        let mut model = match load_wavenet("wavenet.nam") {
            Some(m) => m,
            None => return,
        };

        let input = vec![0.0 as Sample; 128];
        let mut output = vec![0.0 as Sample; 128];
        model.process(&input, &mut output);

        let mut impulse = vec![0.0 as Sample; 128];
        impulse[0] = 1.0 as Sample;
        let mut out2 = vec![0.0 as Sample; 128];
        model.process(&impulse, &mut out2);

        let has_nonzero = out2.iter().any(|&x| x != 0.0);
        assert!(has_nonzero, "WaveNet output was all zeros after impulse");
    }

    #[test]
    fn test_all_example_models_load() {
        let models = [
            "wavenet.nam",
            "wavenet_a1_standard.nam",
            "my_model.nam",
            "wavenet_a2_max.nam",
            "wavenet_condition_dsp.nam",
        ];
        for name in &models {
            let path = Path::new("test_fixtures/models").join(name);
            if !path.exists() {
                eprintln!("Skipping: {:?}", path);
                continue;
            }
            let content = std::fs::read_to_string(&path).unwrap();
            let root: serde_json::Value = serde_json::from_str(&content).unwrap();
            if root["architecture"].as_str() != Some("WaveNet") {
                continue;
            }
            let model = load_wavenet(name);
            assert!(model.is_some(), "Failed to load {}", name);
        }
    }

    #[test]
    fn test_all_example_models_process() {
        let models = [
            "wavenet.nam",
            "wavenet_a1_standard.nam",
            "my_model.nam",
            "wavenet_a2_max.nam",
            "wavenet_condition_dsp.nam",
        ];
        for name in &models {
            let path = Path::new("test_fixtures/models").join(name);
            if !path.exists() {
                eprintln!("Skipping: {:?}", path);
                continue;
            }
            let content = std::fs::read_to_string(&path).unwrap();
            let root: serde_json::Value = serde_json::from_str(&content).unwrap();
            if root["architecture"].as_str() != Some("WaveNet") {
                continue;
            }
            let mut model = match load_wavenet(name) {
                Some(m) => m,
                None => {
                    panic!("Failed to load {}", name);
                }
            };

            // Process some audio
            let input = vec![0.1 as Sample; 64];
            let mut output = vec![0.0 as Sample; 64];
            model.process(&input, &mut output);

            assert!(
                output.iter().all(|&x| (x as f64).is_finite()),
                "Non-finite output from {}",
                name
            );
        }
    }

    #[test]
    fn test_state_persists_across_calls() {
        let path = Path::new("test_fixtures/models/wavenet.nam");
        if !path.exists() {
            return;
        }

        let mut model_split = crate::get_dsp(path).unwrap();
        let input1 = vec![0.5 as Sample; 16];
        let mut out1 = vec![0.0 as Sample; 16];
        model_split.process(&input1, &mut out1);
        let input2 = vec![0.0 as Sample; 16];
        let mut out2a = vec![0.0 as Sample; 16];
        model_split.process(&input2, &mut out2a);

        let mut model_full = crate::get_dsp(path).unwrap();
        let mut full_input = vec![0.5 as Sample; 16];
        full_input.extend(vec![0.0 as Sample; 16]);
        let mut full_output = vec![0.0 as Sample; 32];
        model_full.process(&full_input, &mut full_output);

        for i in 0..16 {
            assert!(
                (out2a[i] - full_output[16 + i]).abs() < 1e-5,
                "State mismatch at {}: split={}, full={}",
                i,
                out2a[i],
                full_output[16 + i]
            );
        }
    }

    #[test]
    fn test_single_sample_vs_block() {
        let path = Path::new("test_fixtures/models/wavenet.nam");
        if !path.exists() {
            return;
        }

        let mut model_single = crate::get_dsp(path).unwrap();
        let mut outputs_single = Vec::new();
        for i in 0..32 {
            let input = vec![if i == 0 { 1.0 } else { 0.0 } as Sample];
            let mut output = vec![0.0 as Sample; 1];
            model_single.process(&input, &mut output);
            outputs_single.push(output[0]);
        }

        let mut model_block = crate::get_dsp(path).unwrap();
        let mut block_input = vec![0.0 as Sample; 32];
        block_input[0] = 1.0 as Sample;
        let mut outputs_block = vec![0.0 as Sample; 32];
        model_block.process(&block_input, &mut outputs_block);

        for i in 0..32 {
            assert!(
                (outputs_single[i] - outputs_block[i]).abs() < 1e-5,
                "Sample {} mismatch: single={}, block={}",
                i,
                outputs_single[i],
                outputs_block[i]
            );
        }
    }

    #[test]
    fn test_prewarm_changes_output() {
        let path = Path::new("test_fixtures/models/wavenet.nam");
        if !path.exists() {
            return;
        }

        let input = vec![0.1 as Sample; 16];

        let mut model_no_pw = crate::get_dsp(path).unwrap();
        let mut out_no_pw = vec![0.0 as Sample; 16];
        model_no_pw.process(&input, &mut out_no_pw);

        let mut model_pw = crate::get_dsp(path).unwrap();
        model_pw.prewarm();
        let mut out_pw = vec![0.0 as Sample; 16];
        model_pw.process(&input, &mut out_pw);

        let any_different = out_no_pw
            .iter()
            .zip(out_pw.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-10);
        assert!(
            any_different,
            "Prewarm should change initial output behavior"
        );
    }

    #[test]
    fn test_prewarm_samples_positive() {
        let path = Path::new("test_fixtures/models/wavenet.nam");
        if !path.exists() {
            return;
        }
        let model = crate::get_dsp(path).unwrap();
        assert!(model.prewarm_samples() > 0);
    }

    #[test]
    fn test_reset_clears_state() {
        let path = Path::new("test_fixtures/models/wavenet.nam");
        if !path.exists() {
            return;
        }

        let mut model = crate::get_dsp(path).unwrap();
        let input = vec![1.0 as Sample; 64];
        let mut output = vec![0.0 as Sample; 64];
        model.process(&input, &mut output);

        model.reset(48000.0, 4096);

        let mut model_fresh = crate::get_dsp(path).unwrap();
        let mut out_reset = vec![0.0 as Sample; 64];
        let mut out_fresh = vec![0.0 as Sample; 64];
        model.process(&input, &mut out_reset);
        model_fresh.process(&input, &mut out_fresh);

        for i in 0..64 {
            assert!(
                (out_reset[i] - out_fresh[i]).abs() < 1e-5,
                "Reset mismatch at {}: reset={}, fresh={}",
                i,
                out_reset[i],
                out_fresh[i]
            );
        }
    }

    #[test]
    fn test_large_standard_model() {
        let path = Path::new("test_fixtures/models/wavenet_a1_standard.nam");
        if !path.exists() {
            return;
        }
        let mut model = crate::get_dsp(path).unwrap();
        model.prewarm();

        let input = vec![0.1 as Sample; 256];
        let mut output = vec![0.0 as Sample; 256];
        model.process(&input, &mut output);

        assert!(output.iter().all(|&x| (x as f64).is_finite()));
        assert!(output.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_process_empty_buffer() {
        let path = Path::new("test_fixtures/models/wavenet.nam");
        if !path.exists() {
            return;
        }
        let mut model = crate::get_dsp(path).unwrap();
        let input: Vec<Sample> = vec![];
        let mut output: Vec<Sample> = vec![];
        model.process(&input, &mut output);
    }

    #[test]
    fn test_receptive_field_calculation() {
        let path = Path::new("test_fixtures/models/wavenet.nam");
        if !path.exists() {
            return;
        }
        let model = crate::get_dsp(path).unwrap();
        assert_eq!(model.prewarm_samples(), 23);
    }

    #[test]
    fn test_a1_standard_receptive_field() {
        let path = Path::new("test_fixtures/models/wavenet_a1_standard.nam");
        if !path.exists() {
            return;
        }
        let model = crate::get_dsp(path).unwrap();
        assert_eq!(model.prewarm_samples(), 4093);
    }

    #[test]
    fn test_zero_input() {
        let path = Path::new("test_fixtures/models/wavenet.nam");
        if !path.exists() {
            return;
        }
        let mut model = crate::get_dsp(path).unwrap();

        let input = vec![0.0 as Sample; 32];
        let mut output = vec![0.0 as Sample; 32];
        model.process(&input, &mut output);

        assert!(output.iter().all(|&x| (x as f64).is_finite()));
    }

    #[test]
    fn test_different_buffer_sizes() {
        let path = Path::new("test_fixtures/models/wavenet.nam");
        if !path.exists() {
            return;
        }
        let mut model = crate::get_dsp(path).unwrap();

        for &size in &[1, 7, 16, 64, 128, 256] {
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

    #[test]
    fn test_multiple_consecutive_calls() {
        let path = Path::new("test_fixtures/models/wavenet.nam");
        if !path.exists() {
            return;
        }
        let mut model = crate::get_dsp(path).unwrap();

        for call in 0..10 {
            let input = vec![0.1 as Sample; 8];
            let mut output = vec![0.0 as Sample; 8];
            model.process(&input, &mut output);
            assert!(
                output.iter().all(|&x| (x as f64).is_finite()),
                "Non-finite at call {}",
                call
            );
        }
    }

    #[test]
    fn test_wavenet_a2_max_loads_and_processes() {
        let mut model = match load_wavenet("wavenet_a2_max.nam") {
            Some(m) => m,
            None => return,
        };

        let input = vec![0.1 as Sample; 64];
        let mut output = vec![0.0 as Sample; 64];
        model.process(&input, &mut output);
        assert!(output.iter().all(|&x| (x as f64).is_finite()));
    }

    #[test]
    fn test_wavenet_condition_dsp_loads_and_processes() {
        let mut model = match load_wavenet("wavenet_condition_dsp.nam") {
            Some(m) => m,
            None => return,
        };

        let input = vec![0.1 as Sample; 64];
        let mut output = vec![0.0 as Sample; 64];
        model.process(&input, &mut output);
        assert!(output.iter().all(|&x| (x as f64).is_finite()));
    }
}
