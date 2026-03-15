// Allow index-based loops in GEMM hot paths where explicit indexing matches C++ Eigen order
#![allow(clippy::needless_range_loop)]

use crate::activations::Activation;
use crate::dsp::{Dsp, DspMetadata, Sample};
use crate::error::NamError;
use crate::util::WeightIter;

/// Use matrixmultiply::sgemm for matrices at or above this size (out_ch * in_ch).
/// Below this threshold, use the hand-written dot-product loop which preserves
/// exact floating-point order for bit-identical results on small models.
const SGEMM_MIN_SIZE: usize = 32;

/// Column-major GEMM: C = alpha * A @ B + beta * C
/// A is (m x k), B is (k x n), C is (m x n), all column-major.
#[inline]
unsafe fn sgemm_colmajor(
    m: usize, k: usize, n: usize,
    alpha: f32, a: *const f32,
    b: *const f32, b_col_stride: isize,
    beta: f32, c: *mut f32,
) {
    matrixmultiply::sgemm(
        m, k, n, alpha,
        a, 1, m as isize,           // A: col-major
        b, 1, b_col_stride,         // B: col-major with given stride
        beta,
        c, 1, m as isize,           // C: col-major
    );
}

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

// ── Column-major 2D matrix helper ───────────────────────────────────────────
// Storage: flat Vec<f32> in column-major order (like Eigen).
// Element at (row, col) is at index: col * num_rows + row
// A column slice for column c starts at: c * num_rows, length = num_rows

/// A 2D matrix stored in column-major order (matching Eigen's default layout).
/// rows = channels, cols = frames.
struct ColMajorMatrix {
    data: Vec<f32>,
    rows: usize,
    // cols is implicit: data.len() / rows (or max_cols for pre-allocated)
    #[allow(dead_code)]
    max_cols: usize,
}

impl ColMajorMatrix {
    fn new(rows: usize, max_cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * max_cols],
            rows,
            max_cols,
        }
    }

    fn zero_cols(&mut self, num_cols: usize) {
        let len = self.rows * num_cols;
        self.data[..len].fill(0.0);
    }

    fn resize(&mut self, rows: usize, max_cols: usize) {
        self.rows = rows;
        self.max_cols = max_cols;
        let needed = rows * max_cols;
        if self.data.len() < needed {
            self.data.resize(needed, 0.0);
        }
        self.data[..needed].fill(0.0);
    }
}

// ── Ring Buffer 2D (column-major, matching C++ RingBuffer) ──────────────────

struct RingBuffer2D {
    storage: Vec<f32>, // column-major: [channels * storage_cols]
    channels: usize,
    storage_cols: usize,
    write_pos: usize, // current write position (column index)
    max_lookback: usize,
    max_buffer_size: usize,
}

impl RingBuffer2D {
    fn new() -> Self {
        Self {
            storage: Vec::new(),
            channels: 0,
            storage_cols: 0,
            write_pos: 0,
            max_lookback: 0,
            max_buffer_size: 0,
        }
    }

    fn set_max_lookback(&mut self, max_lookback: usize) {
        self.max_lookback = max_lookback;
    }

    fn reset(&mut self, channels: usize, max_buffer_size: usize) {
        self.channels = channels;
        self.max_buffer_size = max_buffer_size;
        // Storage size: 2 * max_lookback + max_buffer_size (matching C++)
        self.storage_cols = 2 * self.max_lookback + max_buffer_size;
        self.storage = vec![0.0; channels * self.storage_cols];
        self.write_pos = self.max_lookback;
    }

    /// Write num_frames columns from src (column-major, channels rows) at write_pos
    fn write(&mut self, src: &ColMajorMatrix, num_frames: usize) {
        // Check if we need rewind
        if self.write_pos + num_frames > self.storage_cols {
            self.rewind();
        }

        let ch = self.channels;
        for f in 0..num_frames {
            let src_off = f * src.rows;
            let dst_off = (self.write_pos + f) * ch;
            self.storage[dst_off..dst_off + ch].copy_from_slice(&src.data[src_off..src_off + ch]);
        }
    }

    /// Read num_frames columns starting at (write_pos - lookback).
    /// Returns a pointer to the start; data is column-major with stride = channels.
    #[inline]
    fn read_ptr(&self, _num_frames: usize, lookback: usize) -> &[f32] {
        let read_pos = self.write_pos - lookback;
        let start = read_pos * self.channels;
        &self.storage[start..]
    }

    fn advance(&mut self, num_frames: usize) {
        self.write_pos += num_frames;
    }

    fn rewind(&mut self) {
        if self.max_lookback == 0 {
            self.write_pos = 0;
            return;
        }
        let ch = self.channels;
        let copy_start = self.write_pos - self.max_lookback;
        // Copy max_lookback columns from copy_start to position 0
        for c in 0..self.max_lookback {
            let src_off = (copy_start + c) * ch;
            let dst_off = c * ch;
            // Can't use copy_from_slice because regions may overlap, but since
            // copy_start >= max_lookback (by C++ invariant), they don't overlap.
            for i in 0..ch {
                self.storage[dst_off + i] = self.storage[src_off + i];
            }
        }
        self.write_pos = self.max_lookback;
    }

    #[allow(dead_code)]
    fn zero(&mut self) {
        self.storage.fill(0.0);
        self.write_pos = self.max_lookback;
    }
}

// ── Conv1x1 (with groups support) ───────────────────────────────────────────

/// 1x1 convolution (pointwise linear layer) with optional grouped convolution.
/// Weight stored in column-major order matching Eigen: weight[j * out_channels + i]
/// means weight(i, j) = W[row=i, col=j].
struct Conv1x1 {
    /// Weights stored column-major: [out_channels * in_channels]
    /// weight[j * out_ch + i] = W(i, j)
    weight_colmajor: Vec<f32>,
    bias: Option<Vec<f32>>,
    out_channels: usize,
    in_channels: usize,
    #[allow(dead_code)]
    groups: usize,
    // Pre-allocated output buffer for block processing
    output_buf: ColMajorMatrix,
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

        // Build weight in column-major order matching Eigen layout
        // Eigen column-major: W(i,j) at index j * out_channels + i
        let mut weight_colmajor = vec![0.0f32; out_channels * in_channels];

        // C++ weight order: for group, for out_per_group, for in_per_group
        for g in 0..groups {
            for i in 0..out_per_group {
                for j in 0..in_per_group {
                    let val = iter.take(1)?[0];
                    let row = g * out_per_group + i;
                    let col = g * in_per_group + j;
                    // column-major: index = col * out_channels + row
                    weight_colmajor[col * out_channels + row] = val;
                }
            }
        }

        let bias = if has_bias {
            Some(iter.take(out_channels)?.to_vec())
        } else {
            None
        };

        Ok(Self {
            weight_colmajor,
            bias,
            out_channels,
            in_channels,
            groups,
            output_buf: ColMajorMatrix::new(out_channels, 1),
        })
    }

    fn set_max_buffer_size(&mut self, max_buffer_size: usize) {
        self.output_buf.resize(self.out_channels, max_buffer_size);
    }

    /// Block processing: output = W @ input (+ bias), column-major.
    /// Input: (in_channels x num_frames), Output: written to self.output_buf
    /// Matches Eigen: output.noalias() = weight * input; output.colwise() += bias;
    fn process_block(&mut self, input: &ColMajorMatrix, num_frames: usize) {
        let out_ch = self.out_channels;
        let in_ch = self.in_channels;

        if out_ch * in_ch >= SGEMM_MIN_SIZE {
            // Large matrix: SIMD-optimized sgemm
            if let Some(ref b) = self.bias {
                for f in 0..num_frames {
                    let s = f * out_ch;
                    self.output_buf.data[s..s + out_ch].copy_from_slice(&b[..out_ch]);
                }
            } else {
                self.output_buf.data[..out_ch * num_frames].fill(0.0);
            }
            unsafe {
                sgemm_colmajor(
                    out_ch, in_ch, num_frames, 1.0,
                    self.weight_colmajor.as_ptr(),
                    input.data.as_ptr(), input.rows as isize,
                    1.0, self.output_buf.data.as_mut_ptr(),
                );
            }
        } else {
            // Small matrix: exact dot-product loop (preserves bit-identical results)
            let w = &self.weight_colmajor;
            for f in 0..num_frames {
                let in_col_start = f * input.rows;
                let out_col_start = f * out_ch;
                for o in 0..out_ch {
                    let mut sum = 0.0f32;
                    for i in 0..in_ch {
                        sum += w[i * out_ch + o] * input.data[in_col_start + i];
                    }
                    self.output_buf.data[out_col_start + o] = sum;
                }
            }
            if let Some(ref b) = self.bias {
                for f in 0..num_frames {
                    let col_start = f * out_ch;
                    for o in 0..out_ch {
                        self.output_buf.data[col_start + o] += b[o];
                    }
                }
            }
        }
    }

    /// Block processing with a sub-matrix of input (topRows).
    /// input_stride is the actual row count of the input matrix (for reading columns).
    /// We read only the first in_channels rows from each column.
    fn process_block_with_stride(
        &mut self,
        input_data: &[f32],
        input_stride: usize,
        num_frames: usize,
    ) {
        let out_ch = self.out_channels;
        let in_ch = self.in_channels;

        if out_ch * in_ch >= SGEMM_MIN_SIZE {
            if let Some(ref b) = self.bias {
                for f in 0..num_frames {
                    let s = f * out_ch;
                    self.output_buf.data[s..s + out_ch].copy_from_slice(&b[..out_ch]);
                }
            } else {
                self.output_buf.data[..out_ch * num_frames].fill(0.0);
            }
            unsafe {
                sgemm_colmajor(
                    out_ch, in_ch, num_frames, 1.0,
                    self.weight_colmajor.as_ptr(),
                    input_data.as_ptr(), input_stride as isize,
                    1.0, self.output_buf.data.as_mut_ptr(),
                );
            }
        } else {
            let w = &self.weight_colmajor;
            for f in 0..num_frames {
                let in_col_start = f * input_stride;
                let out_col_start = f * out_ch;
                for o in 0..out_ch {
                    let mut sum = 0.0f32;
                    for i in 0..in_ch {
                        sum += w[i * out_ch + o] * input_data[in_col_start + i];
                    }
                    self.output_buf.data[out_col_start + o] = sum;
                }
            }
            if let Some(ref b) = self.bias {
                for f in 0..num_frames {
                    let col_start = f * out_ch;
                    for o in 0..out_ch {
                        self.output_buf.data[col_start + o] += b[o];
                    }
                }
            }
        }
    }
}

// ── Conv1D (dilated, with groups support) ───────────────────────────────────

/// Dilated 1D convolution with ring buffer. Supports grouped convolution.
/// Weight stored per kernel tap in column-major order matching Eigen.
struct Conv1d {
    /// Weight per kernel tap. weights_colmajor[k] is column-major [out_ch * in_ch]
    /// where W_k(i, j) = weights_colmajor[k][j * out_ch + i]
    weights_colmajor: Vec<Vec<f32>>,
    bias: Vec<f32>,
    kernel_size: usize,
    dilation: usize,
    out_channels: usize,
    in_channels: usize,
    #[allow(dead_code)]
    groups: usize,
    // Block processing state
    input_buffer: RingBuffer2D,
    output_buf: ColMajorMatrix,
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

        let mut tap_weights_colmajor: Vec<Vec<f32>> = (0..kernel_size)
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
                        // column-major: index = col * out_channels + row
                        tap_weights_colmajor[k][col * out_channels + row] = taps[k];
                    }
                }
            }
        }

        let bias = iter.take(out_channels)?.to_vec();

        Ok(Self {
            weights_colmajor: tap_weights_colmajor,
            bias,
            kernel_size,
            dilation,
            out_channels,
            in_channels,
            groups,
            input_buffer: RingBuffer2D::new(),
            output_buf: ColMajorMatrix::new(out_channels, 1),
        })
    }

    /// Receptive field (zero-indexed): dilation * (kernel_size - 1).
    fn receptive_field(&self) -> usize {
        self.dilation * (self.kernel_size - 1)
    }

    fn set_max_buffer_size(&mut self, max_buffer_size: usize) {
        let rf = self.receptive_field();
        self.input_buffer.set_max_lookback(rf);
        self.input_buffer.reset(self.in_channels, max_buffer_size);
        self.output_buf.resize(self.out_channels, max_buffer_size);
    }

    /// Block processing matching C++ Conv1D::Process.
    /// 1. Write input to ring buffer
    /// 2. For each kernel tap k: read from ring buffer with lookback, accumulate GEMM
    /// 3. Add bias
    fn process_block(&mut self, input: &ColMajorMatrix, num_frames: usize) {
        // Write input to ring buffer
        self.input_buffer.write(input, num_frames);

        let out_ch = self.out_channels;
        let in_ch = self.in_channels;
        let ks = self.kernel_size;
        let dil = self.dilation;

        let use_sgemm = out_ch * in_ch >= SGEMM_MIN_SIZE;

        // Initialize output to zero, then accumulate
        let out_len = out_ch * num_frames;
        self.output_buf.data[..out_len].fill(0.0);

        for k in 0..ks {
            // C++ offset = dilation * (k + 1 - kernel_size), lookback = -offset
            let offset_signed: isize = dil as isize * (k as isize + 1 - ks as isize);
            let lookback = (-offset_signed) as usize;

            let tap_data = self.input_buffer.read_ptr(num_frames, lookback);
            let w = &self.weights_colmajor[k];

            if use_sgemm {
                unsafe {
                    sgemm_colmajor(
                        out_ch, in_ch, num_frames, 1.0,
                        w.as_ptr(),
                        tap_data.as_ptr(), in_ch as isize,
                        1.0, self.output_buf.data.as_mut_ptr(),
                    );
                }
            } else {
                for f in 0..num_frames {
                    let in_col_start = f * in_ch;
                    let out_col_start = f * out_ch;
                    for o in 0..out_ch {
                        let mut sum = 0.0f32;
                        for i in 0..in_ch {
                            sum += w[i * out_ch + o] * tap_data[in_col_start + i];
                        }
                        self.output_buf.data[out_col_start + o] += sum;
                    }
                }
            }
        }

        // Add bias (colwise)
        for f in 0..num_frames {
            let col_start = f * out_ch;
            for o in 0..out_ch {
                self.output_buf.data[col_start + o] += self.bias[o];
            }
        }

        // Advance ring buffer write pointer
        self.input_buffer.advance(num_frames);
    }

    #[allow(dead_code)]
    fn zero_state(&mut self) {
        self.input_buffer.zero();
    }
}

// ── FiLM (Feature-wise Linear Modulation) ───────────────────────────────────

struct FiLM {
    cond_to_scale_shift: Conv1x1,
    do_shift: bool,
    input_dim: usize,
    // Pre-allocated output buffer
    output_buf: ColMajorMatrix,
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
            output_buf: ColMajorMatrix::new(input_dim, 1),
        })
    }

    fn set_max_buffer_size(&mut self, max_buffer_size: usize) {
        self.cond_to_scale_shift
            .set_max_buffer_size(max_buffer_size);
        self.output_buf.resize(self.input_dim, max_buffer_size);
    }

    /// Block FiLM: output = input * scale (+ shift)
    /// Writes result to self.output_buf (input_dim x num_frames, column-major)
    fn process_block(
        &mut self,
        input: &ColMajorMatrix,
        condition: &ColMajorMatrix,
        num_frames: usize,
    ) {
        self.cond_to_scale_shift
            .process_block(condition, num_frames);
        let scale_shift = &self.cond_to_scale_shift.output_buf;
        let ss_rows = self.cond_to_scale_shift.out_channels;
        let dim = self.input_dim;

        if self.do_shift {
            for f in 0..num_frames {
                let in_off = f * input.rows;
                let ss_off = f * ss_rows;
                let out_off = f * dim;
                for i in 0..dim {
                    self.output_buf.data[out_off + i] = input.data[in_off + i]
                        * scale_shift.data[ss_off + i]
                        + scale_shift.data[ss_off + dim + i];
                }
            }
        } else {
            for f in 0..num_frames {
                let in_off = f * input.rows;
                let ss_off = f * ss_rows;
                let out_off = f * dim;
                for i in 0..dim {
                    self.output_buf.data[out_off + i] =
                        input.data[in_off + i] * scale_shift.data[ss_off + i];
                }
            }
        }
    }

    /// Block FiLM with input data that has a different stride (e.g. topRows of a larger matrix)
    fn process_block_with_stride(
        &mut self,
        input_data: &[f32],
        input_stride: usize,
        condition: &ColMajorMatrix,
        num_frames: usize,
    ) {
        self.cond_to_scale_shift
            .process_block(condition, num_frames);
        let scale_shift = &self.cond_to_scale_shift.output_buf;
        let ss_rows = self.cond_to_scale_shift.out_channels;
        let dim = self.input_dim;

        if self.do_shift {
            for f in 0..num_frames {
                let in_off = f * input_stride;
                let ss_off = f * ss_rows;
                let out_off = f * dim;
                for i in 0..dim {
                    self.output_buf.data[out_off + i] = input_data[in_off + i]
                        * scale_shift.data[ss_off + i]
                        + scale_shift.data[ss_off + dim + i];
                }
            }
        } else {
            for f in 0..num_frames {
                let in_off = f * input_stride;
                let ss_off = f * ss_rows;
                let out_off = f * dim;
                for i in 0..dim {
                    self.output_buf.data[out_off + i] =
                        input_data[in_off + i] * scale_shift.data[ss_off + i];
                }
            }
        }
    }

    /// In-place FiLM: modifies target_data in-place
    fn process_block_inplace(
        &mut self,
        target_data: &mut [f32],
        target_stride: usize,
        condition: &ColMajorMatrix,
        num_frames: usize,
    ) {
        self.cond_to_scale_shift
            .process_block(condition, num_frames);
        let scale_shift = &self.cond_to_scale_shift.output_buf;
        let ss_rows = self.cond_to_scale_shift.out_channels;
        let dim = self.input_dim;

        if self.do_shift {
            for f in 0..num_frames {
                let t_off = f * target_stride;
                let ss_off = f * ss_rows;
                for i in 0..dim {
                    target_data[t_off + i] = target_data[t_off + i] * scale_shift.data[ss_off + i]
                        + scale_shift.data[ss_off + dim + i];
                }
            }
        } else {
            for f in 0..num_frames {
                let t_off = f * target_stride;
                let ss_off = f * ss_rows;
                for i in 0..dim {
                    target_data[t_off + i] *= scale_shift.data[ss_off + i];
                }
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

    // Skip the head copy optimization (C++ _skip_head_copy)
    skip_head_copy: bool,

    // Pre-allocated block processing buffers
    z_buf: ColMajorMatrix,
    output_next_layer: ColMajorMatrix,
    output_head: ColMajorMatrix,
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

        // C++ _skip_head_copy: when no head1x1 and no gating, GetOutputHead returns _z directly
        let skip_head_copy = !head1x1_params.active && gating_mode == GatingMode::None;

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
            skip_head_copy,
            z_buf: ColMajorMatrix::new(conv_out, 1),
            output_next_layer: ColMajorMatrix::new(channels, 1),
            output_head: ColMajorMatrix::new(head_output_size, 1),
        })
    }

    fn set_max_buffer_size(&mut self, max_buffer_size: usize) {
        self.conv.set_max_buffer_size(max_buffer_size);
        self.input_mixin.set_max_buffer_size(max_buffer_size);

        let z_channels = self.conv.out_channels;
        self.z_buf.resize(z_channels, max_buffer_size);

        if let Some(ref mut l) = self.layer1x1 {
            l.set_max_buffer_size(max_buffer_size);
        }

        let channels = self.channels;
        self.output_next_layer.resize(channels, max_buffer_size);

        if let Some(ref mut h) = self.head1x1 {
            self.output_head.resize(h.out_channels, max_buffer_size);
            h.set_max_buffer_size(max_buffer_size);
        } else {
            self.output_head.resize(self.bottleneck, max_buffer_size);
        }

        // FiLM set_max_buffer_size
        if let Some(ref mut f) = self.conv_pre_film {
            f.set_max_buffer_size(max_buffer_size);
        }
        if let Some(ref mut f) = self.conv_post_film {
            f.set_max_buffer_size(max_buffer_size);
        }
        if let Some(ref mut f) = self.input_mixin_pre_film {
            f.set_max_buffer_size(max_buffer_size);
        }
        if let Some(ref mut f) = self.input_mixin_post_film {
            f.set_max_buffer_size(max_buffer_size);
        }
        if let Some(ref mut f) = self.activation_pre_film {
            f.set_max_buffer_size(max_buffer_size);
        }
        if let Some(ref mut f) = self.activation_post_film {
            f.set_max_buffer_size(max_buffer_size);
        }
        if let Some(ref mut f) = self.layer1x1_post_film {
            f.set_max_buffer_size(max_buffer_size);
        }
        if let Some(ref mut f) = self.head1x1_post_film {
            f.set_max_buffer_size(max_buffer_size);
        }
    }

    /// Block processing matching C++ _Layer::Process.
    /// input: (channels x num_frames), condition: (condition_size x num_frames)
    /// Results stored in self.output_next_layer and self.output_head.
    fn process_block(
        &mut self,
        input: &ColMajorMatrix,
        condition: &ColMajorMatrix,
        num_frames: usize,
    ) {
        let bottleneck = self.bottleneck;
        let z_rows = self.conv.out_channels; // 2*bottleneck when gated, bottleneck when not

        // Step 1: Input convolution
        if let Some(ref mut film) = self.conv_pre_film {
            // FiLM modulate input, then conv
            film.process_block(input, condition, num_frames);
            self.conv.process_block(&film.output_buf, num_frames);
        } else {
            self.conv.process_block(input, num_frames);
        }

        if let Some(ref mut film) = self.conv_post_film {
            // In-place modulate conv output
            film.process_block_inplace(
                &mut self.conv.output_buf.data,
                self.conv.out_channels,
                condition,
                num_frames,
            );
        }

        // Step 2: Input mixin
        if let Some(ref mut film) = self.input_mixin_pre_film {
            // FiLM modulate condition, then mixin
            film.process_block(condition, condition, num_frames);
            self.input_mixin.process_block(&film.output_buf, num_frames);
        } else {
            self.input_mixin.process_block(condition, num_frames);
        }

        if let Some(ref mut film) = self.input_mixin_post_film {
            film.process_block_inplace(
                &mut self.input_mixin.output_buf.data,
                self.input_mixin.out_channels,
                condition,
                num_frames,
            );
        }

        // z = conv_output + mixin_output
        let z_len = z_rows * num_frames;
        for i in 0..z_len {
            self.z_buf.data[i] = self.conv.output_buf.data[i] + self.input_mixin.output_buf.data[i];
        }

        // Optional activation_pre_film
        if let Some(ref mut film) = self.activation_pre_film {
            film.process_block_inplace(&mut self.z_buf.data, z_rows, condition, num_frames);
        }

        // Step 3: Activation + gating/blending
        match self.gating_mode {
            GatingMode::None => {
                // Apply activation in-place to z
                self.activation.apply_colmajor_inplace(
                    &mut self.z_buf.data,
                    bottleneck,
                    num_frames,
                );

                // Optional activation_post_film
                if let Some(ref mut film) = self.activation_post_film {
                    film.process_block_inplace(&mut self.z_buf.data, z_rows, condition, num_frames);
                }

                // layer1x1
                if let Some(ref mut l1x1) = self.layer1x1 {
                    l1x1.process_block(&self.z_buf, num_frames);
                }
            }
            GatingMode::Gated => {
                // Gating: top bottleneck = primary activation, bottom bottleneck = secondary
                // output[i] = primary(z[i]) * secondary(z[i + bottleneck])
                // Process column by column to match C++ per-sample gating
                for f in 0..num_frames {
                    let z_off = f * z_rows;
                    for c in 0..bottleneck {
                        let primary = self
                            .activation
                            .apply_scalar_channel(self.z_buf.data[z_off + c], c);
                        let gate = self
                            .secondary_activation
                            .apply_scalar_channel(self.z_buf.data[z_off + bottleneck + c], c);
                        // Store result in top rows of z
                        self.z_buf.data[z_off + c] = primary * gate;
                    }
                }

                // activation_post_film on topRows(bottleneck)
                if let Some(ref mut film) = self.activation_post_film {
                    // C++: Process() then copy back (non-inplace for gated/blended)
                    film.process_block_with_stride(&self.z_buf.data, z_rows, condition, num_frames);
                    // Copy back to z topRows
                    for f in 0..num_frames {
                        let z_off = f * z_rows;
                        let film_off = f * bottleneck;
                        self.z_buf.data[z_off..z_off + bottleneck].copy_from_slice(
                            &film.output_buf.data[film_off..film_off + bottleneck],
                        );
                    }
                }

                // layer1x1 processes topRows(bottleneck)
                if let Some(ref mut l1x1) = self.layer1x1 {
                    l1x1.process_block_with_stride(&self.z_buf.data, z_rows, num_frames);
                }
            }
            GatingMode::Blended => {
                // Blending: alpha * activated + (1-alpha) * pre_activation
                for f in 0..num_frames {
                    let z_off = f * z_rows;
                    for c in 0..bottleneck {
                        let pre_act = self.z_buf.data[z_off + c];
                        let activated = self.activation.apply_scalar_channel(pre_act, c);
                        let alpha = self
                            .secondary_activation
                            .apply_scalar_channel(self.z_buf.data[z_off + bottleneck + c], c);
                        self.z_buf.data[z_off + c] = alpha * activated + (1.0 - alpha) * pre_act;
                    }
                }

                // activation_post_film
                if let Some(ref mut film) = self.activation_post_film {
                    film.process_block_with_stride(&self.z_buf.data, z_rows, condition, num_frames);
                    for f in 0..num_frames {
                        let z_off = f * z_rows;
                        let film_off = f * bottleneck;
                        self.z_buf.data[z_off..z_off + bottleneck].copy_from_slice(
                            &film.output_buf.data[film_off..film_off + bottleneck],
                        );
                    }
                }

                // layer1x1
                if let Some(ref mut l1x1) = self.layer1x1 {
                    l1x1.process_block_with_stride(&self.z_buf.data, z_rows, num_frames);

                    // layer1x1_post_film only in BLENDED mode (matching C++)
                    if let Some(ref mut film) = self.layer1x1_post_film {
                        film.process_block_inplace(
                            &mut l1x1.output_buf.data,
                            l1x1.out_channels,
                            condition,
                            num_frames,
                        );
                    }
                }
            }
        }

        // Step 4: Head output (head1x1 or direct from z/activated)
        if let Some(ref mut h1x1) = self.head1x1 {
            if self.gating_mode == GatingMode::None {
                h1x1.process_block(&self.z_buf, num_frames);
            } else {
                h1x1.process_block_with_stride(&self.z_buf.data, z_rows, num_frames);
            }

            if let Some(ref mut film) = self.head1x1_post_film {
                film.process_block_inplace(
                    &mut h1x1.output_buf.data,
                    h1x1.out_channels,
                    condition,
                    num_frames,
                );
            }

            // Copy to output_head
            let h_out = h1x1.out_channels;
            let len = h_out * num_frames;
            self.output_head.data[..len].copy_from_slice(&h1x1.output_buf.data[..len]);
        } else if !self.skip_head_copy {
            // Copy from z (topRows if gated)
            let head_rows = self.output_head.rows;
            if self.gating_mode == GatingMode::None {
                let len = head_rows * num_frames;
                self.output_head.data[..len].copy_from_slice(&self.z_buf.data[..len]);
            } else {
                // Copy topRows(bottleneck) from z which has z_rows stride
                for f in 0..num_frames {
                    let z_off = f * z_rows;
                    let out_off = f * head_rows;
                    self.output_head.data[out_off..out_off + head_rows]
                        .copy_from_slice(&self.z_buf.data[z_off..z_off + head_rows]);
                }
            }
        }
        // If skip_head_copy, output_head is z itself (caller reads from z_buf)

        // Step 5: Output to next layer = input + layer1x1_output (or just input)
        let ch = self.channels;
        if let Some(ref l1x1) = self.layer1x1 {
            let total = ch * num_frames;
            for i in 0..total {
                self.output_next_layer.data[i] = input.data[i] + l1x1.output_buf.data[i];
            }
        } else {
            let total = ch * num_frames;
            self.output_next_layer.data[..total].copy_from_slice(&input.data[..total]);
        }
    }

    /// Get head output data (may be z_buf if skip_head_copy)
    fn get_output_head_data(&self) -> &[f32] {
        if self.skip_head_copy {
            &self.z_buf.data
        } else {
            &self.output_head.data
        }
    }

    fn get_output_head_rows(&self) -> usize {
        if self.skip_head_copy {
            self.z_buf.rows
        } else {
            self.output_head.rows
        }
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

    // Pre-allocated block buffers
    layer_outputs: ColMajorMatrix,
    head_inputs: ColMajorMatrix,
}

impl WaveNetLayerArray {
    fn receptive_field(&self) -> usize {
        self.layers.iter().map(|l| l.conv.receptive_field()).sum()
    }

    fn set_max_buffer_size(&mut self, max_buffer_size: usize) {
        self.rechannel.set_max_buffer_size(max_buffer_size);
        self.head_rechannel.set_max_buffer_size(max_buffer_size);
        for layer in &mut self.layers {
            layer.set_max_buffer_size(max_buffer_size);
        }
        self.layer_outputs.resize(self.channels, max_buffer_size);
        self.head_inputs
            .resize(self.head_output_size, max_buffer_size);
    }

    /// Process without previous head input (first layer array).
    /// Matches C++ _LayerArray::Process (2-arg version)
    fn process_first(
        &mut self,
        layer_inputs: &ColMajorMatrix,
        condition: &ColMajorMatrix,
        num_frames: usize,
    ) {
        // Zero head inputs accumulator
        self.head_inputs.zero_cols(num_frames);
        self.process_inner(layer_inputs, condition, num_frames);
    }

    /// Process with previous head input (subsequent layer arrays).
    /// Matches C++ _LayerArray::Process (3-arg version)
    fn process_subsequent(
        &mut self,
        layer_inputs: &ColMajorMatrix,
        condition: &ColMajorMatrix,
        head_inputs: &ColMajorMatrix,
        num_frames: usize,
    ) {
        // Copy head inputs from previous layer array
        let len = self.head_output_size * num_frames;
        self.head_inputs.data[..len].copy_from_slice(&head_inputs.data[..len]);
        self.process_inner(layer_inputs, condition, num_frames);
    }

    /// Common inner processing. Matches C++ _LayerArray::ProcessInner
    fn process_inner(
        &mut self,
        layer_inputs: &ColMajorMatrix,
        condition: &ColMajorMatrix,
        num_frames: usize,
    ) {
        // Rechannel: project input to layer channels
        self.rechannel.process_block(layer_inputs, num_frames);

        // Process layers
        let num_layers = self.layers.len();

        // We need to handle the borrowing carefully.
        // Layer i reads from: rechannel output (if i==0) or layer[i-1].output_next_layer
        // Layer i writes to: its own output_next_layer and output_head
        // Head inputs accumulate from each layer's output_head

        for i in 0..num_layers {
            // Get the input for this layer
            // We use unsafe to work around the borrow checker, since we're reading from
            // layer[i-1].output_next_layer while writing to layer[i].
            // This is safe because we're accessing different layers.
            if i == 0 {
                // First layer uses rechannel output
                let input_ptr = &self.rechannel.output_buf as *const ColMajorMatrix;
                let layer = &mut self.layers[i];
                layer.process_block(unsafe { &*input_ptr }, condition, num_frames);
            } else {
                // Subsequent layers use previous layer's output
                let prev_output = &self.layers[i - 1].output_next_layer as *const ColMajorMatrix;
                let layer = &mut self.layers[i];
                layer.process_block(unsafe { &*prev_output }, condition, num_frames);
            }

            // Accumulate head output from this layer
            let head_out_size = self.head_output_size;
            let layer = &self.layers[i];
            let head_data = layer.get_output_head_data();
            let head_rows = layer.get_output_head_rows();
            for f in 0..num_frames {
                let src_off = f * head_rows;
                let dst_off = f * head_out_size;
                for c in 0..head_out_size {
                    self.head_inputs.data[dst_off + c] += head_data[src_off + c];
                }
            }
        }

        // Store output from last layer
        let last = num_layers - 1;
        let ch = self.channels;
        let len = ch * num_frames;
        self.layer_outputs.data[..len]
            .copy_from_slice(&self.layers[last].output_next_layer.data[..len]);

        // Head rechannel
        self.head_rechannel
            .process_block(&self.head_inputs, num_frames);
    }
}

// ── Top-level WaveNet ───────────────────────────────────────────────────────

pub struct WaveNet {
    layer_arrays: Vec<WaveNetLayerArray>,
    head_scale: f32,
    prewarm_samples_count: usize,
    metadata: DspMetadata,
    in_channels: usize,

    // Optional condition DSP
    condition_dsp: Option<Box<dyn Dsp>>,

    // Block processing buffers
    condition_input: ColMajorMatrix,
    condition_output: ColMajorMatrix,
    max_buffer_size: usize,
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
        let mut condition_size = 0usize;

        let in_channels = config
            .get("in_channels")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;

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

            // Build layer array
            // Per C++ weight order: rechannel, then each layer, then head_rechannel
            let rechannel = Conv1x1::from_weights(input_size, channels, false, 1, &mut iter)?;

            let mut layers = Vec::new();

            for (layer_idx, &dil) in dilations.iter().enumerate() {
                let layer_gating = gating_modes[layer_idx];

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
                layer_outputs: ColMajorMatrix::new(channels, 1),
                head_inputs: ColMajorMatrix::new(head_out_size, 1),
            });
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

        // Determine condition output channels
        let cond_out_ch = if let Some(ref cdsp) = condition_dsp {
            cdsp.num_output_channels()
        } else {
            in_channels
        };

        Ok(Self {
            layer_arrays,
            head_scale,
            prewarm_samples_count,
            metadata,
            in_channels,
            condition_dsp,
            condition_input: ColMajorMatrix::new(in_channels, 1),
            condition_output: ColMajorMatrix::new(cond_out_ch.max(condition_size), 1),
            max_buffer_size: 0,
        })
    }

    fn ensure_buffer_size(&mut self, num_frames: usize) {
        if num_frames <= self.max_buffer_size {
            return;
        }
        // Growing the buffer size requires re-initializing ring buffers,
        // which loses accumulated state. This mirrors C++ behavior where
        // SetMaxBufferSize is called during Reset before prewarm.
        self.set_max_buffer_size_internal(num_frames);
    }

    fn set_max_buffer_size_internal(&mut self, max_buffer_size: usize) {
        self.max_buffer_size = max_buffer_size;

        self.condition_input
            .resize(self.in_channels, max_buffer_size);
        let cond_rows = self.condition_output.rows;
        self.condition_output.resize(cond_rows, max_buffer_size);

        for la in &mut self.layer_arrays {
            la.set_max_buffer_size(max_buffer_size);
        }
    }

    /// Block processing matching C++ WaveNet::process
    fn process_block(&mut self, input: &[Sample], output: &mut [Sample]) {
        let num_frames = input.len();
        if num_frames == 0 {
            return;
        }

        self.ensure_buffer_size(num_frames);

        // Step 1: Fill condition_input (in_channels x num_frames)
        // For standard NAM, in_channels = 1, so each column has one element
        let in_ch = self.in_channels;
        for f in 0..num_frames {
            let col_start = f * in_ch;
            self.condition_input.data[col_start] = input[f] as f32;
            // Zero remaining channels if any (unlikely for NAM)
            for c in 1..in_ch {
                self.condition_input.data[col_start + c] = 0.0;
            }
        }

        // Step 2: Process condition
        if let Some(ref mut cdsp) = self.condition_dsp {
            // Process condition_dsp per-sample for multi-channel output
            let cond_out_ch = cdsp.num_output_channels();
            for f in 0..num_frames {
                let in_sample = input[f];
                let col_start = f * self.condition_output.rows;
                cdsp.process_sample_multi_channel(
                    in_sample,
                    &mut self.condition_output.data[col_start..col_start + cond_out_ch],
                );
            }
        } else {
            // No condition DSP: condition_output = condition_input
            let cond_rows = self.condition_output.rows;
            let in_rows = self.condition_input.rows;
            let copy_rows = cond_rows.min(in_rows);
            for f in 0..num_frames {
                let cond_off = f * cond_rows;
                let in_off = f * in_rows;
                self.condition_output.data[cond_off..cond_off + copy_rows]
                    .copy_from_slice(&self.condition_input.data[in_off..in_off + copy_rows]);
            }
        }

        // Step 3: Process layer arrays
        let num_arrays = self.layer_arrays.len();

        for arr_idx in 0..num_arrays {
            if arr_idx == 0 {
                // First layer array: use condition_input as layer_inputs, condition_output as condition
                let cond_input_ptr = &self.condition_input as *const ColMajorMatrix;
                let cond_output_ptr = &self.condition_output as *const ColMajorMatrix;
                self.layer_arrays[arr_idx].process_first(
                    unsafe { &*cond_input_ptr },
                    unsafe { &*cond_output_ptr },
                    num_frames,
                );
            } else {
                // Subsequent: use previous array's layer_outputs and head_outputs
                let prev_layer_outputs =
                    &self.layer_arrays[arr_idx - 1].layer_outputs as *const ColMajorMatrix;
                let prev_head_outputs = &self.layer_arrays[arr_idx - 1].head_rechannel.output_buf
                    as *const ColMajorMatrix;
                let cond_output_ptr = &self.condition_output as *const ColMajorMatrix;
                self.layer_arrays[arr_idx].process_subsequent(
                    unsafe { &*prev_layer_outputs },
                    unsafe { &*cond_output_ptr },
                    unsafe { &*prev_head_outputs },
                    num_frames,
                );
            }
        }

        // Step 4: Extract output from final head
        let last = num_arrays - 1;
        let final_head = &self.layer_arrays[last].head_rechannel.output_buf;
        let out_ch = self.layer_arrays[last].head_rechannel.out_channels;

        // For single-channel output (typical NAM): data is contiguous
        if out_ch == 1 {
            let scale = self.head_scale;
            for s in 0..num_frames {
                output[s] = (scale * final_head.data[s]) as Sample;
            }
        } else {
            // Multi-channel: take first channel (or scale all)
            let scale = self.head_scale;
            for s in 0..num_frames {
                output[s] = (scale * final_head.data[s * out_ch]) as Sample;
            }
        }
    }

    /// Per-sample processing (fallback for process_sample_multi_channel).
    /// Uses the block path with num_frames=1.
    fn process_sample_for_multi_channel(&mut self, input_sample: f32) {
        self.ensure_buffer_size(1);

        let in_ch = self.in_channels;
        self.condition_input.data[0] = input_sample;
        for c in 1..in_ch {
            self.condition_input.data[c] = 0.0;
        }

        // Process condition
        if let Some(ref mut cdsp) = self.condition_dsp {
            let cond_out_ch = cdsp.num_output_channels();
            cdsp.process_sample_multi_channel(
                input_sample as Sample,
                &mut self.condition_output.data[..cond_out_ch],
            );
        } else {
            let cond_rows = self.condition_output.rows;
            let in_rows = self.condition_input.rows;
            let copy_rows = cond_rows.min(in_rows);
            self.condition_output.data[..copy_rows]
                .copy_from_slice(&self.condition_input.data[..copy_rows]);
        }

        // Process layer arrays with num_frames=1
        let num_arrays = self.layer_arrays.len();
        for arr_idx in 0..num_arrays {
            if arr_idx == 0 {
                let cond_input_ptr = &self.condition_input as *const ColMajorMatrix;
                let cond_output_ptr = &self.condition_output as *const ColMajorMatrix;
                self.layer_arrays[arr_idx].process_first(
                    unsafe { &*cond_input_ptr },
                    unsafe { &*cond_output_ptr },
                    1,
                );
            } else {
                let prev_layer_outputs =
                    &self.layer_arrays[arr_idx - 1].layer_outputs as *const ColMajorMatrix;
                let prev_head_outputs = &self.layer_arrays[arr_idx - 1].head_rechannel.output_buf
                    as *const ColMajorMatrix;
                let cond_output_ptr = &self.condition_output as *const ColMajorMatrix;
                self.layer_arrays[arr_idx].process_subsequent(
                    unsafe { &*prev_layer_outputs },
                    unsafe { &*cond_output_ptr },
                    unsafe { &*prev_head_outputs },
                    1,
                );
            }
        }
    }
}

impl Dsp for WaveNet {
    fn process(&mut self, input: &[Sample], output: &mut [Sample]) {
        self.process_block(input, output);
    }

    fn reset(&mut self, sample_rate: f64, max_buffer_size: usize) {
        // Match C++ Reset: SetMaxBufferSize first (re-allocates and zeros all buffers),
        // then reset condition_dsp. Prewarm is called separately by the caller.
        self.set_max_buffer_size_internal(max_buffer_size);
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
        self.process_sample_for_multi_channel(input_sample as f32);

        let last = self.layer_arrays.len() - 1;
        let final_head = &self.layer_arrays[last].head_rechannel.output_buf;
        let out_ch = self.layer_arrays[last].head_rechannel.out_channels;
        let scale = self.head_scale;

        for (i, o) in out.iter_mut().enumerate() {
            if i < out_ch {
                *o = scale * final_head.data[i];
            }
        }
    }

    fn prewarm_samples(&self) -> usize {
        self.prewarm_samples_count
    }

    fn metadata(&self) -> &DspMetadata {
        &self.metadata
    }
}

// ── Activation helper for column-major block processing ─────────────────────

impl Activation {
    /// Apply activation in-place to a column-major matrix.
    /// The matrix has `rows` per column and `num_cols` columns.
    /// Applies per-channel (row) for PReLU.
    fn apply_colmajor_inplace(&self, data: &mut [f32], rows: usize, num_cols: usize) {
        match self {
            Activation::PReLU(slopes) => {
                for f in 0..num_cols {
                    let off = f * rows;
                    for c in 0..rows {
                        let x = data[off + c];
                        let alpha = slopes.get(c).copied().unwrap_or(0.01);
                        data[off + c] = if x >= 0.0 { x } else { alpha * x };
                    }
                }
            }
            _ => {
                let len = rows * num_cols;
                for x in data[..len].iter_mut() {
                    *x = self.apply_scalar(*x);
                }
            }
        }
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
