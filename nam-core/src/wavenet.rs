use crate::activations::Activation;
use crate::dsp::{ActivationMode, Dsp, DspMetadata, Sample};
use crate::error::NamError;
use crate::util::{
    checked_dimension_add, checked_dimension_mul, config_usize, positive_config_usize, WeightIter,
};

mod a1_conv_backend;
mod matrix_backend;

use a1_conv_backend::PackedA1Conv;
use matrix_backend::{MatrixLayout, SGEMM_MIN_SIZE};

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

    fn from_json(val: &serde_json::Value, field: &str) -> Result<Self, NamError> {
        if val.is_boolean() && !val.as_bool().unwrap_or(false) {
            return Ok(Self::inactive());
        }
        if val.is_null() {
            return Ok(Self::inactive());
        }
        if let Some(obj) = val.as_object() {
            let active = obj.get("active").and_then(|v| v.as_bool()).unwrap_or(true);
            let shift = obj.get("shift").and_then(|v| v.as_bool()).unwrap_or(true);
            let groups = obj
                .get("groups")
                .map(|value| positive_config_usize(value, &format!("{field}.groups")))
                .transpose()?
                .unwrap_or(1);
            Ok(Self {
                active,
                shift,
                groups,
            })
        } else {
            Ok(Self::inactive())
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

fn zero_inactive_rows(data: &mut [f32], rows: usize, active_rows: usize, num_cols: usize) {
    if active_rows >= rows {
        return;
    }
    for f in 0..num_cols {
        let off = f * rows;
        data[off + active_rows..off + rows].fill(0.0);
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
        let needed = channels * self.storage_cols;
        self.storage.resize(needed, 0.0);
        self.storage[..needed].fill(0.0);
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
        let start = self.read_offset(lookback);
        &self.storage[start..]
    }

    fn read_offset(&self, lookback: usize) -> usize {
        (self.write_pos - lookback) * self.channels
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
        let len = self.max_lookback * ch;
        let src_start = copy_start * ch;
        // copy_start >= max_lookback (by C++ invariant), so regions don't overlap.
        self.storage.copy_within(src_start..src_start + len, 0);
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
    matrix_layout: MatrixLayout,
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
        let matrix_layout = MatrixLayout::new(out_channels, in_channels).map_err(|_| {
            NamError::DimensionOverflow {
                context: "1x1 convolution weight matrix",
                left: out_channels,
                right: in_channels,
            }
        })?;
        let out_per_group = out_channels / groups;
        let in_per_group = in_channels / groups;

        // Build weight in column-major order matching Eigen layout
        // Eigen column-major: W(i,j) at index j * out_channels + i
        let mut weight_colmajor = vec![0.0f32; matrix_layout.left_len()];

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
            matrix_layout,
            groups,
            output_buf: ColMajorMatrix::new(out_channels, 1),
        })
    }

    fn set_max_buffer_size(&mut self, max_buffer_size: usize) {
        self.matrix_layout.set_max_buffer_size(max_buffer_size);
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
            self.matrix_layout.multiply(
                num_frames,
                1.0,
                &self.weight_colmajor,
                &input.data,
                input.rows,
                1.0,
                &mut self.output_buf.data,
            );
        } else {
            // Small matrix: specialized paths for common sizes, with fused bias
            self.process_block_small_gemm(&input.data, input.rows, num_frames);
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
            self.matrix_layout.multiply(
                num_frames,
                1.0,
                &self.weight_colmajor,
                input_data,
                input_stride,
                1.0,
                &mut self.output_buf.data,
            );
        } else {
            self.process_block_small_gemm(input_data, input_stride, num_frames);
        }
    }

    /// Evaluate small matrix products in Eigen's product-then-bias order.
    #[inline]
    fn process_block_small_gemm(
        &mut self,
        input_data: &[f32],
        input_stride: usize,
        num_frames: usize,
    ) {
        let out_ch = self.out_channels;
        let in_ch = self.in_channels;
        let w = &self.weight_colmajor;
        let bias = &self.bias;
        let out = &mut self.output_buf.data;

        // Fast-kernels: route all sizes through C for -ffast-math vectorization
        #[cfg(feature = "fast-kernels")]
        {
            crate::fast_kernels::conv1x1_small(
                out,
                w,
                input_data,
                bias.as_deref(),
                out_ch,
                in_ch,
                input_stride,
                num_frames,
            );
            // Return early so the non-fast-kernels fallback below is skipped
            #[allow(clippy::needless_return, unreachable_code)]
            return;
        }

        #[cfg(not(feature = "fast-kernels"))]
        for frame in 0..num_frames {
            let input_column = frame * input_stride;
            let output_column = frame * out_ch;
            for output_channel in 0..out_ch {
                // Keep the first product rounded before the bias is added.
                let mut product = w[output_channel].mul_add(input_data[input_column], 0.0);
                for input_channel in 1..in_ch {
                    product = w[input_channel * out_ch + output_channel]
                        .mul_add(input_data[input_column + input_channel], product);
                }
                out[output_column + output_channel] =
                    product + bias.as_ref().map_or(0.0, |values| values[output_channel]);
            }
        }
    }
}

// ── Conv1D (dilated, with groups support) ───────────────────────────────────

/// Dilated 1D convolution with ring buffer. Supports grouped convolution.
/// Depthwise vs general convolution weight storage.
/// Depthwise is used when groups == in_channels == out_channels, storing a
/// compact per-channel weight vector per tap instead of a full matrix.
enum Conv1dWeights {
    /// General (possibly grouped) convolution.
    /// weights_colmajor[k] is column-major [out_ch * in_ch]
    /// where W_k(i, j) = weights_colmajor[k][j * out_ch + i]
    General(Vec<Vec<f32>>),
    /// Depthwise convolution: depthwise_weights[k] is [channels],
    /// one weight per channel per kernel tap.
    Depthwise(Vec<Vec<f32>>),
}

/// Weight stored per kernel tap in column-major order matching Eigen.
struct Conv1d {
    weights: Conv1dWeights,
    packed_a1: Option<PackedA1Conv>,
    bias: Vec<f32>,
    kernel_size: usize,
    dilation: usize,
    receptive_field: usize,
    out_channels: usize,
    in_channels: usize,
    matrix_layout: MatrixLayout,
    #[allow(dead_code)]
    groups: usize,
    // Block processing state
    input_buffer: RingBuffer2D,
    output_buf: ColMajorMatrix,
    /// Flattened weights for C FFI: all taps concatenated.
    #[cfg(feature = "fast-kernels")]
    flat_weights: Vec<f32>,
    #[cfg(feature = "fast-kernels")]
    tap_offsets: Vec<usize>,
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
        Self::from_weights_with_bias(
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            groups,
            true,
            iter,
        )
    }

    fn from_weights_with_bias(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        groups: usize,
        has_bias: bool,
        iter: &mut WeightIter,
    ) -> Result<Self, NamError> {
        if in_channels == 0 {
            return Err(NamError::InvalidConfigField {
                field: "convolution.in_channels".into(),
                reason: "must be greater than zero",
            });
        }
        if out_channels == 0 {
            return Err(NamError::InvalidConfigField {
                field: "convolution.out_channels".into(),
                reason: "must be greater than zero",
            });
        }
        if kernel_size == 0 {
            return Err(NamError::InvalidConfigField {
                field: "convolution.kernel_size".into(),
                reason: "must be greater than zero",
            });
        }
        if groups == 0
            || !in_channels.is_multiple_of(groups)
            || !out_channels.is_multiple_of(groups)
        {
            return Err(NamError::InvalidConfigField {
                field: "convolution.groups".into(),
                reason: "must be positive and divide both input and output channels",
            });
        }
        let receptive_field =
            checked_dimension_mul("convolution receptive field", dilation, kernel_size - 1)?;
        let matrix_layout = MatrixLayout::new(out_channels, in_channels).map_err(|_| {
            NamError::DimensionOverflow {
                context: "convolution weight matrix",
                left: out_channels,
                right: in_channels,
            }
        })?;
        let is_depthwise = groups == in_channels && in_channels == out_channels;
        let flat_weight_len = checked_dimension_mul(
            "convolution weights",
            kernel_size,
            if is_depthwise {
                out_channels
            } else {
                matrix_layout.left_len()
            },
        )?;
        #[cfg(not(feature = "fast-kernels"))]
        let _ = flat_weight_len;

        let weights = if is_depthwise {
            // Depthwise: one weight per channel per kernel tap
            // C++ weight order: for each channel c, for each kernel tap k
            let mut dw: Vec<Vec<f32>> = (0..kernel_size)
                .map(|_| vec![0.0f32; in_channels])
                .collect();
            // Indexes dw[k][c] and taps[k] simultaneously, can't use iterators
            #[allow(clippy::needless_range_loop)]
            for c in 0..in_channels {
                let taps = iter.take(kernel_size)?;
                for k in 0..kernel_size {
                    dw[k][c] = taps[k];
                }
            }
            Conv1dWeights::Depthwise(dw)
        } else {
            // General (possibly grouped) convolution
            let out_per_group = out_channels / groups;
            let in_per_group = in_channels / groups;

            let mut tap_weights_colmajor: Vec<Vec<f32>> = (0..kernel_size)
                .map(|_| vec![0.0f32; matrix_layout.left_len()])
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
            Conv1dWeights::General(tap_weights_colmajor)
        };

        let bias = if has_bias {
            iter.take(out_channels)?.to_vec()
        } else {
            vec![0.0; out_channels]
        };
        let packed_a1 = match &weights {
            Conv1dWeights::General(taps) if groups == 1 => {
                PackedA1Conv::new(out_channels, in_channels, taps)
            }
            _ => None,
        };

        // Build flat weights for C FFI
        #[cfg(feature = "fast-kernels")]
        let flat_weights = match &weights {
            Conv1dWeights::Depthwise(dw) => {
                // flat_weights[k * ch + c]
                let mut flat = Vec::with_capacity(flat_weight_len);
                for tap in dw {
                    flat.extend_from_slice(tap);
                }
                flat
            }
            Conv1dWeights::General(taps) => {
                // flat_weights[k * (out_ch * in_ch) + ...], just concatenate
                let mut flat = Vec::with_capacity(flat_weight_len);
                for tap in taps {
                    flat.extend_from_slice(tap);
                }
                flat
            }
        };

        Ok(Self {
            weights,
            packed_a1,
            bias,
            kernel_size,
            dilation,
            receptive_field,
            out_channels,
            in_channels,
            matrix_layout,
            groups,
            input_buffer: RingBuffer2D::new(),
            output_buf: ColMajorMatrix::new(out_channels, 1),
            #[cfg(feature = "fast-kernels")]
            flat_weights,
            #[cfg(feature = "fast-kernels")]
            tap_offsets: vec![0; kernel_size],
        })
    }

    /// Receptive field (zero-indexed): dilation * (kernel_size - 1).
    fn receptive_field(&self) -> usize {
        self.receptive_field
    }

    fn set_max_buffer_size(&mut self, max_buffer_size: usize) {
        let rf = self.receptive_field();
        self.matrix_layout.set_max_buffer_size(max_buffer_size);
        if let Some(packed_a1) = &mut self.packed_a1 {
            packed_a1.set_max_buffer_size(max_buffer_size);
        }
        self.input_buffer.set_max_lookback(rf);
        self.input_buffer.reset(self.in_channels, max_buffer_size);
        self.output_buf.resize(self.out_channels, max_buffer_size);
    }

    /// Block processing matching C++ Conv1D::Process.
    /// 1. Write input to ring buffer
    /// 2. For each kernel tap k: read from ring buffer with lookback, accumulate
    /// 3. Add bias
    fn process_block(&mut self, input: &ColMajorMatrix, num_frames: usize) {
        // Write input to ring buffer
        self.input_buffer.write(input, num_frames);

        let out_ch = self.out_channels;
        let in_ch = self.in_channels;
        let ks = self.kernel_size;
        let dil = self.dilation;

        if let Some(packed_a1) = &mut self.packed_a1 {
            let right_offsets = core::array::from_fn(|k| {
                let offset_signed = dil as isize * (k as isize + 1 - ks as isize);
                self.input_buffer.read_offset((-offset_signed) as usize)
            });
            if packed_a1.process(
                num_frames,
                &self.input_buffer.storage,
                right_offsets,
                in_ch,
                &self.bias,
                &mut self.output_buf.data,
            ) {
                self.input_buffer.advance(num_frames);
                return;
            }
        }

        // Fast-kernels path: single C FFI call for the entire Conv1d
        #[cfg(feature = "fast-kernels")]
        {
            for k in 0..ks {
                let offset_signed: isize = dil as isize * (k as isize + 1 - ks as isize);
                let lookback = (-offset_signed) as usize;
                self.tap_offsets[k] = self.input_buffer.read_offset(lookback);
            }
            let use_sgemm = out_ch * in_ch >= SGEMM_MIN_SIZE;
            match &self.weights {
                Conv1dWeights::Depthwise(_) => {
                    crate::fast_kernels::conv1d_depthwise(
                        &mut self.output_buf.data,
                        &self.input_buffer.storage,
                        &self.tap_offsets,
                        &self.flat_weights,
                        &self.bias,
                        out_ch,
                        num_frames,
                    );
                }
                Conv1dWeights::General(_) if !use_sgemm => {
                    crate::fast_kernels::conv1d_small_gemv(
                        &mut self.output_buf.data,
                        &self.input_buffer.storage,
                        &self.tap_offsets,
                        &self.flat_weights,
                        &self.bias,
                        crate::fast_kernels::Conv1dDimensions {
                            out_channels: out_ch,
                            in_channels: in_ch,
                            num_frames,
                        },
                    );
                }
                Conv1dWeights::General(weights_colmajor) => {
                    // Large matrix: initialize with bias, then accumulate via sgemm
                    for f in 0..num_frames {
                        let off = f * out_ch;
                        self.output_buf.data[off..off + out_ch].copy_from_slice(&self.bias);
                    }
                    for (k, w) in weights_colmajor.iter().enumerate().take(ks) {
                        self.matrix_layout.multiply(
                            num_frames,
                            1.0,
                            w,
                            &self.input_buffer.storage[self.tap_offsets[k]..],
                            in_ch,
                            1.0,
                            &mut self.output_buf.data,
                        );
                    }
                }
            }
            self.input_buffer.advance(num_frames);
            // Return early so the non-fast-kernels fallback below is skipped
            #[allow(clippy::needless_return, unreachable_code)]
            return;
        }

        // Eigen accumulates every convolution tap before adding the bias.
        // (unreachable when fast-kernels feature is enabled, the block above returns early)
        #[allow(unreachable_code)]
        self.output_buf.data[..out_ch * num_frames].fill(0.0);

        match &self.weights {
            Conv1dWeights::Depthwise(dw) => {
                // Depthwise: element-wise multiply per channel per tap
                let ch = out_ch; // in_ch == out_ch for depthwise
                if ch == 3 {
                    // 3-channel specialization: fully unrolled inner loop
                    for (k, tap_w) in dw.iter().enumerate() {
                        let offset_signed: isize = dil as isize * (k as isize + 1 - ks as isize);
                        let lookback = (-offset_signed) as usize;
                        let tap_data = self.input_buffer.read_ptr(num_frames, lookback);
                        let w0 = tap_w[0];
                        let w1 = tap_w[1];
                        let w2 = tap_w[2];
                        for f in 0..num_frames {
                            let off = f * 3;
                            self.output_buf.data[off] =
                                w0.mul_add(tap_data[off], self.output_buf.data[off]);
                            self.output_buf.data[off + 1] =
                                w1.mul_add(tap_data[off + 1], self.output_buf.data[off + 1]);
                            self.output_buf.data[off + 2] =
                                w2.mul_add(tap_data[off + 2], self.output_buf.data[off + 2]);
                        }
                    }
                } else {
                    for (k, tap_w) in dw.iter().enumerate() {
                        let offset_signed: isize = dil as isize * (k as isize + 1 - ks as isize);
                        let lookback = (-offset_signed) as usize;
                        let tap_data = self.input_buffer.read_ptr(num_frames, lookback);
                        for f in 0..num_frames {
                            let col_start = f * ch;
                            for c in 0..ch {
                                self.output_buf.data[col_start + c] = tap_w[c].mul_add(
                                    tap_data[col_start + c],
                                    self.output_buf.data[col_start + c],
                                );
                            }
                        }
                    }
                }
            }
            Conv1dWeights::General(weights_colmajor) => {
                let use_sgemm = out_ch * in_ch >= SGEMM_MIN_SIZE;
                for (k, w) in weights_colmajor.iter().enumerate() {
                    let offset_signed: isize = dil as isize * (k as isize + 1 - ks as isize);
                    let lookback = (-offset_signed) as usize;
                    let tap_data = self.input_buffer.read_ptr(num_frames, lookback);

                    if out_ch == 1 && in_ch == 4 {
                        for frame in 0..num_frames {
                            let input = frame * 4;
                            // Preserve Eigen's pairwise reduction and product rounding.
                            let product = (w[0].mul_add(tap_data[input], 0.0)
                                + w[1].mul_add(tap_data[input + 1], 0.0))
                                + (w[2].mul_add(tap_data[input + 2], 0.0)
                                    + w[3].mul_add(tap_data[input + 3], 0.0));
                            self.output_buf.data[frame] += product;
                        }
                    } else if use_sgemm {
                        self.matrix_layout.multiply(
                            num_frames,
                            1.0,
                            w,
                            tap_data,
                            in_ch,
                            1.0,
                            &mut self.output_buf.data,
                        );
                    } else {
                        for f in 0..num_frames {
                            let in_col_start = f * in_ch;
                            let out_col_start = f * out_ch;
                            for o in 0..out_ch {
                                let mut sum = 0.0f32;
                                for i in 0..in_ch {
                                    sum =
                                        w[i * out_ch + o].mul_add(tap_data[in_col_start + i], sum);
                                }
                                self.output_buf.data[out_col_start + o] += sum;
                            }
                        }
                    }
                }
            }
        }

        for frame in 0..num_frames {
            let offset = frame * out_ch;
            for channel in 0..out_ch {
                self.output_buf.data[offset + channel] += self.bias[channel];
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
        self.apply_film_inner(&input.data, input.rows, num_frames);
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
        self.apply_film_inner(input_data, input_stride, num_frames);
    }

    /// Inner FiLM application with 3-channel specialization.
    #[inline]
    fn apply_film_inner(&mut self, input_data: &[f32], input_stride: usize, num_frames: usize) {
        let scale_shift = &self.cond_to_scale_shift.output_buf;
        let ss_rows = self.cond_to_scale_shift.out_channels;
        let dim = self.input_dim;

        #[cfg(feature = "fast-kernels")]
        {
            if self.do_shift {
                crate::fast_kernels::film_scale_shift(
                    &mut self.output_buf.data,
                    input_data,
                    &scale_shift.data,
                    dim,
                    input_stride,
                    dim,
                    ss_rows,
                    num_frames,
                );
            } else {
                crate::fast_kernels::film_scale(
                    &mut self.output_buf.data,
                    input_data,
                    &scale_shift.data,
                    dim,
                    input_stride,
                    dim,
                    ss_rows,
                    num_frames,
                );
            }
            // Return early so the non-fast-kernels fallback below is skipped
            #[allow(clippy::needless_return, unreachable_code)]
            return;
        }

        #[cfg(not(feature = "fast-kernels"))]
        if self.do_shift {
            if dim == 3 {
                for f in 0..num_frames {
                    let in_off = f * input_stride;
                    let ss_off = f * ss_rows;
                    let out_off = f * 3;
                    self.output_buf.data[out_off] = input_data[in_off]
                        .mul_add(scale_shift.data[ss_off], scale_shift.data[ss_off + 3]);
                    self.output_buf.data[out_off + 1] = input_data[in_off + 1]
                        .mul_add(scale_shift.data[ss_off + 1], scale_shift.data[ss_off + 4]);
                    self.output_buf.data[out_off + 2] = input_data[in_off + 2]
                        .mul_add(scale_shift.data[ss_off + 2], scale_shift.data[ss_off + 5]);
                }
            } else {
                for f in 0..num_frames {
                    let in_off = f * input_stride;
                    let ss_off = f * ss_rows;
                    let out_off = f * dim;
                    for i in 0..dim {
                        self.output_buf.data[out_off + i] = input_data[in_off + i].mul_add(
                            scale_shift.data[ss_off + i],
                            scale_shift.data[ss_off + dim + i],
                        );
                    }
                }
            }
        } else if dim == 3 {
            for f in 0..num_frames {
                let in_off = f * input_stride;
                let ss_off = f * ss_rows;
                let out_off = f * 3;
                self.output_buf.data[out_off] = input_data[in_off] * scale_shift.data[ss_off];
                self.output_buf.data[out_off + 1] =
                    input_data[in_off + 1] * scale_shift.data[ss_off + 1];
                self.output_buf.data[out_off + 2] =
                    input_data[in_off + 2] * scale_shift.data[ss_off + 2];
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

        #[cfg(feature = "fast-kernels")]
        {
            if self.do_shift {
                crate::fast_kernels::film_inplace_scale_shift(
                    target_data,
                    &scale_shift.data,
                    dim,
                    target_stride,
                    ss_rows,
                    num_frames,
                );
            } else {
                crate::fast_kernels::film_inplace_scale(
                    target_data,
                    &scale_shift.data,
                    dim,
                    target_stride,
                    ss_rows,
                    num_frames,
                );
            }
            // Return early so the non-fast-kernels fallback below is skipped
            #[allow(clippy::needless_return, unreachable_code)]
            return;
        }

        #[cfg(not(feature = "fast-kernels"))]
        if self.do_shift {
            for f in 0..num_frames {
                let t_off = f * target_stride;
                let ss_off = f * ss_rows;
                for i in 0..dim {
                    target_data[t_off + i] = target_data[t_off + i].mul_add(
                        scale_shift.data[ss_off + i],
                        scale_shift.data[ss_off + dim + i],
                    );
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

    fn zero_state(&mut self) {
        self.conv.zero_state();
    }

    /// Block processing matching C++ _Layer::Process.
    /// input: (channels x num_frames), condition: (condition_size x num_frames)
    /// Results stored in self.output_next_layer and self.output_head.
    fn process_block(
        &mut self,
        input: &ColMajorMatrix,
        condition: &ColMajorMatrix,
        num_frames: usize,
        active_channels: Option<usize>,
        use_fast_tanh: bool,
    ) {
        let bottleneck = self.bottleneck;
        let z_rows = self.conv.out_channels; // 2*bottleneck when gated, bottleneck when not
        let active_bottleneck = active_channels.unwrap_or(bottleneck).min(bottleneck);
        let active_z_rows = if active_channels.is_some() && self.gating_mode != GatingMode::None {
            (2 * active_bottleneck).min(z_rows)
        } else {
            active_bottleneck
        };

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

        // Fast path: fuse add + tanh activation into one C call
        #[cfg(feature = "fast-kernels")]
        let did_fused_add_activate = if self.activation_pre_film.is_none()
            && self.gating_mode == GatingMode::None
            && matches!(self.activation, Activation::Tanh)
        {
            crate::fast_kernels::add_activate(
                &mut self.z_buf.data,
                &self.conv.output_buf.data,
                &self.input_mixin.output_buf.data,
                z_len,
                use_fast_tanh,
            );
            true
        } else {
            false
        };

        #[cfg(not(feature = "fast-kernels"))]
        let did_fused_add_activate = false;

        if !did_fused_add_activate {
            #[cfg(feature = "fast-kernels")]
            crate::fast_kernels::vec_add(
                &mut self.z_buf.data,
                &self.conv.output_buf.data,
                &self.input_mixin.output_buf.data,
                z_len,
            );
            #[cfg(not(feature = "fast-kernels"))]
            for i in 0..z_len {
                self.z_buf.data[i] =
                    self.conv.output_buf.data[i] + self.input_mixin.output_buf.data[i];
            }
        }
        if active_channels.is_some() {
            zero_inactive_rows(&mut self.z_buf.data, z_rows, active_z_rows, num_frames);
        }

        // Optional activation_pre_film
        if let Some(ref mut film) = self.activation_pre_film {
            film.process_block_inplace(&mut self.z_buf.data, z_rows, condition, num_frames);
        }

        // Step 3: Activation + gating/blending
        match self.gating_mode {
            GatingMode::None => {
                if !did_fused_add_activate {
                    // Apply activation in-place to z
                    self.activation.apply_colmajor_inplace(
                        &mut self.z_buf.data,
                        bottleneck,
                        num_frames,
                        use_fast_tanh,
                    );
                }

                // Optional activation_post_film
                if let Some(ref mut film) = self.activation_post_film {
                    film.process_block_inplace(&mut self.z_buf.data, z_rows, condition, num_frames);
                }
                if active_channels.is_some() {
                    zero_inactive_rows(&mut self.z_buf.data, z_rows, active_bottleneck, num_frames);
                }

                // layer1x1
                if let Some(ref mut l1x1) = self.layer1x1 {
                    l1x1.process_block(&self.z_buf, num_frames);
                    if active_channels.is_some() {
                        zero_inactive_rows(
                            &mut l1x1.output_buf.data,
                            l1x1.out_channels,
                            active_channels
                                .unwrap_or(l1x1.out_channels)
                                .min(l1x1.out_channels),
                            num_frames,
                        );
                    }
                }
            }
            GatingMode::Gated => {
                // Gating: output[c] = primary(z[c]) * secondary(z[bottleneck+c])
                if active_channels.is_some() {
                    for f in 0..num_frames {
                        let z_off = f * z_rows;
                        for c in 0..active_bottleneck {
                            let primary = self.activation.apply_scalar_channel_fast(
                                self.z_buf.data[z_off + c],
                                c,
                                use_fast_tanh,
                            );
                            let gate = self.secondary_activation.apply_scalar_channel_fast(
                                self.z_buf.data[z_off + active_bottleneck + c],
                                c,
                                use_fast_tanh,
                            );
                            self.z_buf.data[z_off + c] = primary * gate;
                        }
                    }
                    zero_inactive_rows(&mut self.z_buf.data, z_rows, active_bottleneck, num_frames);
                } else {
                    #[cfg(feature = "fast-kernels")]
                    {
                        let p_id = self.activation.c_type_id();
                        let s_id = self.secondary_activation.c_type_id();
                        if let (Some(p), Some(s)) = (p_id, s_id) {
                            crate::fast_kernels::gated_activation(
                                &mut self.z_buf.data,
                                z_rows,
                                bottleneck,
                                num_frames,
                                p,
                                s,
                                use_fast_tanh,
                            );
                        } else {
                            // Fallback for PReLU/LeakyRelu with per-channel params
                            for f in 0..num_frames {
                                let z_off = f * z_rows;
                                for c in 0..bottleneck {
                                    let primary = self.activation.apply_scalar_channel_fast(
                                        self.z_buf.data[z_off + c],
                                        c,
                                        use_fast_tanh,
                                    );
                                    let gate = self.secondary_activation.apply_scalar_channel_fast(
                                        self.z_buf.data[z_off + bottleneck + c],
                                        c,
                                        use_fast_tanh,
                                    );
                                    self.z_buf.data[z_off + c] = primary * gate;
                                }
                            }
                        }
                    }
                    #[cfg(not(feature = "fast-kernels"))]
                    {
                        for f in 0..num_frames {
                            let z_off = f * z_rows;
                            for c in 0..bottleneck {
                                let primary = self.activation.apply_scalar_channel_fast(
                                    self.z_buf.data[z_off + c],
                                    c,
                                    use_fast_tanh,
                                );
                                let gate = self.secondary_activation.apply_scalar_channel_fast(
                                    self.z_buf.data[z_off + bottleneck + c],
                                    c,
                                    use_fast_tanh,
                                );
                                self.z_buf.data[z_off + c] = primary * gate;
                            }
                        }
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
                // Blending: z[c] = alpha * activated(z[c]) + (1-alpha) * z[c]
                if active_channels.is_some() {
                    for f in 0..num_frames {
                        let z_off = f * z_rows;
                        for c in 0..active_bottleneck {
                            let pre_act = self.z_buf.data[z_off + c];
                            let activated = self.activation.apply_scalar_channel_fast(
                                pre_act,
                                c,
                                use_fast_tanh,
                            );
                            let alpha = self.secondary_activation.apply_scalar_channel_fast(
                                self.z_buf.data[z_off + active_bottleneck + c],
                                c,
                                use_fast_tanh,
                            );
                            self.z_buf.data[z_off + c] =
                                alpha.mul_add(activated - pre_act, pre_act);
                        }
                    }
                    zero_inactive_rows(&mut self.z_buf.data, z_rows, active_bottleneck, num_frames);
                } else {
                    #[cfg(feature = "fast-kernels")]
                    {
                        let p_id = self.activation.c_type_id();
                        let s_id = self.secondary_activation.c_type_id();
                        if let (Some(p), Some(s)) = (p_id, s_id) {
                            crate::fast_kernels::blended_activation(
                                &mut self.z_buf.data,
                                z_rows,
                                bottleneck,
                                num_frames,
                                p,
                                s,
                                use_fast_tanh,
                            );
                        } else {
                            for f in 0..num_frames {
                                let z_off = f * z_rows;
                                for c in 0..bottleneck {
                                    let pre_act = self.z_buf.data[z_off + c];
                                    let activated = self.activation.apply_scalar_channel_fast(
                                        pre_act,
                                        c,
                                        use_fast_tanh,
                                    );
                                    let alpha =
                                        self.secondary_activation.apply_scalar_channel_fast(
                                            self.z_buf.data[z_off + bottleneck + c],
                                            c,
                                            use_fast_tanh,
                                        );
                                    self.z_buf.data[z_off + c] =
                                        alpha.mul_add(activated - pre_act, pre_act);
                                }
                            }
                        }
                    }
                    #[cfg(not(feature = "fast-kernels"))]
                    {
                        for f in 0..num_frames {
                            let z_off = f * z_rows;
                            for c in 0..bottleneck {
                                let pre_act = self.z_buf.data[z_off + c];
                                let activated = self.activation.apply_scalar_channel_fast(
                                    pre_act,
                                    c,
                                    use_fast_tanh,
                                );
                                let alpha = self.secondary_activation.apply_scalar_channel_fast(
                                    self.z_buf.data[z_off + bottleneck + c],
                                    c,
                                    use_fast_tanh,
                                );
                                self.z_buf.data[z_off + c] =
                                    alpha.mul_add(activated - pre_act, pre_act);
                            }
                        }
                    } // end cfg(not(fast-kernels))
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
        if active_channels.is_some() {
            if self.skip_head_copy {
                zero_inactive_rows(&mut self.z_buf.data, z_rows, active_bottleneck, num_frames);
            } else {
                zero_inactive_rows(
                    &mut self.output_head.data,
                    self.output_head.rows,
                    active_bottleneck.min(self.output_head.rows),
                    num_frames,
                );
            }
        }
        // If skip_head_copy, output_head is z itself (caller reads from z_buf)

        // Step 5: Output to next layer = input + layer1x1_output (or just input)
        let ch = self.channels;
        if let Some(ref l1x1) = self.layer1x1 {
            let total = ch * num_frames;
            #[cfg(feature = "fast-kernels")]
            crate::fast_kernels::vec_add(
                &mut self.output_next_layer.data,
                &input.data,
                &l1x1.output_buf.data,
                total,
            );
            #[cfg(not(feature = "fast-kernels"))]
            {
                let inp = &input.data;
                let l1 = &l1x1.output_buf.data;
                let out = &mut self.output_next_layer.data;
                let mut i = 0;
                while i + 3 < total {
                    out[i] = inp[i] + l1[i];
                    out[i + 1] = inp[i + 1] + l1[i + 1];
                    out[i + 2] = inp[i + 2] + l1[i + 2];
                    out[i + 3] = inp[i + 3] + l1[i + 3];
                    i += 4;
                }
                while i < total {
                    out[i] = inp[i] + l1[i];
                    i += 1;
                }
            }
        } else {
            let total = ch * num_frames;
            self.output_next_layer.data[..total].copy_from_slice(&input.data[..total]);
        }
        if let Some(active) = active_channels {
            zero_inactive_rows(
                &mut self.output_next_layer.data,
                ch,
                active.min(ch),
                num_frames,
            );
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

impl LayerFiLMParams {
    fn any_active(&self) -> bool {
        self.conv_pre.active
            || self.conv_post.active
            || self.input_mixin_pre.active
            || self.input_mixin_post.active
            || self.activation_pre.active
            || self.activation_post.active
            || self.layer1x1_post.active
            || self.head1x1_post.active
    }
}

#[derive(Debug, Clone)]
struct SlimmableConfig {
    allowed_channels: Vec<usize>,
    active_channels: usize,
}

impl SlimmableConfig {
    fn from_json(
        value: Option<&serde_json::Value>,
        channels: usize,
    ) -> Result<Option<Self>, NamError> {
        let Some(value) = value else {
            return Ok(None);
        };
        if value.is_null() {
            return Ok(None);
        }
        let method = value
            .get("method")
            .and_then(|v| v.as_str())
            .ok_or_else(|| NamError::MissingField("slimmable.method".into()))?;
        if method != "slice_channels_uniform" {
            return Err(NamError::UnsupportedConfigValue {
                field: "slimmable.method".into(),
                value: method.into(),
            });
        }
        let allowed = value
            .get("kwargs")
            .and_then(|v| v.get("allowed_channels"))
            .and_then(|v| v.as_array())
            .ok_or_else(|| NamError::MissingField("slimmable.kwargs.allowed_channels".into()))?;
        let allowed_channels = allowed
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let field = format!("slimmable.kwargs.allowed_channels[{index}]");
                let channel = positive_config_usize(value, &field)?;
                if channel == 0 || channel > channels {
                    return Err(NamError::ConfigValueOutOfRange {
                        field,
                        value: channel,
                        min: 1,
                        max: channels,
                    });
                }
                Ok(channel)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if allowed_channels.is_empty() {
            return Err(NamError::EmptyConfigArray {
                field: "slimmable.kwargs.allowed_channels".into(),
            });
        }
        for pair in allowed_channels.windows(2) {
            if pair[1] <= pair[0] {
                return Err(NamError::InvalidConfigField {
                    field: "slimmable.kwargs.allowed_channels".into(),
                    reason: "must be strictly increasing",
                });
            }
        }
        if allowed_channels[allowed_channels.len() - 1] != channels {
            return Err(NamError::InvalidConfigField {
                field: "slimmable.kwargs.allowed_channels".into(),
                reason: "the last entry must equal the full channel count",
            });
        }
        let active_channels = allowed_channels[allowed_channels.len() - 1];
        Ok(Some(Self {
            allowed_channels,
            active_channels,
        }))
    }

    fn set_slimming(&mut self, value: f64) -> Result<(), NamError> {
        if !value.is_finite() {
            return Err(NamError::InvalidConfigField {
                field: "slimming".into(),
                reason: "must be finite",
            });
        }
        let ratio = value.clamp(0.0, 1.0);
        let idx = ((ratio * self.allowed_channels.len() as f64).floor() as usize)
            .min(self.allowed_channels.len() - 1);
        self.active_channels = self.allowed_channels[idx];
        Ok(())
    }
}

// ── WaveNet LayerArray ──────────────────────────────────────────────────────

struct WaveNetLayerArray {
    rechannel: Conv1x1,
    layers: Vec<WaveNetLayer>,
    head_rechannel: Conv1d,
    channels: usize,
    head_output_size: usize, // head1x1.out_channels if active, else bottleneck
    slimmable: Option<SlimmableConfig>,

    // Pre-allocated block buffers
    layer_outputs: ColMajorMatrix,
    head_inputs: ColMajorMatrix,
}

impl WaveNetLayerArray {
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

    fn set_slimming(&mut self, value: f64) -> Result<bool, NamError> {
        let Some(ref mut slimmable) = self.slimmable else {
            return Ok(false);
        };
        slimmable.set_slimming(value)?;
        for layer in &mut self.layers {
            layer.zero_state();
        }
        self.head_rechannel.zero_state();
        Ok(true)
    }

    /// Process without previous head input (first layer array).
    /// Matches C++ _LayerArray::Process (2-arg version)
    fn process_first(
        &mut self,
        layer_inputs: &ColMajorMatrix,
        condition: &ColMajorMatrix,
        num_frames: usize,
        use_fast_tanh: bool,
    ) {
        // Zero head inputs accumulator
        self.head_inputs.zero_cols(num_frames);
        self.process_inner(layer_inputs, condition, num_frames, use_fast_tanh);
    }

    /// Process with previous head input (subsequent layer arrays).
    /// Matches C++ _LayerArray::Process (3-arg version)
    fn process_subsequent(
        &mut self,
        layer_inputs: &ColMajorMatrix,
        condition: &ColMajorMatrix,
        head_inputs: &ColMajorMatrix,
        num_frames: usize,
        use_fast_tanh: bool,
    ) {
        // Copy head inputs from previous layer array
        let len = self.head_output_size * num_frames;
        self.head_inputs.data[..len].copy_from_slice(&head_inputs.data[..len]);
        self.process_inner(layer_inputs, condition, num_frames, use_fast_tanh);
    }

    /// Common inner processing. Matches C++ _LayerArray::ProcessInner
    fn process_inner(
        &mut self,
        layer_inputs: &ColMajorMatrix,
        condition: &ColMajorMatrix,
        num_frames: usize,
        use_fast_tanh: bool,
    ) {
        // Rechannel: project input to layer channels
        self.rechannel.process_block(layer_inputs, num_frames);
        let active_channels = self
            .slimmable
            .as_ref()
            .map(|slimmable| slimmable.active_channels);
        if let Some(active) = active_channels {
            zero_inactive_rows(
                &mut self.rechannel.output_buf.data,
                self.rechannel.out_channels,
                active.min(self.rechannel.out_channels),
                num_frames,
            );
        }

        // Process layers
        let num_layers = self.layers.len();

        for i in 0..num_layers {
            if i == 0 {
                let layer = &mut self.layers[0];
                layer.process_block(
                    &self.rechannel.output_buf,
                    condition,
                    num_frames,
                    active_channels,
                    use_fast_tanh,
                );
            } else {
                let (processed, pending) = self.layers.split_at_mut(i);
                let prev_output = &processed[i - 1].output_next_layer;
                let layer = &mut pending[0];
                layer.process_block(
                    prev_output,
                    condition,
                    num_frames,
                    active_channels,
                    use_fast_tanh,
                );
            }

            // Accumulate head output from this layer (4-wide unrolled)
            let head_out_size = self.head_output_size;
            let layer = &self.layers[i];
            let head_data = layer.get_output_head_data();
            let head_rows = layer.get_output_head_rows();
            if head_rows == head_out_size {
                // Contiguous: single pass
                let total = head_out_size * num_frames;
                #[cfg(feature = "fast-kernels")]
                crate::fast_kernels::vec_add_inplace(&mut self.head_inputs.data, head_data, total);
                #[cfg(not(feature = "fast-kernels"))]
                {
                    let dst = &mut self.head_inputs.data;
                    let mut j = 0;
                    while j + 3 < total {
                        dst[j] += head_data[j];
                        dst[j + 1] += head_data[j + 1];
                        dst[j + 2] += head_data[j + 2];
                        dst[j + 3] += head_data[j + 3];
                        j += 4;
                    }
                    while j < total {
                        dst[j] += head_data[j];
                        j += 1;
                    }
                } // end cfg(not(fast-kernels))
            } else {
                // Different strides: per-frame copy
                for f in 0..num_frames {
                    let src_off = f * head_rows;
                    let dst_off = f * head_out_size;
                    for c in 0..head_out_size {
                        self.head_inputs.data[dst_off + c] += head_data[src_off + c];
                    }
                }
            }
        }

        // Store output from last layer
        let last = num_layers - 1;
        let ch = self.channels;
        let len = ch * num_frames;
        self.layer_outputs.data[..len]
            .copy_from_slice(&self.layers[last].output_next_layer.data[..len]);
        if let Some(active) = active_channels {
            zero_inactive_rows(
                &mut self.head_inputs.data,
                self.head_inputs.rows,
                active.min(self.head_inputs.rows),
                num_frames,
            );
        }

        // Head rechannel
        self.head_rechannel
            .process_block(&self.head_inputs, num_frames);
    }
}

// ── Top-level WaveNet head ─────────────────────────────────────────────────

struct WaveNetHeadBlock {
    activation: Activation,
    conv: Conv1d,
    activation_buf: ColMajorMatrix,
}

impl WaveNetHeadBlock {
    fn from_weights(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        activation: &Activation,
        iter: &mut WeightIter,
    ) -> Result<Self, NamError> {
        Ok(Self {
            activation: activation.clone(),
            conv: Conv1d::from_weights(in_channels, out_channels, kernel_size, 1, 1, iter)?,
            activation_buf: ColMajorMatrix::new(in_channels, 1),
        })
    }

    fn receptive_field(&self) -> usize {
        self.conv.receptive_field()
    }

    fn set_max_buffer_size(&mut self, max_buffer_size: usize) {
        self.activation_buf
            .resize(self.conv.in_channels, max_buffer_size);
        self.conv.set_max_buffer_size(max_buffer_size);
    }

    fn process_block(&mut self, input: &ColMajorMatrix, num_frames: usize, use_fast_tanh: bool) {
        let rows = self.conv.in_channels;
        let len = rows * num_frames;
        self.activation_buf.data[..len].copy_from_slice(&input.data[..len]);
        self.activation.apply_colmajor_inplace(
            &mut self.activation_buf.data,
            rows,
            num_frames,
            use_fast_tanh,
        );
        self.conv.process_block(&self.activation_buf, num_frames);
    }
}

struct WaveNetHead {
    blocks: Vec<WaveNetHeadBlock>,
    input_buf: ColMajorMatrix,
}

impl WaveNetHead {
    fn from_config(
        config: &serde_json::Value,
        in_channels: usize,
        iter: &mut WeightIter,
    ) -> Result<Self, NamError> {
        let channels = positive_config_usize(&config["channels"], "head.channels")?;
        let out_channels = positive_config_usize(&config["out_channels"], "head.out_channels")?;
        let activation = Activation::from_json(&config["activation"])?;
        let kernel_sizes = config["kernel_sizes"]
            .as_array()
            .ok_or_else(|| NamError::MissingField("head.kernel_sizes".into()))?;
        if kernel_sizes.is_empty() {
            return Err(NamError::EmptyConfigArray {
                field: "head.kernel_sizes".into(),
            });
        }

        let mut blocks = Vec::with_capacity(kernel_sizes.len());
        let mut block_in_channels = in_channels;
        for (idx, kernel_size) in kernel_sizes.iter().enumerate() {
            let kernel_size =
                positive_config_usize(kernel_size, &format!("head.kernel_sizes[{idx}]"))?;
            let block_out_channels = if idx + 1 == kernel_sizes.len() {
                out_channels
            } else {
                channels
            };
            blocks.push(WaveNetHeadBlock::from_weights(
                block_in_channels,
                block_out_channels,
                kernel_size,
                &activation,
                iter,
            )?);
            block_in_channels = channels;
        }

        Ok(Self {
            blocks,
            input_buf: ColMajorMatrix::new(in_channels, 1),
        })
    }

    fn out_channels(&self) -> usize {
        self.blocks
            .last()
            .map(|block| block.conv.out_channels)
            .unwrap_or(1)
    }

    fn set_max_buffer_size(&mut self, max_buffer_size: usize) {
        let rows = self.input_buf.rows;
        self.input_buf.resize(rows, max_buffer_size);
        for block in &mut self.blocks {
            block.set_max_buffer_size(max_buffer_size);
        }
    }

    fn process_block(
        &mut self,
        input: &ColMajorMatrix,
        input_rows: usize,
        scale: f32,
        num_frames: usize,
        use_fast_tanh: bool,
    ) {
        self.input_buf.resize(input_rows, num_frames);
        let len = input_rows * num_frames;
        for (dst, src) in self.input_buf.data[..len]
            .iter_mut()
            .zip(input.data[..len].iter())
        {
            *dst = scale * *src;
        }

        for index in 0..self.blocks.len() {
            if index == 0 {
                self.blocks[0].process_block(&self.input_buf, num_frames, use_fast_tanh);
            } else {
                let (processed, pending) = self.blocks.split_at_mut(index);
                let input = &processed[index - 1].conv.output_buf;
                pending[0].process_block(input, num_frames, use_fast_tanh);
            }
        }
    }

    fn output(&self) -> Option<&ColMajorMatrix> {
        self.blocks.last().map(|block| &block.conv.output_buf)
    }
}

// ── Top-level WaveNet ───────────────────────────────────────────────────────

pub struct WaveNet {
    layer_arrays: Vec<WaveNetLayerArray>,
    head: Option<WaveNetHead>,
    head_scale: f32,
    prewarm_samples_count: usize,
    metadata: DspMetadata,
    in_channels: usize,

    // Optional condition DSP
    condition_dsp: Option<Box<dyn Dsp>>,

    // Block processing buffers
    condition_input: ColMajorMatrix,
    condition_output: ColMajorMatrix,
    multi_channel_scratch: Vec<Sample>,
    max_buffer_size: usize,
    activation_mode: ActivationMode,
}

fn normalize_wavenet_config(config: &serde_json::Value) -> Result<serde_json::Value, NamError> {
    let mut normalized = config.clone();
    let Some(root) = normalized.as_object_mut() else {
        return Err(NamError::InvalidConfigType {
            field: "config".into(),
            expected: "a JSON object",
        });
    };

    if !root.contains_key("layers") {
        if let Some(layers_configs) = root.get("layers_configs").cloned() {
            root.insert("layers".into(), layers_configs);
        }
    }

    let Some(layers) = root
        .get_mut("layers")
        .and_then(|layers| layers.as_array_mut())
    else {
        return Ok(normalized);
    };

    for layer in layers {
        normalize_layer_array_config(layer)?;
    }

    Ok(normalized)
}

fn normalize_layer_array_config(layer: &mut serde_json::Value) -> Result<(), NamError> {
    let Some(layer_obj) = layer.as_object_mut() else {
        return Err(NamError::InvalidConfigType {
            field: "layers[]".into(),
            expected: "a JSON object",
        });
    };

    if let Some(head) = layer_obj.get("head").and_then(|head| head.as_object()) {
        let out_channels = head.get("out_channels").cloned();
        let bias = head.get("bias").cloned();
        let kernel_size = head.get("kernel_size").cloned();
        let head_dilation = head.get("head_dilation").cloned();
        if !layer_obj.contains_key("head_size") {
            if let Some(out_channels) = out_channels {
                layer_obj.insert("head_size".into(), out_channels);
            }
        }
        if !layer_obj.contains_key("head_bias") {
            if let Some(bias) = bias {
                layer_obj.insert("head_bias".into(), bias);
            }
        }
        if !layer_obj.contains_key("head_kernel_size") {
            if let Some(kernel_size) = kernel_size {
                layer_obj.insert("head_kernel_size".into(), kernel_size);
            }
        }
        if !layer_obj.contains_key("head_dilation") {
            if let Some(head_dilation) = head_dilation {
                layer_obj.insert("head_dilation".into(), head_dilation);
            }
        }
    }

    if !layer_obj.contains_key("head1x1") {
        if let Some(head1x1) = layer_obj.get("head_1x1_config").cloned() {
            layer_obj.insert("head1x1".into(), head1x1);
        }
    }
    if !layer_obj.contains_key("layer1x1") {
        if let Some(layer1x1) = layer_obj.get("layer_1x1_config").cloned() {
            layer_obj.insert("layer1x1".into(), layer1x1);
        }
    }

    if let Some(film_params) = layer_obj.get("film_params").and_then(|v| v.as_object()) {
        let entries: Vec<(String, serde_json::Value)> = film_params
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        for (key, value) in entries {
            layer_obj.entry(key).or_insert(value);
        }
    }

    normalize_pairing_activation_config(layer_obj)?;

    Ok(())
}

fn normalize_pairing_activation_config(
    layer_obj: &mut serde_json::Map<String, serde_json::Value>,
) -> Result<(), NamError> {
    if layer_obj.contains_key("gating_mode") {
        return Ok(());
    }

    let Some(activation) = layer_obj.get("activation").cloned() else {
        return Ok(());
    };

    if let Some(activation_array) = activation.as_array() {
        let mut primary = Vec::with_capacity(activation_array.len());
        let mut gating_modes = Vec::with_capacity(activation_array.len());
        let mut secondary = Vec::with_capacity(activation_array.len());
        let mut saw_pairing = false;

        for entry in activation_array {
            let normalized = normalize_one_activation_entry(entry)?;
            saw_pairing |= normalized.gating_mode != "none";
            primary.push(normalized.primary);
            gating_modes.push(serde_json::Value::String(normalized.gating_mode));
            secondary.push(normalized.secondary.unwrap_or(serde_json::Value::Null));
        }

        if saw_pairing {
            layer_obj.insert("activation".into(), serde_json::Value::Array(primary));
            layer_obj.insert("gating_mode".into(), serde_json::Value::Array(gating_modes));
            layer_obj.insert(
                "secondary_activation".into(),
                serde_json::Value::Array(secondary),
            );
        }
    } else {
        let normalized = normalize_one_activation_entry(&activation)?;
        if normalized.gating_mode != "none" {
            layer_obj.insert("activation".into(), normalized.primary);
            layer_obj.insert(
                "gating_mode".into(),
                serde_json::Value::String(normalized.gating_mode),
            );
            if let Some(secondary) = normalized.secondary {
                layer_obj.insert("secondary_activation".into(), secondary);
            }
        }
    }

    Ok(())
}

struct NormalizedActivationEntry {
    primary: serde_json::Value,
    gating_mode: String,
    secondary: Option<serde_json::Value>,
}

fn normalize_one_activation_entry(
    activation: &serde_json::Value,
) -> Result<NormalizedActivationEntry, NamError> {
    let Some(obj) = activation.as_object() else {
        return Ok(NormalizedActivationEntry {
            primary: activation.clone(),
            gating_mode: "none".into(),
            secondary: None,
        });
    };

    let name = obj
        .get("name")
        .or_else(|| obj.get("type"))
        .and_then(|v| v.as_str());
    let Some(name) = name else {
        return Ok(NormalizedActivationEntry {
            primary: activation.clone(),
            gating_mode: "none".into(),
            secondary: None,
        });
    };

    let gating_mode = match name {
        "PairMultiply" => "gated",
        "PairBlend" => "blended",
        _ => {
            return Ok(NormalizedActivationEntry {
                primary: activation.clone(),
                gating_mode: "none".into(),
                secondary: None,
            });
        }
    };

    let primary = obj
        .get("primary")
        .cloned()
        .ok_or_else(|| NamError::MissingField("activation.primary".into()))?;
    let secondary = obj
        .get("secondary")
        .cloned()
        .ok_or_else(|| NamError::MissingField("activation.secondary".into()))?;

    Ok(NormalizedActivationEntry {
        primary,
        gating_mode: gating_mode.into(),
        secondary: Some(secondary),
    })
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
        let normalized_config = normalize_wavenet_config(config)?;
        let config = &normalized_config;
        let layers_json = config["layers"]
            .as_array()
            .ok_or_else(|| NamError::MissingField("layers".into()))?;

        let mut iter = WeightIter::new(weights);
        let mut layer_arrays = Vec::new();
        let mut condition_size = 0usize;

        let in_channels = config
            .get("in_channels")
            .map(|value| positive_config_usize(value, "in_channels"))
            .transpose()?
            .unwrap_or(1);

        for (la_idx, la_json) in layers_json.iter().enumerate() {
            let input_size =
                positive_config_usize(&la_json["input_size"], "layer_array.input_size")?;
            let cond_size = config_usize(&la_json["condition_size"], "layer_array.condition_size")?;
            let head_size = positive_config_usize(&la_json["head_size"], "layer_array.head_size")?;
            let channels = positive_config_usize(&la_json["channels"], "layer_array.channels")?;
            let bottleneck = la_json
                .get("bottleneck")
                .map(|value| positive_config_usize(value, "layer_array.bottleneck"))
                .transpose()?
                .unwrap_or(channels);
            let dilations_arr = la_json["dilations"]
                .as_array()
                .ok_or_else(|| NamError::MissingField("layer_array.dilations".into()))?;
            let dilations: Vec<usize> = dilations_arr
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    config_usize(value, &format!("layer_array.dilations[{index}]"))
                })
                .collect::<Result<Vec<_>, _>>()?;

            let num_layers = dilations.len();

            // Parse kernel sizes: support legacy single `kernel_size` (int) or
            // new per-layer `kernel_sizes` (array). Mutual exclusivity enforced.
            let kernel_sizes: Vec<usize> = {
                let has_kernel_size = la_json.get("kernel_size").is_some();
                let has_kernel_sizes = la_json.get("kernel_sizes").is_some();

                if has_kernel_size && has_kernel_sizes {
                    return Err(NamError::ConflictingConfigFields {
                        first: format!("layers[{la_idx}].kernel_size"),
                        second: format!("layers[{la_idx}].kernel_sizes"),
                    });
                } else if has_kernel_sizes {
                    let arr = la_json["kernel_sizes"].as_array().ok_or_else(|| {
                        NamError::InvalidConfigType {
                            field: format!("layers[{la_idx}].kernel_sizes"),
                            expected: "an array",
                        }
                    })?;
                    let ks: Vec<usize> = arr
                        .iter()
                        .enumerate()
                        .map(|(index, value)| {
                            positive_config_usize(
                                value,
                                &format!("layers[{la_idx}].kernel_sizes[{index}]"),
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    if ks.len() != num_layers {
                        return Err(NamError::ConfigLengthMismatch {
                            field: format!("layers[{la_idx}].kernel_sizes"),
                            actual: ks.len(),
                            expected_field: format!("layers[{la_idx}].dilations"),
                            expected: num_layers,
                        });
                    }
                    ks
                } else if has_kernel_size {
                    let ks_val = &la_json["kernel_size"];
                    if let Some(arr) = ks_val.as_array() {
                        // Also accept kernel_size as an array (trainer compat)
                        let ks: Vec<usize> = arr
                            .iter()
                            .enumerate()
                            .map(|(index, value)| {
                                positive_config_usize(
                                    value,
                                    &format!("layers[{la_idx}].kernel_size[{index}]"),
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        if ks.len() != num_layers {
                            return Err(NamError::ConfigLengthMismatch {
                                field: format!("layers[{la_idx}].kernel_size"),
                                actual: ks.len(),
                                expected_field: format!("layers[{la_idx}].dilations"),
                                expected: num_layers,
                            });
                        }
                        ks
                    } else {
                        let ks = positive_config_usize(
                            ks_val,
                            &format!("layers[{la_idx}].kernel_size"),
                        )?;
                        vec![ks; num_layers]
                    }
                } else {
                    return Err(NamError::MissingField(
                        "layer_array: either kernel_size or kernel_sizes must be provided".into(),
                    ));
                }
            };
            for (&kernel_size, &dilation) in kernel_sizes.iter().zip(&dilations) {
                checked_dimension_mul("convolution receptive field", dilation, kernel_size - 1)?;
            }

            // Parse activation configs (per-layer or single)
            let activation_configs: Vec<Activation> = {
                let act_val = &la_json["activation"];
                if let Some(arr) = act_val.as_array() {
                    arr.iter()
                        .map(Activation::from_json)
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    let act = Activation::from_json(act_val).unwrap_or(Activation::Tanh);
                    vec![act; num_layers]
                }
            };

            // Parse gating modes (per-layer or single or old bool)
            let (gating_modes, secondary_activations) =
                parse_gating_and_secondary(la_json, num_layers)?;

            let head_bias = la_json["head_bias"].as_bool().unwrap_or(false);
            let head_kernel_size = la_json
                .get("head_kernel_size")
                .map(|value| positive_config_usize(value, "layer_array.head_kernel_size"))
                .transpose()?
                .unwrap_or(1);
            let head_dilation = la_json
                .get("head_dilation")
                .map(|value| positive_config_usize(value, "layer_array.head_dilation"))
                .transpose()?
                .unwrap_or(1);
            checked_dimension_mul(
                "convolution receptive field",
                head_dilation,
                head_kernel_size - 1,
            )?;
            let slimmable = SlimmableConfig::from_json(la_json.get("slimmable"), channels)?;

            // Groups
            let groups_input = la_json
                .get("groups_input")
                .map(|value| positive_config_usize(value, "layer_array.groups_input"))
                .transpose()?
                .unwrap_or(1);
            let groups_input_mixin = la_json
                .get("groups_input_mixin")
                .map(|value| positive_config_usize(value, "layer_array.groups_input_mixin"))
                .transpose()?
                .unwrap_or(1);

            // Layer1x1 config
            let (has_layer1x1, layer1x1_groups) = if let Some(l1x1) = la_json.get("layer1x1") {
                let active = l1x1.get("active").and_then(|v| v.as_bool()).unwrap_or(true);
                let groups = l1x1
                    .get("groups")
                    .map(|value| positive_config_usize(value, "layer_array.layer1x1.groups"))
                    .transpose()?
                    .unwrap_or(1);
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
                    .map(|value| positive_config_usize(value, "layer_array.head1x1.out_channels"))
                    .transpose()?
                    .unwrap_or(channels);
                let groups = h1x1
                    .get("groups")
                    .map(|value| positive_config_usize(value, "layer_array.head1x1.groups"))
                    .transpose()?
                    .unwrap_or(1);
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
                    .map(|value| FiLMParams::from_json(value, "layer_array.conv_pre_film"))
                    .transpose()?
                    .unwrap_or_else(FiLMParams::inactive),
                conv_post: la_json
                    .get("conv_post_film")
                    .map(|value| FiLMParams::from_json(value, "layer_array.conv_post_film"))
                    .transpose()?
                    .unwrap_or_else(FiLMParams::inactive),
                input_mixin_pre: la_json
                    .get("input_mixin_pre_film")
                    .map(|value| FiLMParams::from_json(value, "layer_array.input_mixin_pre_film"))
                    .transpose()?
                    .unwrap_or_else(FiLMParams::inactive),
                input_mixin_post: la_json
                    .get("input_mixin_post_film")
                    .map(|value| FiLMParams::from_json(value, "layer_array.input_mixin_post_film"))
                    .transpose()?
                    .unwrap_or_else(FiLMParams::inactive),
                activation_pre: la_json
                    .get("activation_pre_film")
                    .map(|value| FiLMParams::from_json(value, "layer_array.activation_pre_film"))
                    .transpose()?
                    .unwrap_or_else(FiLMParams::inactive),
                activation_post: la_json
                    .get("activation_post_film")
                    .map(|value| FiLMParams::from_json(value, "layer_array.activation_post_film"))
                    .transpose()?
                    .unwrap_or_else(FiLMParams::inactive),
                layer1x1_post: la_json
                    .get("layer1x1_post_film")
                    .map(|value| FiLMParams::from_json(value, "layer_array.layer1x1_post_film"))
                    .transpose()?
                    .unwrap_or_else(FiLMParams::inactive),
                head1x1_post: la_json
                    .get("head1x1_post_film")
                    .map(|value| FiLMParams::from_json(value, "layer_array.head1x1_post_film"))
                    .transpose()?
                    .unwrap_or_else(FiLMParams::inactive),
            };

            if slimmable.is_some() {
                if condition_dsp.is_some() {
                    return Err(NamError::InvalidConfigField {
                        field: "condition_dsp".into(),
                        reason: "is incompatible with slimmable layers",
                    });
                }
                if layers_json.len() != 1 {
                    return Err(NamError::InvalidConfigField {
                        field: "layers".into(),
                        reason: "must contain exactly one array for a slimmable WaveNet",
                    });
                }
                if groups_input != 1 || groups_input_mixin != 1 {
                    return Err(NamError::InvalidConfigField {
                        field: "groups_input".into(),
                        reason: "grouped convolutions are incompatible with slimmable layers",
                    });
                }
                if head1x1_params.active {
                    return Err(NamError::InvalidConfigField {
                        field: "head1x1.active".into(),
                        reason: "must be false for slimmable layers",
                    });
                }
                if layer1x1_groups != 1 {
                    return Err(NamError::InvalidConfigField {
                        field: "layer1x1.groups".into(),
                        reason: "must be one for slimmable layers",
                    });
                }
                if film_params.any_active() {
                    return Err(NamError::InvalidConfigField {
                        field: "film".into(),
                        reason: "must be inactive for slimmable layers",
                    });
                }
                if head_kernel_size != 1 || head_dilation != 1 {
                    return Err(NamError::InvalidConfigField {
                        field: "head".into(),
                        reason: "kernel size and dilation must be one for slimmable layers",
                    });
                }
            }

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
                    kernel_sizes[layer_idx],
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

            let head_rechannel = Conv1d::from_weights_with_bias(
                head_out_size,
                head_size,
                head_kernel_size,
                head_dilation,
                1,
                head_bias,
                &mut iter,
            )?;

            layer_arrays.push(WaveNetLayerArray {
                rechannel,
                layers,
                head_rechannel,
                channels,
                head_output_size: head_out_size,
                slimmable,
                layer_outputs: ColMajorMatrix::new(channels, 1),
                head_inputs: ColMajorMatrix::new(head_out_size, 1),
            });
        }

        let head_in_channels = layer_arrays
            .last()
            .map(|la| la.head_rechannel.out_channels)
            .ok_or_else(|| NamError::EmptyConfigArray {
                field: "layers".into(),
            })?;
        let head = if config.get("head").is_some_and(|head| !head.is_null()) {
            Some(WaveNetHead::from_config(
                &config["head"],
                head_in_channels,
                &mut iter,
            )?)
        } else {
            None
        };

        let head_scale_from_weights = iter.take(1)?[0];
        let head_scale = head_scale_from_weights;

        iter.assert_exhausted()?;

        // Compute prewarm
        let condition_prewarm = condition_dsp
            .as_ref()
            .map(|d| d.prewarm_samples())
            .unwrap_or(1);
        let layer_prewarm = layer_arrays.iter().try_fold(0usize, |total, layer_array| {
            let layer_total = layer_array.layers.iter().try_fold(
                layer_array.head_rechannel.receptive_field(),
                |layer_total, layer| {
                    checked_dimension_add(
                        "WaveNet layer receptive field",
                        layer_total,
                        layer.conv.receptive_field(),
                    )
                },
            )?;
            checked_dimension_add("WaveNet receptive field", total, layer_total)
        })?;
        let head_prewarm = head
            .as_ref()
            .map(|head| {
                head.blocks.iter().try_fold(0usize, |total, block| {
                    checked_dimension_add(
                        "WaveNet head receptive field",
                        total,
                        block.receptive_field(),
                    )
                })
            })
            .transpose()?
            .unwrap_or(0);
        let prewarm_samples_count =
            checked_dimension_add("WaveNet prewarm", condition_prewarm, layer_prewarm)
                .and_then(|total| checked_dimension_add("WaveNet prewarm", total, head_prewarm))?;

        // Determine condition output channels
        let cond_out_ch = if let Some(ref cdsp) = condition_dsp {
            cdsp.num_output_channels()
        } else {
            in_channels
        };

        #[cfg(feature = "fast-kernels")]
        let _ = crate::fast_kernels::init_vector_math();

        Ok(Self {
            layer_arrays,
            head,
            head_scale,
            prewarm_samples_count,
            metadata,
            in_channels,
            condition_dsp,
            condition_input: ColMajorMatrix::new(in_channels, 1),
            condition_output: ColMajorMatrix::new(cond_out_ch.max(condition_size), 1),
            multi_channel_scratch: Vec::new(),
            max_buffer_size: 0,
            activation_mode: ActivationMode::Accurate,
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
        self.multi_channel_scratch
            .resize(max_buffer_size, Sample::default());

        for la in &mut self.layer_arrays {
            la.set_max_buffer_size(max_buffer_size);
        }
        if let Some(ref mut head) = self.head {
            head.set_max_buffer_size(max_buffer_size);
        }
    }

    pub fn set_slimming(&mut self, value: f64) -> Result<(), NamError> {
        let mut found = false;
        for la in &mut self.layer_arrays {
            found |= la.set_slimming(value)?;
        }
        if found {
            Ok(())
        } else {
            Err(NamError::UnsupportedOperation {
                operation: "slimmable width selection on a non-slimmable WaveNet",
            })
        }
    }

    pub fn slimming_breakpoints(&self) -> Vec<f64> {
        let mut breakpoints = self
            .layer_arrays
            .iter()
            .filter_map(|layer_array| layer_array.slimmable.as_ref())
            .flat_map(|slimmable| {
                (1..slimmable.allowed_channels.len())
                    .map(|index| index as f64 / slimmable.allowed_channels.len() as f64)
            })
            .collect::<Vec<_>>();
        breakpoints.sort_by(f64::total_cmp);
        breakpoints.dedup();
        breakpoints
    }

    /// Block processing matching C++ WaveNet::process
    fn process_block(&mut self, input: &[Sample], output: &mut [Sample]) {
        let num_frames = input.len();
        if num_frames == 0 {
            return;
        }

        self.ensure_buffer_size(num_frames);

        // Step 1: Fill condition_input (in_channels x num_frames)
        // For standard NAM, in_channels = 1, so each column has one element.
        // Buffer is pre-zeroed by resize(), so only write the first channel.
        let in_ch = self.in_channels;
        for (f, &sample) in input.iter().enumerate().take(num_frames) {
            self.condition_input.data[f * in_ch] = crate::dsp::sample_to_f32(sample);
        }

        // Step 2: Process condition
        if let Some(ref mut cdsp) = self.condition_dsp {
            // Process condition_dsp as a block (not per-sample) for efficiency
            let cond_out_ch = cdsp.num_output_channels();
            let cond_rows = self.condition_output.rows;

            cdsp.process_block_multi_channel(
                input,
                &mut self.condition_output.data,
                cond_rows,
                cond_out_ch,
                num_frames,
            );
        } else {
            // No condition DSP: condition_output = condition_input
            let cond_rows = self.condition_output.rows;
            let in_rows = self.condition_input.rows;
            let copy_rows = cond_rows.min(in_rows);
            if cond_rows == in_rows {
                // Same stride: single bulk copy
                let len = copy_rows * num_frames;
                self.condition_output.data[..len]
                    .copy_from_slice(&self.condition_input.data[..len]);
            } else {
                for f in 0..num_frames {
                    let cond_off = f * cond_rows;
                    let in_off = f * in_rows;
                    self.condition_output.data[cond_off..cond_off + copy_rows]
                        .copy_from_slice(&self.condition_input.data[in_off..in_off + copy_rows]);
                }
            }
        }

        // Step 3: Process layer arrays
        let num_arrays = self.layer_arrays.len();

        for arr_idx in 0..num_arrays {
            if arr_idx == 0 {
                // First layer array: use condition_input as layer_inputs, condition_output as condition
                self.layer_arrays[arr_idx].process_first(
                    &self.condition_input,
                    &self.condition_output,
                    num_frames,
                    self.activation_mode.use_fast_tanh(),
                );
            } else {
                // Subsequent: use previous array's layer_outputs and head_outputs
                let (processed, pending) = self.layer_arrays.split_at_mut(arr_idx);
                let previous = &processed[arr_idx - 1];
                pending[0].process_subsequent(
                    &previous.layer_outputs,
                    &self.condition_output,
                    &previous.head_rechannel.output_buf,
                    num_frames,
                    self.activation_mode.use_fast_tanh(),
                );
            }
        }

        // Step 4: Extract output from final layer-array head or optional top-level head
        let last = num_arrays - 1;
        let layer_head = &self.layer_arrays[last].head_rechannel.output_buf;
        let (final_head, out_ch, output_scale) = if let Some(ref mut head) = self.head {
            let layer_out_ch = self.layer_arrays[last].head_rechannel.out_channels;
            head.process_block(
                layer_head,
                layer_out_ch,
                self.head_scale,
                num_frames,
                self.activation_mode.use_fast_tanh(),
            );
            let Some(head_output) = head.output() else {
                output.fill(Sample::default());
                return;
            };
            (head_output, head.out_channels(), 1.0)
        } else {
            (
                layer_head,
                self.layer_arrays[last].head_rechannel.out_channels,
                self.head_scale,
            )
        };

        // For single-channel output (typical NAM): data is contiguous
        if out_ch == 1 {
            let scale = output_scale;
            for (o, &h) in output
                .iter_mut()
                .zip(final_head.data.iter())
                .take(num_frames)
            {
                *o = (scale * h) as Sample;
            }
        } else {
            // Multi-channel: take first channel
            let scale = output_scale;
            for (s, o) in output.iter_mut().enumerate().take(num_frames) {
                *o = (scale * final_head.data[s * out_ch]) as Sample;
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
                self.layer_arrays[arr_idx].process_first(
                    &self.condition_input,
                    &self.condition_output,
                    1,
                    self.activation_mode.use_fast_tanh(),
                );
            } else {
                let (processed, pending) = self.layer_arrays.split_at_mut(arr_idx);
                let previous = &processed[arr_idx - 1];
                pending[0].process_subsequent(
                    &previous.layer_outputs,
                    &self.condition_output,
                    &previous.head_rechannel.output_buf,
                    1,
                    self.activation_mode.use_fast_tanh(),
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
        if let Some(ref head) = self.head {
            head.out_channels()
        } else {
            self.layer_arrays
                .last()
                .map(|la| la.head_rechannel.out_channels)
                .unwrap_or(1)
        }
    }

    fn process_sample_multi_channel(&mut self, input_sample: Sample, out: &mut [f32]) {
        self.process_sample_for_multi_channel(crate::dsp::sample_to_f32(input_sample));

        let last = self.layer_arrays.len() - 1;
        let layer_head = &self.layer_arrays[last].head_rechannel.output_buf;
        let (final_head, out_ch, scale) = if let Some(ref mut head) = self.head {
            let layer_out_ch = self.layer_arrays[last].head_rechannel.out_channels;
            head.process_block(
                layer_head,
                layer_out_ch,
                self.head_scale,
                1,
                self.activation_mode.use_fast_tanh(),
            );
            let Some(head_output) = head.output() else {
                for o in out {
                    *o = 0.0;
                }
                return;
            };
            (head_output, head.out_channels(), 1.0)
        } else {
            (
                layer_head,
                self.layer_arrays[last].head_rechannel.out_channels,
                self.head_scale,
            )
        };

        for (i, o) in out.iter_mut().enumerate() {
            if i < out_ch {
                *o = scale * final_head.data[i];
            }
        }
    }

    fn process_block_multi_channel(
        &mut self,
        input: &[Sample],
        output_data: &mut [f32],
        output_stride: usize,
        out_channels: usize,
        num_frames: usize,
    ) {
        self.ensure_buffer_size(num_frames);
        let mut scratch = std::mem::take(&mut self.multi_channel_scratch);
        self.process_block(input, &mut scratch[..num_frames]);
        self.multi_channel_scratch = scratch;

        // Extract multi-channel head output from the last layer array
        let last = self.layer_arrays.len() - 1;
        let layer_head = &self.layer_arrays[last].head_rechannel.output_buf;
        let (final_head, head_ch, scale) = if let Some(ref head) = self.head {
            let Some(head_output) = head.output() else {
                output_data.fill(0.0);
                return;
            };
            (head_output, head.out_channels(), 1.0)
        } else {
            (
                layer_head,
                self.layer_arrays[last].head_rechannel.out_channels,
                self.head_scale,
            )
        };
        let copy_ch = out_channels.min(head_ch);

        for f in 0..num_frames {
            let out_off = f * output_stride;
            let head_off = f * head_ch;
            for c in 0..copy_ch {
                output_data[out_off + c] = scale * final_head.data[head_off + c];
            }
        }
    }

    fn set_slimming(&mut self, value: f64) -> Result<(), NamError> {
        WaveNet::set_slimming(self, value)
    }

    fn slimming_breakpoints(&self) -> Vec<f64> {
        WaveNet::slimming_breakpoints(self)
    }

    fn set_max_buffer_size(&mut self, max_buffer_size: usize) {
        self.set_max_buffer_size_internal(max_buffer_size);
    }

    fn prewarm_samples(&self) -> usize {
        self.prewarm_samples_count
    }

    fn metadata(&self) -> &DspMetadata {
        &self.metadata
    }

    fn set_activation_mode(&mut self, mode: ActivationMode) {
        self.activation_mode = mode;
        if let Some(condition_dsp) = self.condition_dsp.as_mut() {
            condition_dsp.set_activation_mode(mode);
        }
    }
}

// ── Activation helper for column-major block processing ─────────────────────

impl Activation {
    /// Apply activation in-place to a column-major matrix.
    /// The matrix has `rows` per column and `num_cols` columns.
    /// Applies per-channel (row) for PReLU.
    fn apply_colmajor_inplace(
        &self,
        data: &mut [f32],
        rows: usize,
        num_cols: usize,
        use_fast_tanh: bool,
    ) {
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
                #[cfg(feature = "fast-kernels")]
                if let Some(act_id) = self.c_type_id() {
                    crate::fast_kernels::activation_inplace(data, len, act_id, use_fast_tanh);
                    return;
                }
                for x in data[..len].iter_mut() {
                    *x = self.apply_scalar_fast(*x, use_fast_tanh);
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
            other => Err(NamError::UnsupportedConfigValue {
                field: "gating_mode".into(),
                value: other.into(),
            }),
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
                let mode_str =
                    gm_json
                        .as_str()
                        .ok_or_else(|| NamError::InvalidConfigArrayElement {
                            field: "gating_mode".into(),
                            index: idx,
                            expected: "a string",
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
mod tests;
