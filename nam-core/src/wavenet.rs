use crate::activations::Activation;
use crate::dsp::{ActivationMode, Dsp, DspMetadata, Sample};
use crate::error::NamError;
use crate::util::WeightIter;

mod matrix_backend;

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

    /// Small-matrix GEMM with fused bias for common channel counts.
    /// Specializes for 3x3, 1→3, 3→1 to allow the compiler to fully unroll.
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
        match (out_ch, in_ch) {
            (3, 3) => {
                let w00 = w[0];
                let w10 = w[1];
                let w20 = w[2];
                let w01 = w[3];
                let w11 = w[4];
                let w21 = w[5];
                let w02 = w[6];
                let w12 = w[7];
                let w22 = w[8];
                if let Some(ref b) = bias {
                    let b0 = b[0];
                    let b1 = b[1];
                    let b2 = b[2];
                    for f in 0..num_frames {
                        let ic = f * input_stride;
                        let oc = f * 3;
                        let i0 = input_data[ic];
                        let i1 = input_data[ic + 1];
                        let i2 = input_data[ic + 2];
                        out[oc] = w00.mul_add(i0, w01.mul_add(i1, w02.mul_add(i2, b0)));
                        out[oc + 1] = w10.mul_add(i0, w11.mul_add(i1, w12.mul_add(i2, b1)));
                        out[oc + 2] = w20.mul_add(i0, w21.mul_add(i1, w22.mul_add(i2, b2)));
                    }
                } else {
                    for f in 0..num_frames {
                        let ic = f * input_stride;
                        let oc = f * 3;
                        let i0 = input_data[ic];
                        let i1 = input_data[ic + 1];
                        let i2 = input_data[ic + 2];
                        out[oc] = w00.mul_add(i0, w01.mul_add(i1, w02 * i2));
                        out[oc + 1] = w10.mul_add(i0, w11.mul_add(i1, w12 * i2));
                        out[oc + 2] = w20.mul_add(i0, w21.mul_add(i1, w22 * i2));
                    }
                }
            }
            (3, 1) => {
                let w0 = w[0];
                let w1 = w[1];
                let w2 = w[2];
                if let Some(ref b) = bias {
                    let b0 = b[0];
                    let b1 = b[1];
                    let b2 = b[2];
                    for f in 0..num_frames {
                        let v = input_data[f * input_stride];
                        let oc = f * 3;
                        out[oc] = w0.mul_add(v, b0);
                        out[oc + 1] = w1.mul_add(v, b1);
                        out[oc + 2] = w2.mul_add(v, b2);
                    }
                } else {
                    for f in 0..num_frames {
                        let v = input_data[f * input_stride];
                        let oc = f * 3;
                        out[oc] = w0 * v;
                        out[oc + 1] = w1 * v;
                        out[oc + 2] = w2 * v;
                    }
                }
            }
            (1, 3) => {
                let w0 = w[0];
                let w1 = w[1];
                let w2 = w[2];
                if let Some(ref b) = bias {
                    let b0 = b[0];
                    for (f, o) in out.iter_mut().enumerate().take(num_frames) {
                        let ic = f * input_stride;
                        *o = w0.mul_add(
                            input_data[ic],
                            w1.mul_add(input_data[ic + 1], w2.mul_add(input_data[ic + 2], b0)),
                        );
                    }
                } else {
                    for (f, o) in out.iter_mut().enumerate().take(num_frames) {
                        let ic = f * input_stride;
                        *o = w0.mul_add(
                            input_data[ic],
                            w1.mul_add(input_data[ic + 1], w2 * input_data[ic + 2]),
                        );
                    }
                }
            }
            _ => {
                {
                    // General small-matrix path with fused bias
                    // Uses mul_add (FMA) to match Eigen's SIMD FMA behavior
                    if let Some(ref b) = bias {
                        for f in 0..num_frames {
                            let in_col_start = f * input_stride;
                            let out_col_start = f * out_ch;
                            for o in 0..out_ch {
                                let mut sum = b[o];
                                for i in 0..in_ch {
                                    sum = w[i * out_ch + o]
                                        .mul_add(input_data[in_col_start + i], sum);
                                }
                                out[out_col_start + o] = sum;
                            }
                        }
                    } else {
                        for f in 0..num_frames {
                            let in_col_start = f * input_stride;
                            let out_col_start = f * out_ch;
                            for o in 0..out_ch {
                                let mut sum = 0.0f32;
                                for i in 0..in_ch {
                                    sum = w[i * out_ch + o]
                                        .mul_add(input_data[in_col_start + i], sum);
                                }
                                out[out_col_start + o] = sum;
                            }
                        }
                    }
                }
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
    bias: Vec<f32>,
    kernel_size: usize,
    dilation: usize,
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
        let matrix_layout = MatrixLayout::new(out_channels, in_channels).map_err(|_| {
            NamError::DimensionOverflow {
                context: "convolution weight matrix",
                left: out_channels,
                right: in_channels,
            }
        })?;
        let is_depthwise = groups == in_channels && in_channels == out_channels;
        #[cfg(feature = "fast-kernels")]
        let flat_weight_len = kernel_size
            .checked_mul(if is_depthwise {
                out_channels
            } else {
                matrix_layout.left_len()
            })
            .ok_or(NamError::DimensionOverflow {
                context: "convolution weights",
                left: kernel_size,
                right: if is_depthwise {
                    out_channels
                } else {
                    matrix_layout.left_len()
                },
            })?;

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
            bias,
            kernel_size,
            dilation,
            out_channels,
            in_channels,
            matrix_layout,
            groups,
            input_buffer: RingBuffer2D::new(),
            output_buf: ColMajorMatrix::new(out_channels, 1),
            #[cfg(feature = "fast-kernels")]
            flat_weights,
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
    /// 2. For each kernel tap k: read from ring buffer with lookback, accumulate
    /// 3. Add bias
    fn process_block(&mut self, input: &ColMajorMatrix, num_frames: usize) {
        // Write input to ring buffer
        self.input_buffer.write(input, num_frames);

        let out_ch = self.out_channels;
        let in_ch = self.in_channels;
        let ks = self.kernel_size;
        let dil = self.dilation;

        // Fast-kernels path: single C FFI call for the entire Conv1d
        #[cfg(feature = "fast-kernels")]
        {
            let tap_slices: Vec<&[f32]> = (0..ks)
                .map(|k| {
                    let offset_signed: isize = dil as isize * (k as isize + 1 - ks as isize);
                    let lookback = (-offset_signed) as usize;
                    self.input_buffer.read_ptr(num_frames, lookback)
                })
                .collect();
            let use_sgemm = out_ch * in_ch >= SGEMM_MIN_SIZE;
            match &self.weights {
                Conv1dWeights::Depthwise(_) => {
                    crate::fast_kernels::conv1d_depthwise(
                        &mut self.output_buf.data,
                        &tap_slices,
                        &self.flat_weights,
                        &self.bias,
                        out_ch,
                        num_frames,
                    );
                }
                Conv1dWeights::General(_) if !use_sgemm => {
                    crate::fast_kernels::conv1d_small_gemv(
                        &mut self.output_buf.data,
                        &tap_slices,
                        &self.flat_weights,
                        &self.bias,
                        out_ch,
                        in_ch,
                        num_frames,
                    );
                }
                Conv1dWeights::General(weights_colmajor) => {
                    // Large matrix: initialize with bias, then accumulate via sgemm
                    for f in 0..num_frames {
                        let off = f * out_ch;
                        self.output_buf.data[off..off + out_ch].copy_from_slice(&self.bias);
                    }
                    for k in 0..ks {
                        let w = &weights_colmajor[k];
                        self.matrix_layout.multiply(
                            num_frames,
                            1.0,
                            w,
                            tap_slices[k],
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

        // Initialize output with bias (fused: eliminates separate bias-add pass)
        // (unreachable when fast-kernels feature is enabled, the block above returns early)
        #[allow(unreachable_code)]
        for f in 0..num_frames {
            let off = f * out_ch;
            self.output_buf.data[off..off + out_ch].copy_from_slice(&self.bias);
        }

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

                    if use_sgemm {
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
                                alpha * activated + (1.0 - alpha) * pre_act;
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
                                        alpha * activated + (1.0 - alpha) * pre_act;
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
                                    alpha * activated + (1.0 - alpha) * pre_act;
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
            return Err(NamError::InvalidConfig(format!(
                "Unsupported slimmable method: {}",
                method
            )));
        }
        let allowed = value
            .get("kwargs")
            .and_then(|v| v.get("allowed_channels"))
            .and_then(|v| v.as_array())
            .ok_or_else(|| NamError::MissingField("slimmable.kwargs.allowed_channels".into()))?;
        let allowed_channels = allowed
            .iter()
            .map(|v| {
                let channel = v.as_u64().ok_or_else(|| {
                    NamError::InvalidConfig(
                        "slimmable.kwargs.allowed_channels contains non-integer value".into(),
                    )
                })? as usize;
                if channel == 0 || channel > channels {
                    return Err(NamError::InvalidConfig(format!(
                        "slimmable allowed channel count {} is outside 1..={}",
                        channel, channels
                    )));
                }
                Ok(channel)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if allowed_channels.is_empty() {
            return Err(NamError::InvalidConfig(
                "slimmable.kwargs.allowed_channels must not be empty".into(),
            ));
        }
        let active_channels = allowed_channels[allowed_channels.len() - 1];
        Ok(Some(Self {
            allowed_channels,
            active_channels,
        }))
    }

    fn set_slimming(&mut self, value: f64) -> Result<(), NamError> {
        if !value.is_finite() {
            return Err(NamError::InvalidConfig(
                "Slimming value must be finite".into(),
            ));
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
    fn receptive_field(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.conv.receptive_field())
            .sum::<usize>()
            + self.head_rechannel.receptive_field()
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
        let channels = config["channels"]
            .as_u64()
            .ok_or_else(|| NamError::MissingField("head.channels".into()))?
            as usize;
        let out_channels = config["out_channels"]
            .as_u64()
            .ok_or_else(|| NamError::MissingField("head.out_channels".into()))?
            as usize;
        let activation = Activation::from_json(&config["activation"])?;
        let kernel_sizes = config["kernel_sizes"]
            .as_array()
            .ok_or_else(|| NamError::MissingField("head.kernel_sizes".into()))?;
        if kernel_sizes.is_empty() {
            return Err(NamError::InvalidConfig(
                "head.kernel_sizes must be non-empty".into(),
            ));
        }

        let mut blocks = Vec::with_capacity(kernel_sizes.len());
        let mut block_in_channels = in_channels;
        for (idx, kernel_size) in kernel_sizes.iter().enumerate() {
            let kernel_size = kernel_size.as_u64().ok_or_else(|| {
                NamError::InvalidConfig("head.kernel_sizes contains non-integer value".into())
            })? as usize;
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

    fn receptive_field(&self) -> usize {
        self.blocks
            .iter()
            .map(WaveNetHeadBlock::receptive_field)
            .sum()
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
    max_buffer_size: usize,
    activation_mode: ActivationMode,
}

fn normalize_wavenet_config(config: &serde_json::Value) -> Result<serde_json::Value, NamError> {
    let mut normalized = config.clone();
    let Some(root) = normalized.as_object_mut() else {
        return Err(NamError::InvalidConfig(
            "WaveNet config must be a JSON object".into(),
        ));
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
        return Err(NamError::InvalidConfig(
            "WaveNet layer array config must be a JSON object".into(),
        ));
    };

    if let Some(head) = layer_obj.get("head").and_then(|head| head.as_object()) {
        let out_channels = head.get("out_channels").cloned();
        let bias = head.get("bias").cloned();
        let kernel_size = head.get("kernel_size").cloned();
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
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;

        for (la_idx, la_json) in layers_json.iter().enumerate() {
            let input_size = la_json["input_size"]
                .as_u64()
                .ok_or_else(|| NamError::MissingField("layer_array.input_size".into()))?
                as usize;
            let cond_size = la_json["condition_size"]
                .as_u64()
                .ok_or_else(|| NamError::MissingField("layer_array.condition_size".into()))?
                as usize;
            let head_size = la_json["head_size"]
                .as_u64()
                .ok_or_else(|| NamError::MissingField("layer_array.head_size".into()))?
                as usize;
            let channels = la_json["channels"]
                .as_u64()
                .ok_or_else(|| NamError::MissingField("layer_array.channels".into()))?
                as usize;
            let bottleneck = la_json
                .get("bottleneck")
                .and_then(|v| v.as_u64())
                .unwrap_or(channels as u64) as usize;
            let dilations_arr = la_json["dilations"]
                .as_array()
                .ok_or_else(|| NamError::MissingField("layer_array.dilations".into()))?;
            let dilations: Vec<usize> = dilations_arr
                .iter()
                .map(|v| {
                    v.as_u64()
                        .ok_or_else(|| {
                            NamError::InvalidConfig(
                                "layer_array.dilations contains non-integer value".into(),
                            )
                        })
                        .map(|n| n as usize)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let num_layers = dilations.len();

            // Parse kernel sizes: support legacy single `kernel_size` (int) or
            // new per-layer `kernel_sizes` (array). Mutual exclusivity enforced.
            let kernel_sizes: Vec<usize> = {
                let has_kernel_size = la_json.get("kernel_size").is_some();
                let has_kernel_sizes = la_json.get("kernel_sizes").is_some();

                if has_kernel_size && has_kernel_sizes {
                    return Err(NamError::InvalidConfig(format!(
                        "Layer array {}: only one of kernel_size (int) or kernel_sizes (array) may be provided",
                        la_idx
                    )));
                } else if has_kernel_sizes {
                    let arr = la_json["kernel_sizes"].as_array().ok_or_else(|| {
                        NamError::InvalidConfig(format!(
                            "Layer array {}: kernel_sizes must be an array",
                            la_idx
                        ))
                    })?;
                    let ks: Vec<usize> = arr
                        .iter()
                        .map(|v| {
                            v.as_u64()
                                .ok_or_else(|| {
                                    NamError::InvalidConfig(
                                        "kernel_sizes contains non-integer value".into(),
                                    )
                                })
                                .map(|n| n as usize)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    if ks.len() != num_layers {
                        return Err(NamError::InvalidConfig(format!(
                            "Layer array {}: kernel_sizes length ({}) must match dilations length ({})",
                            la_idx,
                            ks.len(),
                            num_layers
                        )));
                    }
                    ks
                } else if has_kernel_size {
                    let ks_val = &la_json["kernel_size"];
                    if let Some(arr) = ks_val.as_array() {
                        // Also accept kernel_size as an array (trainer compat)
                        let ks: Vec<usize> = arr
                            .iter()
                            .map(|v| {
                                v.as_u64()
                                    .ok_or_else(|| {
                                        NamError::InvalidConfig(
                                            "kernel_size array contains non-integer value".into(),
                                        )
                                    })
                                    .map(|n| n as usize)
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        if ks.len() != num_layers {
                            return Err(NamError::InvalidConfig(format!(
                                "Layer array {}: kernel_size array length ({}) must match dilations length ({})",
                                la_idx,
                                ks.len(),
                                num_layers
                            )));
                        }
                        ks
                    } else {
                        let ks = ks_val.as_u64().ok_or_else(|| {
                            NamError::InvalidConfig(format!(
                                "Layer array {}: kernel_size must be an integer or array",
                                la_idx
                            ))
                        })? as usize;
                        vec![ks; num_layers]
                    }
                } else {
                    return Err(NamError::MissingField(
                        "layer_array: either kernel_size or kernel_sizes must be provided".into(),
                    ));
                }
            };

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
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize;
            let slimmable = SlimmableConfig::from_json(la_json.get("slimmable"), channels)?;

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

            if slimmable.is_some() {
                if condition_dsp.is_some() {
                    return Err(NamError::InvalidConfig(
                        "SlimmableWaveNet does not support condition_dsp".into(),
                    ));
                }
                if layers_json.len() != 1 {
                    return Err(NamError::InvalidConfig(
                        "SlimmableWaveNet supports exactly one layer array".into(),
                    ));
                }
                if groups_input != 1 || groups_input_mixin != 1 {
                    return Err(NamError::InvalidConfig(
                        "SlimmableWaveNet does not support grouped convolutions".into(),
                    ));
                }
                if head1x1_params.active {
                    return Err(NamError::InvalidConfig(
                        "SlimmableWaveNet does not support head1x1".into(),
                    ));
                }
                if layer1x1_groups != 1 {
                    return Err(NamError::InvalidConfig(
                        "SlimmableWaveNet does not support grouped layer1x1".into(),
                    ));
                }
                if film_params.any_active() {
                    return Err(NamError::InvalidConfig(
                        "SlimmableWaveNet does not support FiLM".into(),
                    ));
                }
                if head_kernel_size != 1 {
                    return Err(NamError::InvalidConfig(
                        "SlimmableWaveNet requires head kernel_size 1".into(),
                    ));
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
                1,
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
            .ok_or_else(|| NamError::InvalidConfig("WaveNet requires at least one layer".into()))?;
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
        let prewarm_samples_count = condition_prewarm
            + layer_arrays
                .iter()
                .map(|la| la.receptive_field())
                .sum::<usize>()
            + head.as_ref().map(WaveNetHead::receptive_field).unwrap_or(0);

        // Determine condition output channels
        let cond_out_ch = if let Some(ref cdsp) = condition_dsp {
            cdsp.num_output_channels()
        } else {
            in_channels
        };

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
            Err(NamError::InvalidConfig(
                "WaveNet does not have slimmable layer arrays".into(),
            ))
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
        // Process the full block through the WaveNet
        // We need a dummy output buffer for process_block's mono output
        let mut dummy_output = vec![Sample::default(); num_frames];
        self.process_block(input, &mut dummy_output);

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
    use proptest::prelude::*;
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
    fn test_slimmable_wavenet_set_slimming() {
        let mut model = match load_wavenet("slimmable_wavenet.nam") {
            Some(m) => m,
            None => return,
        };
        let input = vec![0.1 as Sample; 64];
        let mut output = vec![0.0 as Sample; 64];

        model.set_slimming(0.0).unwrap();
        model.reset(48_000.0, 64);
        model.process(&input, &mut output);
        assert!(
            output.iter().all(|&x| x.is_finite()),
            "SlimmableWaveNet smallest width produced non-finite output"
        );

        model.set_slimming(1.0).unwrap();
        model.reset(48_000.0, 64);
        model.process(&input, &mut output);
        assert!(
            output.iter().all(|&x| x.is_finite()),
            "SlimmableWaveNet largest width produced non-finite output"
        );
    }

    #[test]
    fn test_slimmable_wavenet_set_slimming_changes_output() {
        let input = vec![0.1 as Sample; 64];

        let mut smallest = match load_wavenet("slimmable_wavenet.nam") {
            Some(m) => m,
            None => return,
        };
        smallest.set_slimming(0.0).unwrap();
        smallest.reset(48_000.0, 64);
        let mut smallest_output = vec![0.0 as Sample; 64];
        smallest.process(&input, &mut smallest_output);

        let mut largest = match load_wavenet("slimmable_wavenet.nam") {
            Some(m) => m,
            None => return,
        };
        largest.set_slimming(1.0).unwrap();
        largest.reset(48_000.0, 64);
        let mut largest_output = vec![0.0 as Sample; 64];
        largest.process(&input, &mut largest_output);

        assert_ne!(
            smallest_output, largest_output,
            "SlimmableWaveNet width selection should affect the rendered output"
        );
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
                output.iter().all(|&x| x.is_finite()),
                "Non-finite output from {}",
                name
            );
        }
    }

    fn minimal_upstream_layer_config() -> serde_json::Value {
        serde_json::json!({
            "condition_size": 1,
            "input_size": 1,
            "channels": 1,
            "head": {"out_channels": 1, "kernel_size": 1, "bias": false},
            "kernel_size": 1,
            "dilations": [1],
            "activation": "Tanh",
            "layer_1x1_config": {"active": true, "groups": 1},
            "head_1x1_config": {"active": false, "out_channels": 1, "groups": 1}
        })
    }

    fn process_small_config(config: serde_json::Value, weights: Vec<f32>) -> Vec<Sample> {
        let mut model = WaveNet::from_config(&config, &weights, DspMetadata::default()).unwrap();
        model.reset(48_000.0, 8);
        let input = vec![0.1 as Sample; 8];
        let mut output = vec![0.0 as Sample; 8];
        model.process(&input, &mut output);
        output
    }

    fn render_small_config(
        activation: &str,
        weights: &[f32],
        input: &[Sample],
        chunk_size: usize,
    ) -> Vec<Sample> {
        let mut layer = minimal_upstream_layer_config();
        layer["activation"] = serde_json::Value::String(activation.to_string());
        let config = serde_json::json!({
            "layers_configs": [layer],
            "head": null,
            "head_scale": 1.0
        });
        let mut model = WaveNet::from_config(&config, weights, DspMetadata::default()).unwrap();
        model.reset(48_000.0, input.len().max(1));
        let mut output = Vec::with_capacity(input.len());
        for chunk in input.chunks(chunk_size) {
            let mut chunk_output = vec![Sample::default(); chunk.len()];
            model.process(chunk, &mut chunk_output);
            output.extend(chunk_output);
        }
        output
    }

    proptest! {
        #[test]
        fn backend_equivalence_matrix_matches_column_major_oracle(
            m in 1usize..12,
            k in 1usize..12,
            n in 1usize..12,
            padding in 0usize..4,
            raw_a in prop::collection::vec(-1.0f32..1.0, 1..144),
            raw_b in prop::collection::vec(-1.0f32..1.0, 1..180),
            raw_c in prop::collection::vec(-1.0f32..1.0, 1..144),
        ) {
            let stride = k + padding;
            let a = raw_a.into_iter().cycle().take(m * k).collect::<Vec<_>>();
            let b = raw_b.into_iter().cycle().take(stride * n).collect::<Vec<_>>();
            let initial = raw_c.into_iter().cycle().take(m * n).collect::<Vec<_>>();
            let mut actual = initial.clone();
            let mut expected = initial;
            let alpha = 0.75f32;
            let beta = -0.25f32;

            let layout = MatrixLayout::new(m, k).unwrap();
            prop_assert!(layout.multiply(n, alpha, &a, &b, stride, beta, &mut actual));
            for column in 0..n {
                for row in 0..m {
                    let mut product = 0.0f32;
                    for inner in 0..k {
                        product += a[inner * m + row] * b[column * stride + inner];
                    }
                    let index = column * m + row;
                    expected[index] = alpha * product + beta * expected[index];
                }
            }

            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                prop_assert!(
                    (actual - expected).abs() <= 2.0e-4,
                    "matrix output {index}: expected {expected}, got {actual}"
                );
            }
        }

        #[test]
        fn backend_equivalence_randomized_wavenet_is_block_partition_invariant(
            activation in prop_oneof![
                Just("Tanh"),
                Just("ReLU"),
                Just("Sigmoid"),
                Just("Softsign"),
            ],
            weights in prop::collection::vec(-0.75f32..0.75, 8),
            input in prop::collection::vec(-0.5f64..0.5, 1..96),
            chunk_size in 1usize..24,
        ) {
            let input = input
                .into_iter()
                .map(crate::dsp::sample_from_f64)
                .collect::<Vec<_>>();
            let full = render_small_config(activation, &weights, &input, input.len());
            let partitioned = render_small_config(activation, &weights, &input, chunk_size);
            for (index, (&full, &partitioned)) in full.iter().zip(&partitioned).enumerate() {
                prop_assert!(
                    (full - partitioned).abs() <= 2.0e-5,
                    "sample {index}: full={full}, partitioned={partitioned}"
                );
            }
        }
    }

    #[test]
    fn test_upstream_layers_configs_and_head_object_load() {
        let config = serde_json::json!({
            "layers_configs": [minimal_upstream_layer_config()],
            "head": null,
            "head_scale": 1.0
        });
        let weights = vec![1.0, 0.5, 0.0, 0.25, 1.0, 0.0, 1.0, 1.0];
        let output = process_small_config(config, weights);
        assert!(output.iter().all(|&sample| sample.is_finite()));
    }

    #[test]
    fn test_upstream_layer_aliases_and_film_params_load() {
        let mut layer = minimal_upstream_layer_config();
        layer["film_params"] = serde_json::json!({
            "conv_pre_film": {"active": true, "shift": true, "groups": 1}
        });
        let config = serde_json::json!({
            "layers_configs": [layer],
            "head": null,
            "head_scale": 1.0
        });
        let weights = vec![
            1.0, // rechannel
            0.5, 0.0,  // conv
            0.25, // input mixin
            1.0, 0.0, // layer1x1
            1.0, 0.0, 0.0, 0.0, // conv_pre_film
            1.0, // head rechannel
            1.0, // head_scale
        ];
        let output = process_small_config(config, weights);
        assert!(output.iter().all(|&sample| sample.is_finite()));
    }

    #[test]
    fn test_upstream_pairmultiply_activation_config_loads() {
        let mut layer = minimal_upstream_layer_config();
        layer["activation"] = serde_json::json!({
            "name": "PairMultiply",
            "primary": "Tanh",
            "secondary": "Sigmoid"
        });
        let config = serde_json::json!({
            "layers_configs": [layer],
            "head": null,
            "head_scale": 1.0
        });
        let weights = vec![
            1.0, // rechannel
            0.5, 0.5, 0.0, 0.0, // gated conv
            0.25, 0.25, // gated input mixin
            1.0, 0.0, // layer1x1
            1.0, // head rechannel
            1.0, // head_scale
        ];
        let output = process_small_config(config, weights);
        assert!(output.iter().all(|&sample| sample.is_finite()));
    }

    #[test]
    fn test_top_level_head_loads_and_processes() {
        let config = serde_json::json!({
            "layers_configs": [minimal_upstream_layer_config()],
            "head": {
                "channels": 1,
                "activation": {"name": "Softsigmoid"},
                "out_channels": 1,
                "kernel_sizes": [1]
            },
            "head_scale": 1.0
        });
        let weights = vec![
            1.0, // rechannel
            0.5, 0.0,  // conv
            0.25, // input mixin
            1.0, 0.0, // layer1x1
            1.0, // layer-array head rechannel
            1.0, 0.0, // top-level head conv
            1.0, // head_scale
        ];
        let output = process_small_config(config, weights);
        assert!(output.iter().all(|&sample| sample.is_finite()));
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

        assert!(output.iter().all(|&x| x.is_finite()));
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

        assert!(output.iter().all(|&x| x.is_finite()));
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
                output.iter().all(|&x| x.is_finite()),
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
                output.iter().all(|&x| x.is_finite()),
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
        assert!(output.iter().all(|&x| x.is_finite()));
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
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    // ── Per-layer kernel_sizes tests ────────────────────────────────────────

    /// Helper: build a minimal WaveNet JSON config and matching weight vec.
    /// Returns (config_json, weights) for simple 1-channel, no-gating configs.
    fn make_kernel_size_config(kernel_field: &str, num_layers: usize) -> (String, Vec<f32>) {
        // Weight budget for 1-ch, 1-bottleneck, no-gating, layer1x1-active, no-head1x1:
        //   rechannel: 1, per layer with kernel K: K+4, head_rechannel: 1, head_scale: 1
        // Parse kernel sizes from the field string to compute exact weight count.
        let kernel_sizes: Vec<usize> = if kernel_field.contains('[') {
            // Array form: extract numbers from brackets
            let start = kernel_field.find('[').unwrap();
            let end = kernel_field.find(']').unwrap();
            kernel_field[start + 1..end]
                .split(',')
                .map(|s| s.trim().parse::<usize>().unwrap())
                .collect()
        } else {
            // Scalar form: extract the number
            let num: usize = kernel_field
                .split(':')
                .next_back()
                .unwrap()
                .trim()
                .trim_matches('"')
                .parse()
                .unwrap();
            vec![num; num_layers]
        };
        let num_weights = 1 + kernel_sizes.iter().map(|k| k + 4).sum::<usize>() + 1 + 1;
        let weights = vec![1.0f32; num_weights];
        let dilations: Vec<String> = (0..num_layers).map(|i| format!("{}", 1 << i)).collect();
        let config_str = format!(
            r#"{{
                "layers": [{{
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 1,
                    {},
                    "dilations": [{}],
                    "activation": "ReLU",
                    "gated": false,
                    "head_bias": false
                }}],
                "head_scale": 1.0
            }}"#,
            kernel_field,
            dilations.join(", ")
        );
        (config_str, weights)
    }

    #[test]
    fn test_kernel_size_int_compat() {
        // Legacy single kernel_size integer should be expanded to all layers
        let (config_str, weights) = make_kernel_size_config(r#""kernel_size": 3"#, 3);
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();
        let metadata = DspMetadata::default();
        let result = WaveNet::from_config(&config, &weights, metadata);
        assert!(
            result.is_ok(),
            "Legacy kernel_size int should parse: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_kernel_sizes_per_layer_array() {
        // New per-layer kernel_sizes array
        let (config_str, weights) = make_kernel_size_config(r#""kernel_sizes": [2, 3, 5]"#, 3);
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();
        let metadata = DspMetadata::default();
        let result = WaveNet::from_config(&config, &weights, metadata);
        assert!(
            result.is_ok(),
            "Per-layer kernel_sizes should parse: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_kernel_size_mutual_exclusivity() {
        // Providing both kernel_size and kernel_sizes should error
        let (config_str, weights) =
            make_kernel_size_config(r#""kernel_size": 3, "kernel_sizes": [2, 3, 5]"#, 3);
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();
        let metadata = DspMetadata::default();
        let result = WaveNet::from_config(&config, &weights, metadata);
        let err = result
            .err()
            .expect("Both kernel_size and kernel_sizes should be rejected");
        let err_msg = format!("{}", err);
        assert!(
            err_msg.contains("only one of"),
            "Error should mention mutual exclusivity: {}",
            err_msg
        );
    }

    #[test]
    fn test_kernel_sizes_length_mismatch() {
        // kernel_sizes length != dilations length should error
        let (config_str, weights) = make_kernel_size_config(r#""kernel_sizes": [2, 3]"#, 3);
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();
        let metadata = DspMetadata::default();
        let result = WaveNet::from_config(&config, &weights, metadata);
        let err = result
            .err()
            .expect("Mismatched kernel_sizes length should be rejected");
        let err_msg = format!("{}", err);
        assert!(
            err_msg.contains("must match"),
            "Error should mention length mismatch: {}",
            err_msg
        );
    }

    #[test]
    fn test_no_kernel_size_field_errors() {
        // Neither kernel_size nor kernel_sizes should error
        let config_str = r#"{
            "layers": [{
                "input_size": 1,
                "condition_size": 1,
                "head_size": 1,
                "channels": 1,
                "dilations": [1, 2],
                "activation": "ReLU",
                "gated": false,
                "head_bias": false
            }],
            "head_scale": 1.0
        }"#;
        let config: serde_json::Value = serde_json::from_str(config_str).unwrap();
        let weights = vec![1.0f32; 500];
        let metadata = DspMetadata::default();
        let result = WaveNet::from_config(&config, &weights, metadata);
        assert!(result.is_err(), "Missing kernel_size should be rejected");
    }

    #[test]
    fn test_kernel_sizes_per_layer_different_receptive_fields() {
        // With kernel_sizes [2, 3] and dilations [1, 2]:
        //   layer 0: RF = 1 * (2-1) = 1
        //   layer 1: RF = 2 * (3-1) = 4
        //   total RF = 5, prewarm = 1 (base) + 5 = 6
        let (config_str, weights) = make_kernel_size_config(r#""kernel_sizes": [2, 3]"#, 2);
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();
        let metadata = DspMetadata::default();
        let model = WaveNet::from_config(&config, &weights, metadata).unwrap();
        assert_eq!(model.prewarm_samples(), 6);
    }

    #[test]
    fn test_kernel_size_as_array() {
        // kernel_size (singular key) with array value should also be accepted
        // for compatibility with trainer exports
        let (config_str, weights) = make_kernel_size_config(r#""kernel_size": [2, 3]"#, 2);
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();
        let metadata = DspMetadata::default();
        let result = WaveNet::from_config(&config, &weights, metadata);
        assert!(
            result.is_ok(),
            "kernel_size as array should parse: {:?}",
            result.err()
        );
        let model = result.unwrap();
        // dilations [1, 2], kernel_sizes [2, 3]: RF = 1*(2-1) + 2*(3-1) = 5, prewarm = 1 + 5 = 6
        assert_eq!(model.prewarm_samples(), 6);
    }

    #[test]
    fn test_kernel_size_int_receptive_field() {
        // With kernel_size=3 and dilations [1, 2]:
        //   layer 0: RF = 1 * (3-1) = 2
        //   layer 1: RF = 2 * (3-1) = 4
        //   total RF = 6, prewarm = 1 + 6 = 7
        let (config_str, weights) = make_kernel_size_config(r#""kernel_size": 3"#, 2);
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();
        let metadata = DspMetadata::default();
        let model = WaveNet::from_config(&config, &weights, metadata).unwrap();
        assert_eq!(model.prewarm_samples(), 7);
    }

    // ── Depthwise convolution tests ─────────────────────────────────────────

    #[test]
    fn test_conv1d_depthwise_detected() {
        // groups == in_channels == out_channels triggers depthwise path
        let weights_data = vec![1.0f32; 100];
        let mut iter = crate::util::WeightIter::new(&weights_data);
        let conv = Conv1d::from_weights(4, 4, 3, 1, 4, &mut iter).unwrap();
        assert!(matches!(conv.weights, Conv1dWeights::Depthwise(_)));
    }

    #[test]
    fn test_conv1d_general_when_not_depthwise() {
        // groups != in_channels should use general path
        let weights_data = vec![1.0f32; 100];
        let mut iter = crate::util::WeightIter::new(&weights_data);
        let conv = Conv1d::from_weights(4, 4, 3, 1, 2, &mut iter).unwrap();
        assert!(matches!(conv.weights, Conv1dWeights::General(_)));
    }

    #[test]
    fn test_conv1d_depthwise_identity() {
        // 2-channel depthwise with kernel_size=1, weights=[1,1]
        // Should act as identity (plus bias)
        let weights_data = vec![1.0, 1.0, 0.0, 0.0]; // 2 weights + 2 bias
        let mut iter = crate::util::WeightIter::new(&weights_data);
        let mut conv = Conv1d::from_weights(2, 2, 1, 1, 2, &mut iter).unwrap();
        conv.set_max_buffer_size(4);

        let mut input = ColMajorMatrix::new(2, 4);
        // Frame 0: [3.0, 5.0], Frame 1: [7.0, 11.0]
        input.data[0] = 3.0;
        input.data[1] = 5.0;
        input.data[2] = 7.0;
        input.data[3] = 11.0;

        conv.process_block(&input, 2);
        // With weight=1 and bias=0: output should equal input
        assert!((conv.output_buf.data[0] - 3.0).abs() < 1e-6);
        assert!((conv.output_buf.data[1] - 5.0).abs() < 1e-6);
        assert!((conv.output_buf.data[2] - 7.0).abs() < 1e-6);
        assert!((conv.output_buf.data[3] - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_conv1d_depthwise_scaling() {
        // 2-channel depthwise with kernel_size=1, weights=[2, 3], bias=[10, 20]
        let weights_data = vec![2.0, 3.0, 10.0, 20.0];
        let mut iter = crate::util::WeightIter::new(&weights_data);
        let mut conv = Conv1d::from_weights(2, 2, 1, 1, 2, &mut iter).unwrap();
        conv.set_max_buffer_size(4);

        let mut input = ColMajorMatrix::new(2, 4);
        input.data[0] = 1.0; // ch0, frame0
        input.data[1] = 1.0; // ch1, frame0

        conv.process_block(&input, 1);
        // ch0: 2*1 + 10 = 12, ch1: 3*1 + 20 = 23
        assert!((conv.output_buf.data[0] - 12.0).abs() < 1e-6);
        assert!((conv.output_buf.data[1] - 23.0).abs() < 1e-6);
    }

    #[test]
    fn test_conv1d_depthwise_multi_tap() {
        // 2-channel depthwise with kernel_size=2, dilation=1
        // weights: ch0=[1, 2], ch1=[3, 4], bias=[0, 0]
        // C++ weight order for depthwise: for each channel c, for each tap k
        let weights_data = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0];
        let mut iter = crate::util::WeightIter::new(&weights_data);
        let mut conv = Conv1d::from_weights(2, 2, 2, 1, 2, &mut iter).unwrap();
        conv.set_max_buffer_size(4);

        // Process two calls to build up ring buffer history
        let mut input1 = ColMajorMatrix::new(2, 4);
        input1.data[0] = 1.0; // ch0
        input1.data[1] = 0.0; // ch1
        conv.process_block(&input1, 1);

        let mut input2 = ColMajorMatrix::new(2, 4);
        input2.data[0] = 0.0; // ch0
        input2.data[1] = 1.0; // ch1
        conv.process_block(&input2, 1);
        // Tap ordering: k=0 has lookback=1 (prev), k=1 has lookback=0 (current)
        // Frame 1 output:
        //   ch0: w[0]*prev[ch0] + w[1]*now[ch0] = 1*1 + 2*0 = 1
        //   ch1: w[0]*prev[ch1] + w[1]*now[ch1] = 3*0 + 4*1 = 4
        assert!(
            (conv.output_buf.data[0] - 1.0).abs() < 1e-6,
            "ch0: {}",
            conv.output_buf.data[0]
        );
        assert!(
            (conv.output_buf.data[1] - 4.0).abs() < 1e-6,
            "ch1: {}",
            conv.output_buf.data[1]
        );
    }
}
