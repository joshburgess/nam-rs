/// FFI bindings to coarse-grained C kernels compiled with -ffast-math.
/// Each function processes an entire operation to amortize FFI call overhead.

extern "C" {
    /// Full Conv1d depthwise: all taps + bias in one call.
    /// weights layout: weights[k * ch + c] for tap k, channel c.
    /// tap_ptrs: array of kernel_size pointers to input data per tap.
    pub fn fast_conv1d_depthwise(
        output: *mut f32,
        tap_ptrs: *const *const f32,
        weights: *const f32,
        bias: *const f32,
        ch: usize,
        kernel_size: usize,
        num_frames: usize,
    );

    /// Full Conv1d small-matrix: all taps + bias in one call.
    /// weights layout: weights[k * (out_ch * in_ch) + i * out_ch + o] col-major per tap.
    pub fn fast_conv1d_small_gemv(
        output: *mut f32,
        tap_ptrs: *const *const f32,
        weights: *const f32,
        bias: *const f32,
        out_ch: usize,
        in_ch: usize,
        kernel_size: usize,
        num_frames: usize,
    );

    /// Fused z = activation(conv_out + mixin_out).
    pub fn fast_add_activate(
        z_out: *mut f32,
        conv_out: *const f32,
        mixin_out: *const f32,
        len: usize,
        use_fast_tanh: i32,
    );

    /// Conv1x1 small GEMM with optional bias.
    pub fn fast_conv1x1_small(
        output: *mut f32,
        weights: *const f32,
        input: *const f32,
        bias: *const f32, // null if no bias
        out_ch: usize,
        in_ch: usize,
        input_stride: usize,
        num_frames: usize,
    );

    /// FiLM scale+shift (out-of-place).
    pub fn fast_film_scale_shift(
        output: *mut f32,
        input: *const f32,
        scale_shift: *const f32,
        dim: usize,
        input_stride: usize,
        output_stride: usize,
        ss_rows: usize,
        num_frames: usize,
    );

    /// FiLM scale only (out-of-place).
    pub fn fast_film_scale(
        output: *mut f32,
        input: *const f32,
        scale: *const f32,
        dim: usize,
        input_stride: usize,
        output_stride: usize,
        ss_rows: usize,
        num_frames: usize,
    );

    /// FiLM in-place scale+shift.
    pub fn fast_film_inplace_scale_shift(
        data: *mut f32,
        scale_shift: *const f32,
        dim: usize,
        data_stride: usize,
        ss_rows: usize,
        num_frames: usize,
    );

    /// FiLM in-place scale only.
    pub fn fast_film_inplace_scale(
        data: *mut f32,
        scale: *const f32,
        dim: usize,
        data_stride: usize,
        ss_rows: usize,
        num_frames: usize,
    );

    /// Element-wise vector add: c[i] = a[i] + b[i]
    pub fn fast_vec_add(c: *mut f32, a: *const f32, b: *const f32, len: usize);

    /// Element-wise vector add in-place: a[i] += b[i]
    pub fn fast_vec_add_inplace(a: *mut f32, b: *const f32, len: usize);

    /// Add bias to each column: output[f*ch + o] += bias[o]
    pub fn fast_add_bias(output: *mut f32, bias: *const f32, ch: usize, num_frames: usize);

    /// Gated activation: z[c] = primary(z[c]) * secondary(z[bottleneck+c])
    pub fn fast_gated_activation(
        z: *mut f32, z_rows: usize, bottleneck: usize, num_frames: usize,
        primary_type: i32, secondary_type: i32, use_fast_tanh: i32,
    );

    /// Blended activation: z[c] = alpha * activated + (1-alpha) * pre_act
    pub fn fast_blended_activation(
        z: *mut f32, z_rows: usize, bottleneck: usize, num_frames: usize,
        primary_type: i32, secondary_type: i32, use_fast_tanh: i32,
    );

    /// Generic activation in-place.
    pub fn fast_activation_inplace(data: *mut f32, len: usize, act_type: i32, use_fast_tanh: i32);

    /// Tanh in-place (standard math, compiled with -ffast-math).
    pub fn fast_tanh_inplace(data: *mut f32, len: usize);

    /// Fast tanh polynomial in-place.
    pub fn fast_tanh_poly_inplace(data: *mut f32, len: usize);
}
