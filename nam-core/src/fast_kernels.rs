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

    /// Tanh in-place (standard math, compiled with -ffast-math).
    pub fn fast_tanh_inplace(data: *mut f32, len: usize);

    /// Fast tanh polynomial in-place.
    pub fn fast_tanh_poly_inplace(data: *mut f32, len: usize);
}
