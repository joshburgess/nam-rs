mod ffi {
    #[allow(dead_code)]
    extern "C" {
        #[link_name = "fast_conv1d_depthwise"]
        pub(super) fn conv1d_depthwise(
            output: *mut f32,
            tap_ptrs: *const *const f32,
            weights: *const f32,
            bias: *const f32,
            ch: usize,
            kernel_size: usize,
            num_frames: usize,
        );
        #[link_name = "fast_conv1d_small_gemv"]
        pub(super) fn conv1d_small_gemv(
            output: *mut f32,
            tap_ptrs: *const *const f32,
            weights: *const f32,
            bias: *const f32,
            out_ch: usize,
            in_ch: usize,
            kernel_size: usize,
            num_frames: usize,
        );
        #[link_name = "fast_add_activate"]
        pub(super) fn add_activate(
            output: *mut f32,
            left: *const f32,
            right: *const f32,
            len: usize,
            use_fast_tanh: i32,
        );
        #[link_name = "fast_conv1x1_small"]
        pub(super) fn conv1x1_small(
            output: *mut f32,
            weights: *const f32,
            input: *const f32,
            bias: *const f32,
            out_ch: usize,
            in_ch: usize,
            input_stride: usize,
            num_frames: usize,
        );
        #[link_name = "fast_film_scale_shift"]
        pub(super) fn film_scale_shift(
            output: *mut f32,
            input: *const f32,
            scale_shift: *const f32,
            dim: usize,
            input_stride: usize,
            output_stride: usize,
            scale_shift_rows: usize,
            num_frames: usize,
        );
        #[link_name = "fast_film_scale"]
        pub(super) fn film_scale(
            output: *mut f32,
            input: *const f32,
            scale: *const f32,
            dim: usize,
            input_stride: usize,
            output_stride: usize,
            scale_rows: usize,
            num_frames: usize,
        );
        #[link_name = "fast_film_inplace_scale_shift"]
        pub(super) fn film_inplace_scale_shift(
            data: *mut f32,
            scale_shift: *const f32,
            dim: usize,
            data_stride: usize,
            scale_shift_rows: usize,
            num_frames: usize,
        );
        #[link_name = "fast_film_inplace_scale"]
        pub(super) fn film_inplace_scale(
            data: *mut f32,
            scale: *const f32,
            dim: usize,
            data_stride: usize,
            scale_rows: usize,
            num_frames: usize,
        );
        #[link_name = "fast_vec_add"]
        pub(super) fn vec_add(output: *mut f32, left: *const f32, right: *const f32, len: usize);
        #[link_name = "fast_vec_add_inplace"]
        pub(super) fn vec_add_inplace(left: *mut f32, right: *const f32, len: usize);
        #[link_name = "fast_add_bias"]
        pub(super) fn add_bias(
            output: *mut f32,
            bias: *const f32,
            channels: usize,
            num_frames: usize,
        );
        #[link_name = "fast_gated_activation"]
        pub(super) fn gated_activation(
            data: *mut f32,
            rows: usize,
            bottleneck: usize,
            num_frames: usize,
            primary_type: i32,
            secondary_type: i32,
            use_fast_tanh: i32,
        );
        #[link_name = "fast_blended_activation"]
        pub(super) fn blended_activation(
            data: *mut f32,
            rows: usize,
            bottleneck: usize,
            num_frames: usize,
            primary_type: i32,
            secondary_type: i32,
            use_fast_tanh: i32,
        );
        #[link_name = "fast_activation_inplace"]
        pub(super) fn activation_inplace(
            data: *mut f32,
            len: usize,
            activation_type: i32,
            use_fast_tanh: i32,
        );
        #[link_name = "fast_tanh_inplace"]
        pub(super) fn tanh_inplace(data: *mut f32, len: usize);
        #[link_name = "fast_tanh_poly_inplace"]
        pub(super) fn tanh_poly_inplace(data: *mut f32, len: usize);
    }
}

fn product(left: usize, right: usize, label: &str) -> usize {
    left.checked_mul(right)
        .unwrap_or_else(|| panic!("{label} dimensions overflow usize"))
}

fn strided_len(stride: usize, width: usize, frames: usize, label: &str) -> usize {
    assert!(stride >= width, "{label} stride is smaller than its width");
    if frames == 0 {
        return 0;
    }
    product(frames - 1, stride, label)
        .checked_add(width)
        .unwrap_or_else(|| panic!("{label} dimensions overflow usize"))
}

fn require_len<T>(slice: &[T], required: usize, label: &str) {
    assert!(
        slice.len() >= required,
        "{label} requires {required} elements, received {}",
        slice.len()
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1x1_small(
    output: &mut [f32],
    weights: &[f32],
    input: &[f32],
    bias: Option<&[f32]>,
    out_channels: usize,
    in_channels: usize,
    input_stride: usize,
    num_frames: usize,
) {
    require_len(
        output,
        product(out_channels, num_frames, "conv1x1 output"),
        "conv1x1 output",
    );
    require_len(
        weights,
        product(out_channels, in_channels, "conv1x1 weights"),
        "conv1x1 weights",
    );
    require_len(
        input,
        strided_len(input_stride, in_channels, num_frames, "conv1x1 input"),
        "conv1x1 input",
    );
    if let Some(bias) = bias {
        require_len(bias, out_channels, "conv1x1 bias");
    }
    let bias = bias.map_or(core::ptr::null(), <[f32]>::as_ptr);
    // SAFETY: All pointer ranges were validated above, and mutable output is
    // exclusively borrowed for the duration of the call.
    unsafe {
        ffi::conv1x1_small(
            output.as_mut_ptr(),
            weights.as_ptr(),
            input.as_ptr(),
            bias,
            out_channels,
            in_channels,
            input_stride,
            num_frames,
        );
    }
}

pub(crate) fn conv1d_depthwise(
    output: &mut [f32],
    taps: &[&[f32]],
    weights: &[f32],
    bias: &[f32],
    channels: usize,
    num_frames: usize,
) {
    let kernel_size = taps.len();
    let frame_elements = product(channels, num_frames, "depthwise input");
    for tap in taps {
        require_len(tap, frame_elements, "depthwise input tap");
    }
    require_len(output, frame_elements, "depthwise output");
    require_len(
        weights,
        product(channels, kernel_size, "depthwise weights"),
        "depthwise weights",
    );
    require_len(bias, channels, "depthwise bias");
    let tap_ptrs = taps.iter().map(|tap| tap.as_ptr()).collect::<Vec<_>>();
    // SAFETY: Every tap and all dense buffers were validated above. The pointer
    // array remains alive and immutable for the duration of the call.
    unsafe {
        ffi::conv1d_depthwise(
            output.as_mut_ptr(),
            tap_ptrs.as_ptr(),
            weights.as_ptr(),
            bias.as_ptr(),
            channels,
            kernel_size,
            num_frames,
        );
    }
}

pub(crate) fn conv1d_small_gemv(
    output: &mut [f32],
    taps: &[&[f32]],
    weights: &[f32],
    bias: &[f32],
    out_channels: usize,
    in_channels: usize,
    num_frames: usize,
) {
    let kernel_size = taps.len();
    for tap in taps {
        require_len(
            tap,
            product(in_channels, num_frames, "conv1d input"),
            "conv1d input tap",
        );
    }
    require_len(
        output,
        product(out_channels, num_frames, "conv1d output"),
        "conv1d output",
    );
    let matrix_elements = product(out_channels, in_channels, "conv1d weights");
    require_len(
        weights,
        product(matrix_elements, kernel_size, "conv1d weights"),
        "conv1d weights",
    );
    require_len(bias, out_channels, "conv1d bias");
    let tap_ptrs = taps.iter().map(|tap| tap.as_ptr()).collect::<Vec<_>>();
    // SAFETY: Every pointer range and matrix dimension was validated above.
    unsafe {
        ffi::conv1d_small_gemv(
            output.as_mut_ptr(),
            tap_ptrs.as_ptr(),
            weights.as_ptr(),
            bias.as_ptr(),
            out_channels,
            in_channels,
            kernel_size,
            num_frames,
        );
    }
}

pub(crate) fn add_activate(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    len: usize,
    use_fast_tanh: bool,
) {
    require_len(output, len, "activation output");
    require_len(left, len, "activation left input");
    require_len(right, len, "activation right input");
    // SAFETY: The three disjoint borrows cover `len` elements.
    unsafe {
        ffi::add_activate(
            output.as_mut_ptr(),
            left.as_ptr(),
            right.as_ptr(),
            len,
            i32::from(use_fast_tanh),
        );
    }
}

pub(crate) fn vec_add(output: &mut [f32], left: &[f32], right: &[f32], len: usize) {
    require_len(output, len, "vector output");
    require_len(left, len, "vector left input");
    require_len(right, len, "vector right input");
    // SAFETY: The three disjoint borrows cover `len` elements.
    unsafe { ffi::vec_add(output.as_mut_ptr(), left.as_ptr(), right.as_ptr(), len) }
}

pub(crate) fn vec_add_inplace(left: &mut [f32], right: &[f32], len: usize) {
    require_len(left, len, "vector destination");
    require_len(right, len, "vector input");
    // SAFETY: Both disjoint borrows cover `len` elements.
    unsafe { ffi::vec_add_inplace(left.as_mut_ptr(), right.as_ptr(), len) }
}

#[allow(dead_code)]
pub(crate) fn add_bias(output: &mut [f32], bias: &[f32], channels: usize, num_frames: usize) {
    require_len(
        output,
        product(channels, num_frames, "bias output"),
        "bias output",
    );
    require_len(bias, channels, "bias");
    // SAFETY: Both buffers cover the validated matrix dimensions.
    unsafe { ffi::add_bias(output.as_mut_ptr(), bias.as_ptr(), channels, num_frames) }
}

fn film_lengths(
    data: &[f32],
    data_stride: usize,
    scale: &[f32],
    scale_rows: usize,
    dim: usize,
    scale_width: usize,
    num_frames: usize,
) {
    require_len(
        data,
        strided_len(data_stride, dim, num_frames, "FiLM data"),
        "FiLM data",
    );
    require_len(
        scale,
        strided_len(scale_rows, scale_width, num_frames, "FiLM scale"),
        "FiLM scale",
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn film_scale_shift(
    output: &mut [f32],
    input: &[f32],
    scale_shift: &[f32],
    dim: usize,
    input_stride: usize,
    output_stride: usize,
    scale_shift_rows: usize,
    num_frames: usize,
) {
    film_lengths(
        input,
        input_stride,
        scale_shift,
        scale_shift_rows,
        dim,
        product(dim, 2, "FiLM scale and shift"),
        num_frames,
    );
    require_len(
        output,
        strided_len(output_stride, dim, num_frames, "FiLM output"),
        "FiLM output",
    );
    // SAFETY: All strided matrix extents were validated above.
    unsafe {
        ffi::film_scale_shift(
            output.as_mut_ptr(),
            input.as_ptr(),
            scale_shift.as_ptr(),
            dim,
            input_stride,
            output_stride,
            scale_shift_rows,
            num_frames,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn film_scale(
    output: &mut [f32],
    input: &[f32],
    scale: &[f32],
    dim: usize,
    input_stride: usize,
    output_stride: usize,
    scale_rows: usize,
    num_frames: usize,
) {
    film_lengths(input, input_stride, scale, scale_rows, dim, dim, num_frames);
    require_len(
        output,
        strided_len(output_stride, dim, num_frames, "FiLM output"),
        "FiLM output",
    );
    // SAFETY: All strided matrix extents were validated above.
    unsafe {
        ffi::film_scale(
            output.as_mut_ptr(),
            input.as_ptr(),
            scale.as_ptr(),
            dim,
            input_stride,
            output_stride,
            scale_rows,
            num_frames,
        );
    }
}

pub(crate) fn film_inplace_scale_shift(
    data: &mut [f32],
    scale_shift: &[f32],
    dim: usize,
    data_stride: usize,
    scale_shift_rows: usize,
    num_frames: usize,
) {
    film_lengths(
        data,
        data_stride,
        scale_shift,
        scale_shift_rows,
        dim,
        product(dim, 2, "FiLM scale and shift"),
        num_frames,
    );
    // SAFETY: The mutable data and scale matrix extents were validated above.
    unsafe {
        ffi::film_inplace_scale_shift(
            data.as_mut_ptr(),
            scale_shift.as_ptr(),
            dim,
            data_stride,
            scale_shift_rows,
            num_frames,
        );
    }
}

pub(crate) fn film_inplace_scale(
    data: &mut [f32],
    scale: &[f32],
    dim: usize,
    data_stride: usize,
    scale_rows: usize,
    num_frames: usize,
) {
    film_lengths(data, data_stride, scale, scale_rows, dim, dim, num_frames);
    // SAFETY: The mutable data and scale matrix extents were validated above.
    unsafe {
        ffi::film_inplace_scale(
            data.as_mut_ptr(),
            scale.as_ptr(),
            dim,
            data_stride,
            scale_rows,
            num_frames,
        );
    }
}

pub(crate) fn gated_activation(
    data: &mut [f32],
    rows: usize,
    bottleneck: usize,
    num_frames: usize,
    primary_type: i32,
    secondary_type: i32,
    use_fast_tanh: bool,
) {
    assert!(
        rows >= product(bottleneck, 2, "gated activation"),
        "gated activation requires two bottleneck regions"
    );
    require_len(
        data,
        product(rows, num_frames, "gated activation"),
        "gated activation data",
    );
    // SAFETY: The matrix extent and both bottleneck regions were validated above.
    unsafe {
        ffi::gated_activation(
            data.as_mut_ptr(),
            rows,
            bottleneck,
            num_frames,
            primary_type,
            secondary_type,
            i32::from(use_fast_tanh),
        );
    }
}

pub(crate) fn blended_activation(
    data: &mut [f32],
    rows: usize,
    bottleneck: usize,
    num_frames: usize,
    primary_type: i32,
    secondary_type: i32,
    use_fast_tanh: bool,
) {
    assert!(
        rows >= product(bottleneck, 2, "blended activation"),
        "blended activation requires two bottleneck regions"
    );
    require_len(
        data,
        product(rows, num_frames, "blended activation"),
        "blended activation data",
    );
    // SAFETY: The matrix extent and both bottleneck regions were validated above.
    unsafe {
        ffi::blended_activation(
            data.as_mut_ptr(),
            rows,
            bottleneck,
            num_frames,
            primary_type,
            secondary_type,
            i32::from(use_fast_tanh),
        );
    }
}

pub(crate) fn activation_inplace(
    data: &mut [f32],
    len: usize,
    activation_type: i32,
    use_fast_tanh: bool,
) {
    require_len(data, len, "activation data");
    // SAFETY: The mutable slice covers `len` initialized elements.
    unsafe {
        ffi::activation_inplace(
            data.as_mut_ptr(),
            len,
            activation_type,
            i32::from(use_fast_tanh),
        );
    }
}

#[allow(dead_code)]
pub(crate) fn tanh_inplace(data: &mut [f32], len: usize) {
    require_len(data, len, "tanh data");
    // SAFETY: The mutable slice covers `len` initialized elements.
    unsafe { ffi::tanh_inplace(data.as_mut_ptr(), len) }
}

#[allow(dead_code)]
pub(crate) fn tanh_poly_inplace(data: &mut [f32], len: usize) {
    require_len(data, len, "polynomial tanh data");
    // SAFETY: The mutable slice covers `len` initialized elements.
    unsafe { ffi::tanh_poly_inplace(data.as_mut_ptr(), len) }
}

#[cfg(all(test, feature = "fast-kernels"))]
mod tests {
    use proptest::prelude::*;

    #[test]
    #[should_panic(expected = "conv1x1 output requires")]
    fn safe_facade_rejects_short_output_buffer() {
        super::conv1x1_small(&mut [], &[1.0], &[1.0], None, 1, 1, 1, 1);
    }

    #[test]
    #[should_panic(expected = "FiLM data stride is smaller")]
    fn safe_facade_rejects_invalid_stride() {
        super::film_scale(&mut [0.0; 2], &[1.0; 2], &[1.0; 2], 2, 1, 2, 2, 1);
    }

    proptest! {
        #[test]
        fn backend_equivalence_fused_add_activation_matches_scalar_rust(
            left in prop::collection::vec(-3.0f32..3.0, 0..256),
            right_seed in prop::collection::vec(-3.0f32..3.0, 1..256),
        ) {
            let right = right_seed
                .into_iter()
                .cycle()
                .take(left.len())
                .collect::<Vec<_>>();
            let expected = left
                .iter()
                .zip(&right)
                .map(|(&left, &right)| (left + right).tanh())
                .collect::<Vec<_>>();
            let mut actual = vec![0.0f32; left.len()];

            super::add_activate(&mut actual, &left, &right, left.len(), false);

            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                prop_assert!(
                    (actual - expected).abs() <= 2.0e-6,
                    "activation output {index}: expected {expected}, got {actual}"
                );
            }
        }

        #[test]
        fn backend_equivalence_small_c_gemm_matches_scalar_rust(
            out_channels in 1usize..8,
            in_channels in 1usize..8,
            frames in 1usize..24,
            padding in 0usize..4,
            raw_weights in prop::collection::vec(-1.0f32..1.0, 1..64),
            raw_input in prop::collection::vec(-1.0f32..1.0, 1..256),
            raw_bias in prop::collection::vec(-1.0f32..1.0, 1..8),
        ) {
            let stride = in_channels + padding;
            let weights = raw_weights
                .into_iter()
                .cycle()
                .take(out_channels * in_channels)
                .collect::<Vec<_>>();
            let input = raw_input
                .into_iter()
                .cycle()
                .take(stride * frames)
                .collect::<Vec<_>>();
            let bias = raw_bias
                .into_iter()
                .cycle()
                .take(out_channels)
                .collect::<Vec<_>>();
            let mut actual = vec![0.0f32; out_channels * frames];
            let mut expected = vec![0.0f32; out_channels * frames];

            for frame in 0..frames {
                for output_channel in 0..out_channels {
                    let mut sum = bias[output_channel];
                    for input_channel in 0..in_channels {
                        sum += weights[input_channel * out_channels + output_channel]
                            * input[frame * stride + input_channel];
                    }
                    expected[frame * out_channels + output_channel] = sum;
                }
            }

            super::conv1x1_small(
                &mut actual,
                &weights,
                &input,
                Some(&bias),
                out_channels,
                in_channels,
                stride,
                frames,
            );

            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                prop_assert!(
                    (actual - expected).abs() <= 2.0e-5,
                    "GEMM output {index}: expected {expected}, got {actual}"
                );
            }
        }
    }
}
