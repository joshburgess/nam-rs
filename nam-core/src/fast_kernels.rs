//! Validate every extent before crossing into the C kernels
//!
//! Invalid layouts are no-ops. Valid calls pass exclusive output ranges and
//! initialized input ranges matching the C `restrict` and size contracts.

mod ffi {
    #[allow(dead_code)]
    extern "C" {
        #[link_name = "fast_conv1d_depthwise"]
        pub(super) fn conv1d_depthwise(
            output: *mut f32,
            input: *const f32,
            tap_offsets: *const usize,
            weights: *const f32,
            bias: *const f32,
            ch: usize,
            kernel_size: usize,
            num_frames: usize,
        );
        #[link_name = "fast_conv1d_small_gemv"]
        pub(super) fn conv1d_small_gemv(
            output: *mut f32,
            input: *const f32,
            tap_offsets: *const usize,
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

pub(crate) struct Conv1dDimensions {
    pub(crate) out_channels: usize,
    pub(crate) in_channels: usize,
    pub(crate) num_frames: usize,
}

fn product(left: usize, right: usize) -> usize {
    left.saturating_mul(right)
}

fn strided_len(stride: usize, width: usize, frames: usize) -> usize {
    if stride < width {
        return usize::MAX;
    }
    if frames == 0 {
        return 0;
    }
    product(frames - 1, stride).saturating_add(width)
}

fn has_len<T>(slice: &[T], required: usize) -> bool {
    slice.len() >= required
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
    if !has_len(output, product(out_channels, num_frames))
        || !has_len(weights, product(out_channels, in_channels))
        || !has_len(input, strided_len(input_stride, in_channels, num_frames))
        || bias.is_some_and(|bias| !has_len(bias, out_channels))
    {
        return;
    }
    let bias = bias.map_or(core::ptr::null(), <[f32]>::as_ptr);
    // SAFETY: The validated extents cover every C access. Rust's mutable output
    // borrow cannot alias the shared inputs required by the C `restrict` contract
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
    input: &[f32],
    tap_offsets: &[usize],
    weights: &[f32],
    bias: &[f32],
    channels: usize,
    num_frames: usize,
) {
    let kernel_size = tap_offsets.len();
    let frame_elements = product(channels, num_frames);
    if tap_offsets.iter().any(|offset| {
        offset
            .checked_add(frame_elements)
            .is_none_or(|end| end > input.len())
    }) || !has_len(output, frame_elements)
        || !has_len(weights, product(channels, kernel_size))
        || !has_len(bias, channels)
    {
        return;
    }
    // SAFETY: Every offset and dense buffer covers the C loop bounds, and mutable
    // output cannot alias the immutable ring-buffer input
    unsafe {
        ffi::conv1d_depthwise(
            output.as_mut_ptr(),
            input.as_ptr(),
            tap_offsets.as_ptr(),
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
    input: &[f32],
    tap_offsets: &[usize],
    weights: &[f32],
    bias: &[f32],
    dimensions: Conv1dDimensions,
) {
    let Conv1dDimensions {
        out_channels,
        in_channels,
        num_frames,
    } = dimensions;
    let kernel_size = tap_offsets.len();
    let input_elements = product(in_channels, num_frames);
    let matrix_elements = product(out_channels, in_channels);
    if tap_offsets.iter().any(|offset| {
        offset
            .checked_add(input_elements)
            .is_none_or(|end| end > input.len())
    }) || !has_len(output, product(out_channels, num_frames))
        || !has_len(weights, product(matrix_elements, kernel_size))
        || !has_len(bias, out_channels)
    {
        return;
    }
    // SAFETY: Every offset covers the validated matrix dimensions, and mutable
    // output cannot alias the input, weight, or bias buffers
    unsafe {
        ffi::conv1d_small_gemv(
            output.as_mut_ptr(),
            input.as_ptr(),
            tap_offsets.as_ptr(),
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
    if !has_len(output, len) || !has_len(left, len) || !has_len(right, len) {
        return;
    }
    // SAFETY: All slices cover `len`, and Rust's mutable borrow makes the output
    // disjoint from both inputs as required by C `restrict`
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
    if !has_len(output, len) || !has_len(left, len) || !has_len(right, len) {
        return;
    }
    // SAFETY: All slices cover `len`, and mutable output cannot alias either input
    unsafe { ffi::vec_add(output.as_mut_ptr(), left.as_ptr(), right.as_ptr(), len) }
}

pub(crate) fn vec_add_inplace(left: &mut [f32], right: &[f32], len: usize) {
    if !has_len(left, len) || !has_len(right, len) {
        return;
    }
    // SAFETY: Both slices cover `len`, and mutable `left` cannot alias `right`
    unsafe { ffi::vec_add_inplace(left.as_mut_ptr(), right.as_ptr(), len) }
}

#[allow(dead_code)]
pub(crate) fn add_bias(output: &mut [f32], bias: &[f32], channels: usize, num_frames: usize) {
    if !has_len(output, product(channels, num_frames)) || !has_len(bias, channels) {
        return;
    }
    // SAFETY: Both buffers cover the matrix dimensions, and mutable output cannot alias bias
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
) -> bool {
    has_len(data, strided_len(data_stride, dim, num_frames))
        && has_len(scale, strided_len(scale_rows, scale_width, num_frames))
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
    if !film_lengths(
        input,
        input_stride,
        scale_shift,
        scale_shift_rows,
        dim,
        product(dim, 2),
        num_frames,
    ) || !has_len(output, strided_len(output_stride, dim, num_frames))
    {
        return;
    }
    // SAFETY: Every strided range covers the C loop bounds, and mutable output
    // cannot alias the input or scale-and-shift parameters
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
    if !film_lengths(input, input_stride, scale, scale_rows, dim, dim, num_frames)
        || !has_len(output, strided_len(output_stride, dim, num_frames))
    {
        return;
    }
    // SAFETY: Every strided range covers the C loop bounds, and mutable output
    // cannot alias the input or scale parameters
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
    if !film_lengths(
        data,
        data_stride,
        scale_shift,
        scale_shift_rows,
        dim,
        product(dim, 2),
        num_frames,
    ) {
        return;
    }
    // SAFETY: Both strided ranges cover the C loop bounds, and mutable data cannot
    // alias the scale-and-shift parameters
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
    if !film_lengths(data, data_stride, scale, scale_rows, dim, dim, num_frames) {
        return;
    }
    // SAFETY: Both strided ranges cover the C loop bounds, and mutable data cannot
    // alias the scale parameters
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
    if rows < product(bottleneck, 2) || !has_len(data, product(rows, num_frames)) {
        return;
    }
    // SAFETY: Data covers every frame and both bottleneck regions used by C
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
    if rows < product(bottleneck, 2) || !has_len(data, product(rows, num_frames)) {
        return;
    }
    // SAFETY: Data covers every frame and both bottleneck regions used by C
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
    if !has_len(data, len) {
        return;
    }
    // SAFETY: Data contains `len` initialized elements under an exclusive borrow
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
    if !has_len(data, len) {
        return;
    }
    // SAFETY: Data contains `len` initialized elements under an exclusive borrow
    unsafe { ffi::tanh_inplace(data.as_mut_ptr(), len) }
}

#[allow(dead_code)]
pub(crate) fn tanh_poly_inplace(data: &mut [f32], len: usize) {
    if !has_len(data, len) {
        return;
    }
    // SAFETY: Data contains `len` initialized elements under an exclusive borrow
    unsafe { ffi::tanh_poly_inplace(data.as_mut_ptr(), len) }
}

#[cfg(all(test, feature = "fast-kernels"))]
mod tests {
    use proptest::prelude::*;

    #[test]
    fn safe_facade_rejects_short_output_buffer() {
        super::conv1x1_small(&mut [], &[1.0], &[1.0], None, 1, 1, 1, 1);
    }

    #[test]
    fn safe_facade_rejects_invalid_stride() {
        let mut output = [7.0; 2];
        super::film_scale(&mut output, &[1.0; 2], &[1.0; 2], 2, 1, 2, 2, 1);
        assert_eq!(output, [7.0; 2]);
    }

    #[test]
    fn safe_facade_rejects_short_depthwise_taps() {
        let mut output = [7.0; 2];
        super::conv1d_depthwise(&mut output, &[1.0], &[0], &[1.0; 2], &[0.0; 2], 2, 1);
        assert_eq!(output, [7.0; 2]);
    }

    #[test]
    fn safe_facade_rejects_short_film_parameters() {
        let mut output = [7.0; 4];
        super::film_scale_shift(&mut output, &[1.0; 4], &[1.0; 3], 2, 2, 2, 4, 2);
        assert_eq!(output, [7.0; 4]);
    }

    #[test]
    fn safe_facade_rejects_overflowed_dimensions() {
        let mut output = [7.0];
        super::conv1x1_small(&mut output, &[1.0], &[1.0], None, usize::MAX, 2, 2, 1);
        assert_eq!(output, [7.0]);

        let mut activation = [7.0];
        super::gated_activation(&mut activation, usize::MAX, usize::MAX, 1, 0, 6, false);
        assert_eq!(activation, [7.0]);
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
