//! Validate every extent before crossing into the C kernels
//!
//! Invalid layouts are no-ops. Valid calls pass exclusive output ranges and
//! initialized input ranges matching the C `restrict` and size contracts.

mod ffi {
    #[allow(dead_code)]
    extern "C" {
        #[link_name = "fast_init_vector_math"]
        pub(super) fn init_vector_math() -> i32;
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
        #[link_name = "fast_conv1d_grouped_12x3_k2"]
        pub(super) fn conv1d_grouped_12x3_k2(
            output: *mut f32,
            input: *const f32,
            tap_offsets: *const usize,
            weights: *const f32,
            bias: *const f32,
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
        #[link_name = "fast_film_rank1_scale_shift"]
        pub(super) fn film_rank1_scale_shift(
            output: *mut f32,
            input: *const f32,
            condition: *const f32,
            weights: *const f32,
            bias: *const f32,
            dim: usize,
            input_stride: usize,
            output_stride: usize,
            condition_stride: usize,
            num_frames: usize,
        );
        #[link_name = "fast_film_rank1_scale"]
        pub(super) fn film_rank1_scale(
            output: *mut f32,
            input: *const f32,
            condition: *const f32,
            weights: *const f32,
            bias: *const f32,
            dim: usize,
            input_stride: usize,
            output_stride: usize,
            condition_stride: usize,
            num_frames: usize,
        );
        #[link_name = "fast_film_rank1_inplace_scale_shift"]
        pub(super) fn film_rank1_inplace_scale_shift(
            data: *mut f32,
            condition: *const f32,
            weights: *const f32,
            bias: *const f32,
            dim: usize,
            data_stride: usize,
            condition_stride: usize,
            num_frames: usize,
        );
        #[link_name = "fast_film_rank1_inplace_scale"]
        pub(super) fn film_rank1_inplace_scale(
            data: *mut f32,
            condition: *const f32,
            weights: *const f32,
            bias: *const f32,
            dim: usize,
            data_stride: usize,
            condition_stride: usize,
            num_frames: usize,
        );
        #[link_name = "fast_film_8x8_scale_shift"]
        pub(super) fn film_8x8_scale_shift(
            output: *mut f32,
            input: *const f32,
            condition: *const f32,
            weights: *const f32,
            bias: *const f32,
            input_stride: usize,
            output_stride: usize,
            condition_stride: usize,
            num_frames: usize,
        );
        #[link_name = "fast_film_8x8_scale"]
        pub(super) fn film_8x8_scale(
            output: *mut f32,
            input: *const f32,
            condition: *const f32,
            weights: *const f32,
            bias: *const f32,
            input_stride: usize,
            output_stride: usize,
            condition_stride: usize,
            num_frames: usize,
        );
        #[link_name = "fast_film_8x8_inplace_scale_shift"]
        pub(super) fn film_8x8_inplace_scale_shift(
            data: *mut f32,
            condition: *const f32,
            weights: *const f32,
            bias: *const f32,
            data_stride: usize,
            condition_stride: usize,
            num_frames: usize,
        );
        #[link_name = "fast_film_8x8_inplace_scale"]
        pub(super) fn film_8x8_inplace_scale(
            data: *mut f32,
            condition: *const f32,
            weights: *const f32,
            bias: *const f32,
            data_stride: usize,
            condition_stride: usize,
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

pub(crate) fn init_vector_math() -> bool {
    // SAFETY: The C initializer is process-global, idempotent, and takes no arguments
    unsafe { ffi::init_vector_math() != 0 }
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

pub(crate) fn conv1d_grouped_12x3_k2(
    output: &mut [f32],
    input: &[f32],
    tap_offsets: &[usize],
    weights: &[f32],
    bias: &[f32],
    num_frames: usize,
) -> bool {
    const OUT_CHANNELS: usize = 12;
    const IN_CHANNELS: usize = 3;
    const KERNEL_SIZE: usize = 2;

    let input_elements = product(IN_CHANNELS, num_frames);
    if tap_offsets.len() != KERNEL_SIZE
        || tap_offsets.iter().any(|offset| {
            offset
                .checked_add(input_elements)
                .is_none_or(|end| end > input.len())
        })
        || !has_len(output, product(OUT_CHANNELS, num_frames))
        || !has_len(weights, OUT_CHANNELS * KERNEL_SIZE)
        || !has_len(bias, OUT_CHANNELS)
    {
        return false;
    }
    // SAFETY: Both tap ranges and every fixed-size output, weight, and bias
    // access are covered by the validated slice extents
    unsafe {
        ffi::conv1d_grouped_12x3_k2(
            output.as_mut_ptr(),
            input.as_ptr(),
            tap_offsets.as_ptr(),
            weights.as_ptr(),
            bias.as_ptr(),
            num_frames,
        );
    }
    true
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
pub(crate) fn film_rank1_scale_shift(
    output: &mut [f32],
    input: &[f32],
    condition: &[f32],
    weights: &[f32],
    bias: &[f32],
    dim: usize,
    input_stride: usize,
    output_stride: usize,
    condition_stride: usize,
    num_frames: usize,
) {
    let parameters = product(dim, 2);
    if !has_len(input, strided_len(input_stride, dim, num_frames))
        || !has_len(output, strided_len(output_stride, dim, num_frames))
        || !has_len(condition, strided_len(condition_stride, 1, num_frames))
        || !has_len(weights, parameters)
        || !has_len(bias, parameters)
    {
        return;
    }
    // SAFETY: All dense and strided ranges cover the C loop bounds. Mutable
    // output cannot alias the input, condition, weights, or bias.
    unsafe {
        ffi::film_rank1_scale_shift(
            output.as_mut_ptr(),
            input.as_ptr(),
            condition.as_ptr(),
            weights.as_ptr(),
            bias.as_ptr(),
            dim,
            input_stride,
            output_stride,
            condition_stride,
            num_frames,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn film_rank1_scale(
    output: &mut [f32],
    input: &[f32],
    condition: &[f32],
    weights: &[f32],
    bias: &[f32],
    dim: usize,
    input_stride: usize,
    output_stride: usize,
    condition_stride: usize,
    num_frames: usize,
) {
    if !has_len(input, strided_len(input_stride, dim, num_frames))
        || !has_len(output, strided_len(output_stride, dim, num_frames))
        || !has_len(condition, strided_len(condition_stride, 1, num_frames))
        || !has_len(weights, dim)
        || !has_len(bias, dim)
    {
        return;
    }
    // SAFETY: All dense and strided ranges cover the C loop bounds. Mutable
    // output cannot alias the input, condition, weights, or bias.
    unsafe {
        ffi::film_rank1_scale(
            output.as_mut_ptr(),
            input.as_ptr(),
            condition.as_ptr(),
            weights.as_ptr(),
            bias.as_ptr(),
            dim,
            input_stride,
            output_stride,
            condition_stride,
            num_frames,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn film_rank1_inplace_scale_shift(
    data: &mut [f32],
    condition: &[f32],
    weights: &[f32],
    bias: &[f32],
    dim: usize,
    data_stride: usize,
    condition_stride: usize,
    num_frames: usize,
) {
    let parameters = product(dim, 2);
    if !has_len(data, strided_len(data_stride, dim, num_frames))
        || !has_len(condition, strided_len(condition_stride, 1, num_frames))
        || !has_len(weights, parameters)
        || !has_len(bias, parameters)
    {
        return;
    }
    // SAFETY: Data covers every in-place element, and the shared parameters
    // are disjoint from its exclusive borrow.
    unsafe {
        ffi::film_rank1_inplace_scale_shift(
            data.as_mut_ptr(),
            condition.as_ptr(),
            weights.as_ptr(),
            bias.as_ptr(),
            dim,
            data_stride,
            condition_stride,
            num_frames,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn film_rank1_inplace_scale(
    data: &mut [f32],
    condition: &[f32],
    weights: &[f32],
    bias: &[f32],
    dim: usize,
    data_stride: usize,
    condition_stride: usize,
    num_frames: usize,
) {
    if !has_len(data, strided_len(data_stride, dim, num_frames))
        || !has_len(condition, strided_len(condition_stride, 1, num_frames))
        || !has_len(weights, dim)
        || !has_len(bias, dim)
    {
        return;
    }
    // SAFETY: Data covers every in-place element, and the shared parameters
    // are disjoint from its exclusive borrow.
    unsafe {
        ffi::film_rank1_inplace_scale(
            data.as_mut_ptr(),
            condition.as_ptr(),
            weights.as_ptr(),
            bias.as_ptr(),
            dim,
            data_stride,
            condition_stride,
            num_frames,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn film_8x8_scale_shift(
    output: &mut [f32],
    input: &[f32],
    condition: &[f32],
    weights: &[f32],
    bias: &[f32],
    input_stride: usize,
    output_stride: usize,
    condition_stride: usize,
    num_frames: usize,
) {
    const DIM: usize = 8;
    const PARAMETERS: usize = DIM * 2;
    if !has_len(input, strided_len(input_stride, DIM, num_frames))
        || !has_len(output, strided_len(output_stride, DIM, num_frames))
        || !has_len(condition, strided_len(condition_stride, DIM, num_frames))
        || !has_len(weights, PARAMETERS * DIM)
        || !has_len(bias, PARAMETERS)
    {
        return;
    }
    // SAFETY: All fixed 8-wide ranges cover the C loop bounds. Mutable output
    // cannot alias the input or parameters.
    unsafe {
        ffi::film_8x8_scale_shift(
            output.as_mut_ptr(),
            input.as_ptr(),
            condition.as_ptr(),
            weights.as_ptr(),
            bias.as_ptr(),
            input_stride,
            output_stride,
            condition_stride,
            num_frames,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn film_8x8_scale(
    output: &mut [f32],
    input: &[f32],
    condition: &[f32],
    weights: &[f32],
    bias: &[f32],
    input_stride: usize,
    output_stride: usize,
    condition_stride: usize,
    num_frames: usize,
) {
    const DIM: usize = 8;
    if !has_len(input, strided_len(input_stride, DIM, num_frames))
        || !has_len(output, strided_len(output_stride, DIM, num_frames))
        || !has_len(condition, strided_len(condition_stride, DIM, num_frames))
        || !has_len(weights, DIM * DIM)
        || !has_len(bias, DIM)
    {
        return;
    }
    // SAFETY: All fixed 8-wide ranges cover the C loop bounds. Mutable output
    // cannot alias the input or parameters.
    unsafe {
        ffi::film_8x8_scale(
            output.as_mut_ptr(),
            input.as_ptr(),
            condition.as_ptr(),
            weights.as_ptr(),
            bias.as_ptr(),
            input_stride,
            output_stride,
            condition_stride,
            num_frames,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn film_8x8_inplace_scale_shift(
    data: &mut [f32],
    condition: &[f32],
    weights: &[f32],
    bias: &[f32],
    data_stride: usize,
    condition_stride: usize,
    num_frames: usize,
) {
    const DIM: usize = 8;
    const PARAMETERS: usize = DIM * 2;
    if !has_len(data, strided_len(data_stride, DIM, num_frames))
        || !has_len(condition, strided_len(condition_stride, DIM, num_frames))
        || !has_len(weights, PARAMETERS * DIM)
        || !has_len(bias, PARAMETERS)
    {
        return;
    }
    // SAFETY: Data covers every fixed-width in-place element, and the shared
    // parameters are disjoint from its exclusive borrow.
    unsafe {
        ffi::film_8x8_inplace_scale_shift(
            data.as_mut_ptr(),
            condition.as_ptr(),
            weights.as_ptr(),
            bias.as_ptr(),
            data_stride,
            condition_stride,
            num_frames,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn film_8x8_inplace_scale(
    data: &mut [f32],
    condition: &[f32],
    weights: &[f32],
    bias: &[f32],
    data_stride: usize,
    condition_stride: usize,
    num_frames: usize,
) {
    const DIM: usize = 8;
    if !has_len(data, strided_len(data_stride, DIM, num_frames))
        || !has_len(condition, strided_len(condition_stride, DIM, num_frames))
        || !has_len(weights, DIM * DIM)
        || !has_len(bias, DIM)
    {
        return;
    }
    // SAFETY: Data covers every fixed-width in-place element, and the shared
    // parameters are disjoint from its exclusive borrow.
    unsafe {
        ffi::film_8x8_inplace_scale(
            data.as_mut_ptr(),
            condition.as_ptr(),
            weights.as_ptr(),
            bias.as_ptr(),
            data_stride,
            condition_stride,
            num_frames,
        );
    }
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
    fn safe_facade_rejects_short_small_conv1d_buffers() {
        let mut output = [7.0; 4];
        super::conv1d_small_gemv(
            &mut output,
            &[1.0; 16],
            &[0, 8],
            &[1.0; 32],
            &[0.0; 4],
            super::Conv1dDimensions {
                out_channels: 4,
                in_channels: 4,
                num_frames: 2,
            },
        );
        assert_eq!(output, [7.0; 4]);
    }

    #[test]
    fn safe_facade_rejects_short_grouped_conv1d_buffers() {
        let mut output = [7.0; 12];
        let processed = super::conv1d_grouped_12x3_k2(
            &mut output,
            &[1.0; 6],
            &[0, 3],
            &[1.0; 24],
            &[0.0; 12],
            2,
        );
        assert!(!processed);
        assert_eq!(output, [7.0; 12]);
    }

    #[test]
    fn grouped_12x3_k2_conv1d_matches_scalar_reference() {
        let num_frames = 19;
        let tap_elements = 3 * num_frames;
        let input = (0..tap_elements * 2)
            .map(|index| ((index + 1) as f32 * 0.017).sin())
            .collect::<Vec<_>>();
        let tap_offsets = [0, tap_elements];
        let weights = (0..24)
            .map(|index| ((index + 1) as f32 * 0.031).cos() * 0.25)
            .collect::<Vec<_>>();
        let bias = (0..12)
            .map(|index| index as f32 * 0.01 - 0.03)
            .collect::<Vec<_>>();
        let mut expected = vec![0.0f32; 12 * num_frames];

        for (tap, &tap_offset) in tap_offsets.iter().enumerate() {
            for frame in 0..num_frames {
                for group in 0..3 {
                    for output in 0..4 {
                        let output_channel = group * 4 + output;
                        expected[frame * 12 + output_channel] += weights[tap * 12 + output_channel]
                            * input[tap_offset + frame * 3 + group];
                    }
                }
            }
        }
        for frame in 0..num_frames {
            for output in 0..12 {
                expected[frame * 12 + output] += bias[output];
            }
        }

        let mut actual = vec![0.0f32; expected.len()];
        assert!(super::conv1d_grouped_12x3_k2(
            &mut actual,
            &input,
            &tap_offsets,
            &weights,
            &bias,
            num_frames,
        ));

        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 2.0e-6,
                "grouped 12x3/k2 output {index}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn specialized_small_conv1d_shapes_match_scalar_reference() {
        for (out_channels, in_channels, kernel_size) in
            [(8, 4, 4), (4, 4, 4), (4, 4, 3), (12, 3, 2)]
        {
            let num_frames = 19;
            let tap_elements = in_channels * num_frames;
            let input = (0..tap_elements * kernel_size)
                .map(|index| ((index + 1) as f32 * 0.017).sin())
                .collect::<Vec<_>>();
            let tap_offsets = (0..kernel_size)
                .map(|tap| tap * tap_elements)
                .collect::<Vec<_>>();
            let weights = (0..out_channels * in_channels * kernel_size)
                .map(|index| ((index + 1) as f32 * 0.031).cos() * 0.25)
                .collect::<Vec<_>>();
            let bias = (0..out_channels)
                .map(|index| index as f32 * 0.01 - 0.03)
                .collect::<Vec<_>>();
            let mut expected = vec![0.0f32; out_channels * num_frames];

            for (tap, &tap_offset) in tap_offsets.iter().enumerate() {
                let weight_offset = tap * out_channels * in_channels;
                for frame in 0..num_frames {
                    for output_channel in 0..out_channels {
                        let mut sum = 0.0f32;
                        for input_channel in 0..in_channels {
                            sum += weights
                                [weight_offset + input_channel * out_channels + output_channel]
                                * input[tap_offset + frame * in_channels + input_channel];
                        }
                        expected[frame * out_channels + output_channel] += sum;
                    }
                }
            }
            for frame in 0..num_frames {
                for output_channel in 0..out_channels {
                    expected[frame * out_channels + output_channel] += bias[output_channel];
                }
            }

            let mut actual = vec![0.0f32; expected.len()];
            super::conv1d_small_gemv(
                &mut actual,
                &input,
                &tap_offsets,
                &weights,
                &bias,
                super::Conv1dDimensions {
                    out_channels,
                    in_channels,
                    num_frames,
                },
            );

            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 2.0e-6,
                    "{out_channels}x{in_channels}/k{kernel_size} output {index}: \
                     expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn safe_facade_rejects_short_film_parameters() {
        let mut output = [7.0; 4];
        super::film_scale_shift(&mut output, &[1.0; 4], &[1.0; 3], 2, 2, 2, 4, 2);
        assert_eq!(output, [7.0; 4]);

        super::film_rank1_scale_shift(
            &mut output,
            &[1.0; 4],
            &[1.0; 2],
            &[1.0; 3],
            &[1.0; 4],
            2,
            2,
            2,
            1,
            2,
        );
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

    #[test]
    fn accurate_fused_tanh_handles_chunk_boundaries() {
        let _ = super::init_vector_math();
        for len in [0, 1, 255, 256, 257, 511, 512, 513, 1024] {
            let left = (0..len)
                .map(|index| (index as f32 * 0.037).sin() * 2.0)
                .collect::<Vec<_>>();
            let right = (0..len)
                .map(|index| (index as f32 * 0.021).cos() * 0.75)
                .collect::<Vec<_>>();
            let expected = left
                .iter()
                .zip(&right)
                .map(|(&left, &right)| (left + right).tanh())
                .collect::<Vec<_>>();
            let mut actual = vec![0.0; len];

            super::add_activate(&mut actual, &left, &right, len, false);

            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 2.0e-6,
                    "length {len}, output {index}: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn accurate_fused_tanh_handles_special_values() {
        let _ = super::init_vector_math();
        let left = [
            f32::NEG_INFINITY,
            -100.0,
            -20.0,
            -3.0,
            -1.0,
            -0.0,
            0.0,
            1.0,
            3.0,
            20.0,
            100.0,
            f32::INFINITY,
            f32::NAN,
        ];
        let mut right = [0.0; 13];
        right[5] = -0.0;
        let mut actual = [0.0; 13];

        super::add_activate(&mut actual, &left, &right, left.len(), false);

        for (index, ((&actual, &left), &right)) in actual.iter().zip(&left).zip(&right).enumerate()
        {
            let expected = (left + right).tanh();
            if expected.is_nan() {
                assert!(actual.is_nan(), "output {index} did not preserve NaN");
            } else {
                assert!(
                    (actual - expected).abs() <= 2.0e-6,
                    "output {index}: expected {expected}, got {actual}"
                );
            }
        }
        assert!(actual[5].is_sign_negative());
    }

    #[test]
    fn vector_math_initialization_is_idempotent() {
        let lanes = super::init_vector_math();
        assert_eq!(
            lanes,
            super::init_vector_math(),
            "vector-math availability changed after initialization"
        );
    }

    fn assert_film_close(actual: &[f32], expected: &[f32], stride: usize, dim: usize) {
        for (frame, (actual, expected)) in actual
            .chunks(stride)
            .zip(expected.chunks(stride))
            .enumerate()
        {
            for channel in 0..dim {
                assert!(
                    (actual[channel] - expected[channel]).abs() <= 2.0e-5,
                    "frame {frame}, channel {channel}: expected {}, got {}",
                    expected[channel],
                    actual[channel]
                );
            }
        }
    }

    fn verify_fused_film(in_channels: usize, dim: usize, do_shift: bool) {
        let frames = 3;
        let input_stride = dim + 2;
        let output_stride = dim + 1;
        let condition_stride = in_channels + 2;
        let parameters = if do_shift { dim * 2 } else { dim };
        let input = (0..frames * input_stride)
            .map(|index| index as f32 * 0.031 - 0.4)
            .collect::<Vec<_>>();
        let condition = (0..frames * condition_stride)
            .map(|index| index as f32 * -0.027 + 0.3)
            .collect::<Vec<_>>();
        let weights = (0..parameters * in_channels)
            .map(|index| index as f32 * 0.013 - 0.2)
            .collect::<Vec<_>>();
        let bias = (0..parameters)
            .map(|index| index as f32 * -0.019 + 0.1)
            .collect::<Vec<_>>();
        let mut expected = vec![0.0; frames * output_stride];

        for frame in 0..frames {
            for channel in 0..dim {
                let mut scale =
                    weights[channel] * condition[frame * condition_stride] + bias[channel];
                for input_channel in 1..in_channels {
                    scale += weights[input_channel * parameters + channel]
                        * condition[frame * condition_stride + input_channel];
                }
                let shift = if do_shift {
                    let mut shift = weights[dim + channel] * condition[frame * condition_stride]
                        + bias[dim + channel];
                    for input_channel in 1..in_channels {
                        shift += weights[input_channel * parameters + dim + channel]
                            * condition[frame * condition_stride + input_channel];
                    }
                    shift
                } else {
                    0.0
                };
                expected[frame * output_stride + channel] =
                    input[frame * input_stride + channel] * scale + shift;
            }
        }

        let mut actual = vec![0.0; expected.len()];
        match (in_channels, do_shift) {
            (1, true) => super::film_rank1_scale_shift(
                &mut actual,
                &input,
                &condition,
                &weights,
                &bias,
                dim,
                input_stride,
                output_stride,
                condition_stride,
                frames,
            ),
            (1, false) => super::film_rank1_scale(
                &mut actual,
                &input,
                &condition,
                &weights,
                &bias,
                dim,
                input_stride,
                output_stride,
                condition_stride,
                frames,
            ),
            (8, true) => super::film_8x8_scale_shift(
                &mut actual,
                &input,
                &condition,
                &weights,
                &bias,
                input_stride,
                output_stride,
                condition_stride,
                frames,
            ),
            (8, false) => super::film_8x8_scale(
                &mut actual,
                &input,
                &condition,
                &weights,
                &bias,
                input_stride,
                output_stride,
                condition_stride,
                frames,
            ),
            _ => panic!("unsupported fused FiLM test shape"),
        }
        assert_film_close(&actual, &expected, output_stride, dim);

        let mut inplace = input.clone();
        let mut inplace_expected = input.clone();
        for frame in 0..frames {
            inplace_expected[frame * input_stride..frame * input_stride + dim]
                .copy_from_slice(&expected[frame * output_stride..frame * output_stride + dim]);
        }
        match (in_channels, do_shift) {
            (1, true) => super::film_rank1_inplace_scale_shift(
                &mut inplace,
                &condition,
                &weights,
                &bias,
                dim,
                input_stride,
                condition_stride,
                frames,
            ),
            (1, false) => super::film_rank1_inplace_scale(
                &mut inplace,
                &condition,
                &weights,
                &bias,
                dim,
                input_stride,
                condition_stride,
                frames,
            ),
            (8, true) => super::film_8x8_inplace_scale_shift(
                &mut inplace,
                &condition,
                &weights,
                &bias,
                input_stride,
                condition_stride,
                frames,
            ),
            (8, false) => super::film_8x8_inplace_scale(
                &mut inplace,
                &condition,
                &weights,
                &bias,
                input_stride,
                condition_stride,
                frames,
            ),
            _ => panic!("unsupported fused FiLM test shape"),
        }
        assert_film_close(&inplace, &inplace_expected, input_stride, dim);
    }

    #[test]
    fn fused_film_matches_two_stage_reference() {
        for (in_channels, dim) in [(1, 6), (8, 8)] {
            for do_shift in [false, true] {
                verify_fused_film(in_channels, dim, do_shift);
            }
        }
    }

    #[cfg(all(target_os = "linux", target_env = "gnu", target_arch = "x86_64"))]
    #[test]
    fn vector_math_availability_matches_glibc_version() {
        use std::ffi::{c_char, CStr};

        unsafe extern "C" {
            fn gnu_get_libc_version() -> *const c_char;
        }

        // SAFETY: glibc returns a process-lifetime, null-terminated version string
        let version = unsafe { CStr::from_ptr(gnu_get_libc_version()) }
            .to_str()
            .expect("glibc version was not UTF-8");
        let mut components = version.split('.');
        let major = components
            .next()
            .expect("glibc version had no major component")
            .parse::<u32>()
            .expect("glibc major version was not numeric");
        let minor = components
            .next()
            .expect("glibc version had no minor component")
            .parse::<u32>()
            .expect("glibc minor version was not numeric");

        assert_eq!(super::init_vector_math(), (major, minor) >= (2, 35));
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
            in_channels in 1usize..9,
            frames in 1usize..24,
            padding in 0usize..4,
            raw_weights in prop::collection::vec(-1.0f32..1.0, 1..64),
            raw_input in prop::collection::vec(-1.0f32..1.0, 1..256),
            raw_bias in prop::collection::vec(-1.0f32..1.0, 1..8),
            use_bias in any::<bool>(),
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
                    let mut sum = weights[output_channel] * input[frame * stride];
                    for input_channel in 1..in_channels {
                        sum += weights[input_channel * out_channels + output_channel]
                            * input[frame * stride + input_channel];
                    }
                    if use_bias {
                        sum += bias[output_channel];
                    }
                    expected[frame * out_channels + output_channel] = sum;
                }
            }

            super::conv1x1_small(
                &mut actual,
                &weights,
                &input,
                use_bias.then_some(&bias),
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
