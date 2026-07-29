#![cfg_attr(not(target_arch = "x86_64"), allow(dead_code, unused_variables))]

#[derive(Clone, Copy)]
pub(super) enum SmallMatrixBackend {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2Fma,
}

impl SmallMatrixBackend {
    pub(super) fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return Self::Avx2Fma;
        }

        Self::Scalar
    }

    pub(super) fn conv1x1(
        self,
        output: &mut [f32],
        weights: &[f32],
        input: &[f32],
        bias: Option<&[f32]>,
        dimensions: Conv1x1Dimensions,
    ) -> bool {
        #[cfg(target_arch = "x86_64")]
        if matches!(self, Self::Avx2Fma) && dimensions.validate(output, weights, input, bias) {
            // SAFETY: Backend detection established AVX2 and FMA support, and
            // validation covers every matrix extent used by the kernel.
            unsafe {
                conv1x1_avx2_fma(output, weights, input, bias, dimensions);
            }
            return true;
        }

        false
    }

    pub(super) fn conv1d_one_output_tap(
        self,
        output: &mut [f32],
        weights: &[f32],
        input: &[f32],
        in_channels: usize,
        num_frames: usize,
    ) -> bool {
        #[cfg(target_arch = "x86_64")]
        if matches!(self, Self::Avx2Fma)
            && output.len() >= num_frames
            && weights.len() >= in_channels
            && required_input_len(in_channels, in_channels, num_frames)
                .is_some_and(|required| input.len() >= required)
        {
            // SAFETY: Backend detection established AVX2 and FMA support, and
            // the slice checks cover every frame and channel used by the kernel.
            unsafe {
                conv1d_one_output_tap_avx2_fma(output, weights, input, in_channels, num_frames);
            }
            return true;
        }

        false
    }
}

#[derive(Clone, Copy)]
pub(super) struct Conv1x1Dimensions {
    pub(super) out_channels: usize,
    pub(super) in_channels: usize,
    pub(super) input_stride: usize,
    pub(super) num_frames: usize,
}

impl Conv1x1Dimensions {
    fn validate(
        self,
        output: &[f32],
        weights: &[f32],
        input: &[f32],
        bias: Option<&[f32]>,
    ) -> bool {
        let Some(output_len) = self.out_channels.checked_mul(self.num_frames) else {
            return false;
        };
        let Some(weight_len) = self.out_channels.checked_mul(self.in_channels) else {
            return false;
        };
        output.len() >= output_len
            && weights.len() >= weight_len
            && required_input_len(self.input_stride, self.in_channels, self.num_frames)
                .is_some_and(|required| input.len() >= required)
            && bias.is_none_or(|values| values.len() >= self.out_channels)
    }
}

fn required_input_len(stride: usize, channels: usize, frames: usize) -> Option<usize> {
    if frames == 0 {
        Some(0)
    } else {
        (frames - 1).checked_mul(stride)?.checked_add(channels)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::needless_range_loop)]
unsafe fn conv1x1_avx2_fma(
    output: &mut [f32],
    weights: &[f32],
    input: &[f32],
    bias: Option<&[f32]>,
    dimensions: Conv1x1Dimensions,
) {
    use std::arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setr_ps,
        _mm256_setzero_ps, _mm256_storeu_ps,
    };

    let Conv1x1Dimensions {
        out_channels,
        in_channels,
        input_stride,
        num_frames,
    } = dimensions;

    if out_channels == 1 {
        let mut frame = 0;
        while frame + 8 <= num_frames {
            let mut product = _mm256_setzero_ps();
            for input_channel in 0..in_channels {
                // SAFETY: The validated input extent covers all eight frames,
                // and this function enables every feature used by the helper.
                let input_values =
                    unsafe { gather_frames(input, frame, input_stride, input_channel) };
                let weight = _mm256_set1_ps(weights[input_channel]);
                product = _mm256_fmadd_ps(weight, input_values, product);
            }
            if let Some(values) = bias {
                product = _mm256_add_ps(product, _mm256_set1_ps(values[0]));
            }
            // SAFETY: The validated output extent includes these eight frames.
            unsafe {
                _mm256_storeu_ps(output.as_mut_ptr().add(frame), product);
            }
            frame += 8;
        }
        for frame in frame..num_frames {
            let input_column = frame * input_stride;
            let mut product = weights[0].mul_add(input[input_column], 0.0);
            for input_channel in 1..in_channels {
                product =
                    weights[input_channel].mul_add(input[input_column + input_channel], product);
            }
            output[frame] = product + bias.map_or(0.0, |values| values[0]);
        }
        return;
    }

    for frame in 0..num_frames {
        let input_column = frame * input_stride;
        let output_column = frame * out_channels;
        let mut output_channel = 0;
        while output_channel + 8 <= out_channels {
            let mut product = _mm256_setzero_ps();
            for input_channel in 0..in_channels {
                let weight_offset = input_channel * out_channels + output_channel;
                // SAFETY: The validated weight extent includes this eight-value block.
                let weight = unsafe { _mm256_loadu_ps(weights.as_ptr().add(weight_offset)) };
                let input_value = _mm256_set1_ps(input[input_column + input_channel]);
                product = _mm256_fmadd_ps(weight, input_value, product);
            }
            if let Some(values) = bias {
                // SAFETY: Bias validation includes this eight-value block.
                let bias_values = unsafe { _mm256_loadu_ps(values.as_ptr().add(output_channel)) };
                product = _mm256_add_ps(product, bias_values);
            }
            // SAFETY: The validated output extent includes this eight-value block.
            unsafe {
                _mm256_storeu_ps(
                    output.as_mut_ptr().add(output_column + output_channel),
                    product,
                );
            }
            output_channel += 8;
        }
        for output_channel in output_channel..out_channels {
            let mut product = weights[output_channel].mul_add(input[input_column], 0.0);
            for input_channel in 1..in_channels {
                product = weights[input_channel * out_channels + output_channel]
                    .mul_add(input[input_column + input_channel], product);
            }
            output[output_column + output_channel] =
                product + bias.map_or(0.0, |values| values[output_channel]);
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn gather_frames(
        input: &[f32],
        frame: usize,
        stride: usize,
        input_channel: usize,
    ) -> __m256 {
        let offset = frame * stride + input_channel;
        _mm256_setr_ps(
            input[offset],
            input[offset + stride],
            input[offset + 2 * stride],
            input[offset + 3 * stride],
            input[offset + 4 * stride],
            input[offset + 5 * stride],
            input[offset + 6 * stride],
            input[offset + 7 * stride],
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::needless_range_loop)]
unsafe fn conv1d_one_output_tap_avx2_fma(
    output: &mut [f32],
    weights: &[f32],
    input: &[f32],
    in_channels: usize,
    num_frames: usize,
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setr_ps,
        _mm256_setzero_ps, _mm256_storeu_ps,
    };

    let mut frame = 0;
    while frame + 8 <= num_frames {
        let mut product = _mm256_setzero_ps();
        for input_channel in 0..in_channels {
            let offset = frame * in_channels + input_channel;
            let input_values = _mm256_setr_ps(
                input[offset],
                input[offset + in_channels],
                input[offset + 2 * in_channels],
                input[offset + 3 * in_channels],
                input[offset + 4 * in_channels],
                input[offset + 5 * in_channels],
                input[offset + 6 * in_channels],
                input[offset + 7 * in_channels],
            );
            product = _mm256_fmadd_ps(
                _mm256_set1_ps(weights[input_channel]),
                input_values,
                product,
            );
        }
        // SAFETY: The validated output extent includes these eight frames.
        let previous = unsafe { _mm256_loadu_ps(output.as_ptr().add(frame)) };
        // SAFETY: The validated output extent includes these eight frames.
        unsafe {
            _mm256_storeu_ps(
                output.as_mut_ptr().add(frame),
                _mm256_add_ps(previous, product),
            );
        }
        frame += 8;
    }
    for frame in frame..num_frames {
        let input_column = frame * in_channels;
        let mut product = 0.0;
        for input_channel in 0..in_channels {
            product = weights[input_channel].mul_add(input[input_column + input_channel], product);
        }
        output[frame] += product;
    }
}
