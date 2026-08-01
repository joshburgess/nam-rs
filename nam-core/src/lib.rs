pub mod activations;
pub mod build_info;
pub mod convnet;
pub mod dsp;
pub mod error;
#[cfg(feature = "fast-kernels")]
pub(crate) mod fast_kernels;
pub mod get_dsp;
pub mod linear;
pub mod lstm;
pub mod util;
pub mod version;
pub mod wavenet;

pub use dsp::{ActivationMode, Dsp, Sample};
pub use error::NamError;
pub use get_dsp::get_dsp;

#[cfg(all(feature = "benchmark-internals", feature = "fast-kernels"))]
pub mod benchmark {
    pub use crate::wavenet::GroupedConv1x1Benchmark;

    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_small_gemv(
        output: &mut [f32],
        input: &[f32],
        tap_offsets: &[usize],
        weights: &[f32],
        bias: &[f32],
        out_channels: usize,
        in_channels: usize,
        num_frames: usize,
    ) {
        crate::fast_kernels::conv1d_small_gemv(
            output,
            input,
            tap_offsets,
            weights,
            bias,
            crate::fast_kernels::Conv1dDimensions {
                out_channels,
                in_channels,
                num_frames,
            },
        );
    }

    pub fn conv1d_grouped_12x3_k2(
        output: &mut [f32],
        input: &[f32],
        tap_offsets: &[usize],
        weights: &[f32],
        bias: &[f32],
        num_frames: usize,
    ) -> bool {
        crate::fast_kernels::conv1d_grouped_12x3_k2(
            output,
            input,
            tap_offsets,
            weights,
            bias,
            num_frames,
        )
    }
}
