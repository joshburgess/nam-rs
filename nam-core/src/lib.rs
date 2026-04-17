// `Sample` is a type alias that resolves to `f64` by default and `f32` under
// the `float_io` feature (used by `nam-wasm`). Several internal buffers are
// always `f32`, so the crate contains `value as f32` casts that are real
// narrowing conversions in the default config but no-ops when `float_io` is
// enabled. Clippy can only see the currently-compiled config and flags the
// no-op cases as `unnecessary_cast`, which is a false positive for this
// feature-aliased pattern — allow it crate-wide.
#![allow(clippy::unnecessary_cast)]

pub mod activations;
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

pub use dsp::{Dsp, Sample};
pub use error::NamError;
pub use get_dsp::{get_dsp, get_dsp_from_json};
pub use util::{disable_fast_tanh, enable_fast_tanh, is_fast_tanh_enabled};
