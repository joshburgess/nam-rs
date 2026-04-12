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
pub use get_dsp::get_dsp;
pub use util::{disable_fast_tanh, enable_fast_tanh, is_fast_tanh_enabled};
