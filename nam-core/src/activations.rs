use crate::error::NamError;
use crate::util::{fast_sigmoid, fast_tanh, is_fast_tanh_enabled};

#[derive(Debug, Clone)]
pub enum Activation {
    Relu,
    Tanh,
    Sigmoid,
    HardTanh,
    LeakyRelu(f32),
    Silu,
    Softsign,
    HardSwish,
    LeakyHardTanh {
        min_val: f32,
        max_val: f32,
        min_slope: f32,
        max_slope: f32,
    },
    /// Per-channel parametric ReLU. Each channel has its own negative slope.
    PReLU(Vec<f32>),
}

impl Activation {
    /// Apply activation to a single value. For hot loops, prefer `apply_slice`
    /// which hoists the fast_tanh flag check out of the loop.
    #[inline]
    pub fn apply_scalar(&self, x: f32) -> f32 {
        self.apply_scalar_fast(x, is_fast_tanh_enabled())
    }

    #[inline]
    pub fn apply_scalar_fast(&self, x: f32, use_fast: bool) -> f32 {
        match self {
            Activation::Relu => x.max(0.0),
            Activation::Tanh => {
                if use_fast {
                    fast_tanh(x)
                } else {
                    x.tanh()
                }
            }
            Activation::Sigmoid => {
                if use_fast {
                    fast_sigmoid(x)
                } else {
                    1.0 / (1.0 + (-x).exp())
                }
            }
            Activation::HardTanh => x.clamp(-1.0, 1.0),
            Activation::LeakyRelu(alpha) => {
                if x >= 0.0 {
                    x
                } else {
                    alpha * x
                }
            }
            Activation::Silu => {
                let sig = if use_fast {
                    fast_sigmoid(x)
                } else {
                    1.0 / (1.0 + (-x).exp())
                };
                x * sig
            }
            Activation::Softsign => x / (1.0 + x.abs()),
            Activation::HardSwish => x * (x + 3.0).clamp(0.0, 6.0) * (1.0 / 6.0),
            Activation::LeakyHardTanh {
                min_val,
                max_val,
                min_slope,
                max_slope,
            } => {
                if x < *min_val {
                    (x - min_val) * min_slope + min_val
                } else if x > *max_val {
                    (x - max_val) * max_slope + max_val
                } else {
                    x
                }
            }
            Activation::PReLU(slopes) => {
                let alpha = slopes.first().copied().unwrap_or(0.01);
                if x >= 0.0 {
                    x
                } else {
                    alpha * x
                }
            }
        }
    }

    /// Apply activation at a specific channel index. Only differs from apply_scalar
    /// for PReLU, which has per-channel slopes.
    #[inline]
    pub fn apply_scalar_channel(&self, x: f32, channel: usize) -> f32 {
        self.apply_scalar_channel_fast(x, channel, is_fast_tanh_enabled())
    }

    #[inline]
    pub fn apply_scalar_channel_fast(&self, x: f32, channel: usize, use_fast: bool) -> f32 {
        match self {
            Activation::PReLU(slopes) => {
                let alpha = slopes.get(channel).copied().unwrap_or(0.01);
                if x >= 0.0 {
                    x
                } else {
                    alpha * x
                }
            }
            _ => self.apply_scalar_fast(x, use_fast),
        }
    }

    #[inline]
    pub fn apply_slice(&self, data: &mut [f32]) {
        let use_fast = is_fast_tanh_enabled();
        for x in data.iter_mut() {
            *x = self.apply_scalar_fast(*x, use_fast);
        }
    }

    /// Apply activation to a slice with per-channel support. For PReLU, `data[i]`
    /// uses slope for `channel = i % num_channels`.
    #[inline]
    pub fn apply_slice_channels(&self, data: &mut [f32], num_channels: usize) {
        match self {
            Activation::PReLU(slopes) => {
                for (i, x) in data.iter_mut().enumerate() {
                    let ch = i % num_channels;
                    let alpha = slopes.get(ch).copied().unwrap_or(0.01);
                    *x = if *x >= 0.0 { *x } else { alpha * *x };
                }
            }
            _ => self.apply_slice(data),
        }
    }

    /// Parse from the string name in a .nam config.
    pub fn from_name(name: &str) -> Result<Self, NamError> {
        match name {
            "Relu" | "ReLU" => Ok(Activation::Relu),
            "Tanh" => Ok(Activation::Tanh),
            "Sigmoid" => Ok(Activation::Sigmoid),
            "HardTanh" | "Hardtanh" => Ok(Activation::HardTanh),
            "SiLU" => Ok(Activation::Silu),
            "Softsign" => Ok(Activation::Softsign),
            "Hardswish" => Ok(Activation::HardSwish),
            "LeakyHardtanh" | "LeakyHardTanh" => Ok(Activation::LeakyHardTanh {
                min_val: -1.0,
                max_val: 1.0,
                min_slope: 0.01,
                max_slope: 0.01,
            }),
            other => Err(NamError::UnknownActivation(other.into())),
        }
    }

    /// Parse from a serde_json::Value, which can be a string or an object.
    pub fn from_json(val: &serde_json::Value) -> Result<Self, NamError> {
        if let Some(name) = val.as_str() {
            return Self::from_name(name);
        }
        if let Some(obj) = val.as_object() {
            let type_name = obj.get("type").and_then(|v| v.as_str()).ok_or_else(|| {
                NamError::InvalidConfig("activation object missing 'type'".into())
            })?;

            match type_name {
                "LeakyReLU" => {
                    let slope = obj
                        .get("negative_slope")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.01) as f32;
                    Ok(Activation::LeakyRelu(slope))
                }
                "PReLU" => {
                    if let Some(slopes) = obj.get("negative_slopes").and_then(|v| v.as_array()) {
                        let slopes: Vec<f32> = slopes
                            .iter()
                            .map(|v| v.as_f64().unwrap_or(0.01) as f32)
                            .collect();
                        return Ok(Activation::PReLU(slopes));
                    }
                    let slope = obj
                        .get("negative_slope")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.01) as f32;
                    Ok(Activation::PReLU(vec![slope]))
                }
                "LeakyHardtanh" | "LeakyHardTanh" => {
                    let min_val =
                        obj.get("min_val").and_then(|v| v.as_f64()).unwrap_or(-1.0) as f32;
                    let max_val = obj.get("max_val").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                    let min_slope = obj
                        .get("min_slope")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.01) as f32;
                    let max_slope = obj
                        .get("max_slope")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.01) as f32;
                    Ok(Activation::LeakyHardTanh {
                        min_val,
                        max_val,
                        min_slope,
                        max_slope,
                    })
                }
                other => Self::from_name(other),
            }
        } else {
            // Empty or null -> default identity-like (just use Tanh as fallback)
            Err(NamError::InvalidConfig(format!(
                "activation config is neither string nor object: {:?}",
                val
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let act = Activation::Relu;
        assert_eq!(act.apply_scalar(-1.0), 0.0);
        assert_eq!(act.apply_scalar(0.0), 0.0);
        assert_eq!(act.apply_scalar(1.0), 1.0);
    }

    #[test]
    fn test_tanh() {
        let act = Activation::Tanh;
        let x = 0.5f32;
        assert!((act.apply_scalar(x) - x.tanh()).abs() < 1e-7);
    }

    #[test]
    fn test_sigmoid() {
        let act = Activation::Sigmoid;
        assert!((act.apply_scalar(0.0) - 0.5).abs() < 1e-7);
        let x = 1.0f32;
        let expected = 1.0 / (1.0 + (-x).exp());
        assert!((act.apply_scalar(x) - expected).abs() < 1e-7);
    }

    #[test]
    fn test_hard_tanh() {
        let act = Activation::HardTanh;
        assert_eq!(act.apply_scalar(-2.0), -1.0);
        assert_eq!(act.apply_scalar(0.5), 0.5);
        assert_eq!(act.apply_scalar(2.0), 1.0);
    }

    #[test]
    fn test_silu() {
        let act = Activation::Silu;
        assert!((act.apply_scalar(0.0) - 0.0).abs() < 1e-7);
        let x = 1.0f32;
        let expected = x / (1.0 + (-x).exp());
        assert!((act.apply_scalar(x) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_apply_slice() {
        let act = Activation::Relu;
        let mut data = vec![-1.0, 0.0, 1.0, 2.0];
        act.apply_slice(&mut data);
        assert_eq!(data, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_from_name() {
        assert!(Activation::from_name("Relu").is_ok());
        assert!(Activation::from_name("Tanh").is_ok());
        assert!(Activation::from_name("Sigmoid").is_ok());
        assert!(Activation::from_name("HardTanh").is_ok());
        assert!(Activation::from_name("SiLU").is_ok());
        assert!(Activation::from_name("Softsign").is_ok());
        assert!(Activation::from_name("Hardswish").is_ok());
        assert!(Activation::from_name("LeakyHardtanh").is_ok());
        assert!(Activation::from_name("LeakyHardTanh").is_ok());
        assert!(Activation::from_name("Unknown").is_err());
    }

    #[test]
    fn test_relu_large_values() {
        let act = Activation::Relu;
        assert_eq!(act.apply_scalar(1e10), 1e10);
        assert_eq!(act.apply_scalar(-1e10), 0.0);
    }

    #[test]
    fn test_tanh_saturation() {
        let act = Activation::Tanh;
        assert!((act.apply_scalar(100.0) - 1.0).abs() < 1e-6);
        assert!((act.apply_scalar(-100.0) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_at_extremes() {
        let act = Activation::Sigmoid;
        assert!(act.apply_scalar(50.0) > 0.999);
        assert!(act.apply_scalar(-50.0) < 0.001);
    }

    #[test]
    fn test_silu_negative_is_small() {
        let act = Activation::Silu;
        let val = act.apply_scalar(-10.0);
        assert!(val.abs() < 0.001);
        assert!(val < 0.0);
    }

    #[test]
    fn test_leaky_relu_slope() {
        let act = Activation::LeakyRelu(0.1);
        assert_eq!(act.apply_scalar(5.0), 5.0);
        assert!((act.apply_scalar(-5.0) - (-0.5)).abs() < 1e-7);
        assert_eq!(act.apply_scalar(0.0), 0.0);
    }

    #[test]
    fn test_apply_slice_empty() {
        let act = Activation::Relu;
        let mut data: Vec<f32> = vec![];
        act.apply_slice(&mut data);
    }

    #[test]
    fn test_from_name_case_sensitive() {
        assert!(Activation::from_name("relu").is_err());
        assert!(Activation::from_name("RELU").is_err());
        assert!(Activation::from_name("tanh").is_err());
        assert!(Activation::from_name("Relu").is_ok());
    }

    // New activation tests

    #[test]
    fn test_softsign() {
        let act = Activation::Softsign;
        assert_eq!(act.apply_scalar(0.0), 0.0);
        assert!((act.apply_scalar(1.0) - 0.5).abs() < 1e-7);
        assert!((act.apply_scalar(-1.0) - (-0.5)).abs() < 1e-7);
        // Approaches +/-1 for large values
        assert!((act.apply_scalar(100.0) - 1.0).abs() < 0.02);
        assert!((act.apply_scalar(-100.0) + 1.0).abs() < 0.02);
    }

    #[test]
    fn test_hardswish() {
        let act = Activation::HardSwish;
        // x <= -3: 0
        assert_eq!(act.apply_scalar(-3.0), 0.0);
        assert_eq!(act.apply_scalar(-5.0), 0.0);
        // x >= 3: x
        assert!((act.apply_scalar(3.0) - 3.0).abs() < 1e-6);
        assert!((act.apply_scalar(5.0) - 5.0).abs() < 1e-6);
        // x = 0: 0 * (0+3)/6 = 0
        assert_eq!(act.apply_scalar(0.0), 0.0);
        // x = 1: 1 * (1+3)/6 = 4/6 = 0.6667
        assert!((act.apply_scalar(1.0) - (4.0 / 6.0)).abs() < 1e-6);
    }

    #[test]
    fn test_leaky_hardtanh() {
        let act = Activation::LeakyHardTanh {
            min_val: -1.0,
            max_val: 1.0,
            min_slope: 0.01,
            max_slope: 0.01,
        };
        // In range: identity
        assert_eq!(act.apply_scalar(0.0), 0.0);
        assert_eq!(act.apply_scalar(0.5), 0.5);
        assert_eq!(act.apply_scalar(-0.5), -0.5);
        // At boundaries
        assert_eq!(act.apply_scalar(1.0), 1.0);
        assert_eq!(act.apply_scalar(-1.0), -1.0);
        // Beyond boundaries: leaky
        assert!((act.apply_scalar(2.0) - (1.0 + 0.01)).abs() < 1e-6);
        assert!((act.apply_scalar(-2.0) - (-1.0 - 0.01)).abs() < 1e-6);
    }

    #[test]
    fn test_leaky_hardtanh_custom_params() {
        let act = Activation::LeakyHardTanh {
            min_val: 0.0,
            max_val: 0.9,
            min_slope: 0.0,
            max_slope: 0.02,
        };
        assert_eq!(act.apply_scalar(0.5), 0.5);
        assert_eq!(act.apply_scalar(-1.0), 0.0); // (x - 0.0) * 0.0 + 0.0 = 0
        assert!((act.apply_scalar(1.0) - (0.9 + 0.1 * 0.02)).abs() < 1e-6);
    }

    #[test]
    fn test_from_json_string() {
        let val = serde_json::json!("Tanh");
        let act = Activation::from_json(&val).unwrap();
        assert!((act.apply_scalar(0.5) - 0.5f32.tanh()).abs() < 1e-7);
    }

    #[test]
    fn test_from_json_object_leaky_relu() {
        let val = serde_json::json!({"type": "LeakyReLU", "negative_slope": 0.2});
        let act = Activation::from_json(&val).unwrap();
        assert!((act.apply_scalar(-1.0) - (-0.2)).abs() < 1e-7);
    }

    #[test]
    fn test_from_json_object_leaky_hardtanh() {
        let val = serde_json::json!({
            "type": "LeakyHardtanh",
            "min_val": -0.5,
            "max_val": 0.5,
            "min_slope": 0.1,
            "max_slope": 0.1
        });
        let act = Activation::from_json(&val).unwrap();
        assert_eq!(act.apply_scalar(0.0), 0.0);
        assert!((act.apply_scalar(1.0) - (0.5 + 0.5 * 0.1)).abs() < 1e-6);
    }

    #[test]
    fn test_from_json_object_softsign() {
        let val = serde_json::json!({"type": "Softsign"});
        let act = Activation::from_json(&val).unwrap();
        assert!((act.apply_scalar(1.0) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_from_json_object_prelu() {
        let val = serde_json::json!({"type": "PReLU", "negative_slopes": [0.04, 0.05]});
        let act = Activation::from_json(&val).unwrap();
        // Should use first slope for scalar apply
        assert!((act.apply_scalar(-1.0) - (-0.04)).abs() < 1e-7);
    }
}
