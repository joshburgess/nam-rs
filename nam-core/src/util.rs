use crate::dsp::ActivationMode;
use crate::error::NamError;

pub fn config_usize(value: &serde_json::Value, field: &str) -> Result<usize, NamError> {
    if value.is_null() {
        return Err(NamError::MissingField(field.into()));
    }
    let value = value.as_u64().ok_or_else(|| NamError::InvalidConfigType {
        field: field.into(),
        expected: "a non-negative integer",
    })?;
    usize::try_from(value).map_err(|_| NamError::ConfigIntegerOutOfRange {
        field: field.into(),
        value,
    })
}

pub fn positive_config_usize(value: &serde_json::Value, field: &str) -> Result<usize, NamError> {
    let value = config_usize(value, field)?;
    if value == 0 {
        return Err(NamError::InvalidConfigField {
            field: field.into(),
            reason: "must be greater than zero",
        });
    }
    Ok(value)
}

pub fn checked_dimension_add(
    context: &'static str,
    left: usize,
    right: usize,
) -> Result<usize, NamError> {
    left.checked_add(right).ok_or(NamError::DimensionOverflow {
        context,
        left,
        right,
    })
}

pub fn checked_dimension_mul(
    context: &'static str,
    left: usize,
    right: usize,
) -> Result<usize, NamError> {
    left.checked_mul(right).ok_or(NamError::DimensionOverflow {
        context,
        left,
        right,
    })
}

/// Fast tanh polynomial approximation matching C++ NAM implementation.
/// Max error ~3e-4 vs std::tanh.
#[inline]
pub fn fast_tanh(x: f32) -> f32 {
    let ax = x.abs();
    let x2 = x * x;
    (x * (2.455_507_5_f32 + 2.455_507_5_f32 * ax + (0.893_229_84_f32 + 0.821_226_66_f32 * ax) * x2))
        / (2.445_066_5_f32 + (2.445_066_5_f32 + x2) * (x + 0.814_642_7_f32 * x * ax).abs())
}

/// Fast sigmoid using fast_tanh: sigmoid(x) = 0.5 * (fast_tanh(x/2) + 1)
#[inline]
pub fn fast_sigmoid(x: f32) -> f32 {
    0.5 * (fast_tanh(x * 0.5) + 1.0)
}

#[inline]
pub fn tanh(x: f32, mode: ActivationMode) -> f32 {
    if mode.use_fast_tanh() {
        fast_tanh(x)
    } else {
        x.tanh()
    }
}

#[inline]
pub fn sigmoid(x: f32, mode: ActivationMode) -> f32 {
    if mode.use_fast_tanh() {
        fast_sigmoid(x)
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Helper to consume chunks from a flat weight array in order.
pub struct WeightIter<'a> {
    weights: &'a [f32],
    pos: usize,
}

impl<'a> WeightIter<'a> {
    pub fn new(weights: &'a [f32]) -> Self {
        Self { weights, pos: 0 }
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [f32], NamError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(NamError::WeightRangeOverflow {
                position: self.pos,
                requested: n,
            })?;
        if end > self.weights.len() {
            return Err(NamError::WeightMismatch {
                expected: end,
                actual: self.weights.len(),
            });
        }
        let slice = &self.weights[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    pub fn take_matrix(
        &mut self,
        rows: usize,
        cols: usize,
    ) -> Result<ndarray::Array2<f32>, NamError> {
        let len = checked_dimension_mul("weight matrix", rows, cols)?;
        let data = self.take(len)?;
        Ok(ndarray::Array2::from_shape_vec(
            (rows, cols),
            data.to_vec(),
        )?)
    }

    pub fn take_vector(&mut self, len: usize) -> Result<ndarray::Array1<f32>, NamError> {
        let data = self.take(len)?;
        Ok(ndarray::Array1::from(data.to_vec()))
    }

    pub fn assert_exhausted(&self) -> Result<(), NamError> {
        if self.pos != self.weights.len() {
            return Err(NamError::WeightMismatch {
                expected: self.pos,
                actual: self.weights.len(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_iter_take_exact() {
        let data = vec![1.0, 2.0, 3.0];
        let mut iter = WeightIter::new(&data);
        let chunk = iter.take(2).unwrap();
        assert_eq!(chunk, &[1.0, 2.0]);
        let chunk = iter.take(1).unwrap();
        assert_eq!(chunk, &[3.0]);
        iter.assert_exhausted().unwrap();
    }

    #[test]
    fn test_weight_iter_take_overflow_errors() {
        let data = vec![1.0, 2.0];
        let mut iter = WeightIter::new(&data);
        assert!(iter.take(3).is_err());
    }

    #[test]
    fn test_weight_iter_rejects_range_overflow() {
        let data = [1.0];
        let mut iter = WeightIter::new(&data);
        iter.take(1).unwrap();
        assert!(matches!(
            iter.take(usize::MAX),
            Err(NamError::WeightRangeOverflow { .. })
        ));
    }

    #[test]
    fn test_weight_iter_rejects_matrix_size_overflow() {
        let mut iter = WeightIter::new(&[]);
        assert!(matches!(
            iter.take_matrix(usize::MAX, 2),
            Err(NamError::DimensionOverflow {
                context: "weight matrix",
                ..
            })
        ));
    }

    #[test]
    fn test_weight_iter_assert_exhausted_fails_when_remaining() {
        let data = vec![1.0, 2.0, 3.0];
        let mut iter = WeightIter::new(&data);
        iter.take(2).unwrap();
        assert!(iter.assert_exhausted().is_err());
    }

    #[test]
    fn test_weight_iter_take_matrix_shape() {
        let data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
        let mut iter = WeightIter::new(&data);
        let mat = iter.take_matrix(2, 3).unwrap();
        assert_eq!(mat.shape(), &[2, 3]);
        assert_eq!(mat[[0, 0]], 1.0);
        assert_eq!(mat[[0, 2]], 3.0);
        assert_eq!(mat[[1, 0]], 4.0);
        assert_eq!(mat[[1, 2]], 6.0);
        iter.assert_exhausted().unwrap();
    }

    #[test]
    fn test_weight_iter_take_vector_contents() {
        let data = vec![10.0, 20.0, 30.0];
        let mut iter = WeightIter::new(&data);
        let vec = iter.take_vector(3).unwrap();
        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0], 10.0);
        assert_eq!(vec[2], 30.0);
    }

    #[test]
    fn test_weight_iter_empty_weights() {
        let data: Vec<f32> = vec![];
        let iter = WeightIter::new(&data);
        iter.assert_exhausted().unwrap();
    }

    #[test]
    fn test_weight_iter_take_zero() {
        let data = vec![1.0];
        let mut iter = WeightIter::new(&data);
        let chunk = iter.take(0).unwrap();
        assert!(chunk.is_empty());
    }

    #[test]
    fn test_fast_tanh_accuracy() {
        // fast_tanh should be within 4e-4 of std tanh
        for &x in &[0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, -1.0, -3.0] {
            let diff = (fast_tanh(x) - x.tanh()).abs();
            assert!(diff < 4e-4, "fast_tanh({}) diff={:.2e}", x, diff);
        }
    }

    #[test]
    fn test_fast_sigmoid_accuracy() {
        for &x in &[0.0f32, 0.5, 1.0, 2.0, -1.0, -2.0] {
            let expected = 1.0f32 / (1.0f32 + (-x).exp());
            let diff = (fast_sigmoid(x) - expected).abs();
            assert!(diff < 4e-4, "fast_sigmoid({}) diff={:.2e}", x, diff);
        }
    }

    #[test]
    fn test_activation_modes_are_explicit() {
        let std_result = tanh(1.0, ActivationMode::Accurate);
        assert_eq!(std_result, 1.0f32.tanh());

        let fast_result = tanh(1.0, ActivationMode::Fast);
        assert_eq!(fast_result, fast_tanh(1.0));
        assert_ne!(fast_result, std_result);
    }

    #[test]
    fn test_sigmoid_at_zero() {
        assert!((sigmoid(0.0, ActivationMode::Accurate) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_sigmoid_symmetry() {
        for &x in &[0.5, 1.0, 2.0, 5.0] {
            let s_pos = sigmoid(x, ActivationMode::Accurate);
            let s_neg = sigmoid(-x, ActivationMode::Accurate);
            assert!((s_pos + s_neg - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sigmoid_bounds() {
        assert!(sigmoid(-10.0, ActivationMode::Accurate) > 0.0);
        assert!(sigmoid(-10.0, ActivationMode::Accurate) < 0.001);
        assert!(sigmoid(10.0, ActivationMode::Accurate) > 0.999);
        assert!(sigmoid(10.0, ActivationMode::Accurate) < 1.0);
    }
}
