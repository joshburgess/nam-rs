use crate::error::NamError;
use std::sync::atomic::{AtomicBool, Ordering};

/// Global toggle for fast tanh/sigmoid approximations.
/// When enabled, uses polynomial approximations matching C++ NAM's fast_tanh.
/// Faster (~1.5-2x) but introduces ~3e-4 error per call.
static USE_FAST_TANH: AtomicBool = AtomicBool::new(false);

/// Enable fast tanh/sigmoid polynomial approximations (performance mode).
pub fn enable_fast_tanh() {
    USE_FAST_TANH.store(true, Ordering::Relaxed);
}

/// Disable fast tanh/sigmoid, using standard math (accuracy mode, default).
pub fn disable_fast_tanh() {
    USE_FAST_TANH.store(false, Ordering::Relaxed);
}

/// Returns true if fast tanh/sigmoid is currently enabled.
pub fn is_fast_tanh_enabled() -> bool {
    USE_FAST_TANH.load(Ordering::Relaxed)
}

/// Fast tanh polynomial approximation matching C++ NAM implementation.
/// Max error ~3e-4 vs std::tanh.
#[inline]
pub fn fast_tanh(x: f32) -> f32 {
    let ax = x.abs();
    let x2 = x * x;
    (x * (2.45550750702956f32
        + 2.45550750702956f32 * ax
        + (0.893229853513558f32 + 0.821226666969744f32 * ax) * x2))
        / (2.44506634652299f32
            + (2.44506634652299f32 + x2) * (x + 0.814642734961073f32 * x * ax).abs())
}

/// Fast sigmoid using fast_tanh: sigmoid(x) = 0.5 * (fast_tanh(x/2) + 1)
#[inline]
pub fn fast_sigmoid(x: f32) -> f32 {
    0.5 * (fast_tanh(x * 0.5) + 1.0)
}

/// Tanh that respects the fast_tanh toggle.
#[inline]
pub fn tanh_auto(x: f32) -> f32 {
    if USE_FAST_TANH.load(Ordering::Relaxed) {
        fast_tanh(x)
    } else {
        x.tanh()
    }
}

/// Sigmoid that respects the fast_tanh toggle.
#[inline]
pub fn sigmoid_auto(x: f32) -> f32 {
    if USE_FAST_TANH.load(Ordering::Relaxed) {
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
        if self.pos + n > self.weights.len() {
            return Err(NamError::WeightMismatch {
                expected: self.pos + n,
                actual: self.weights.len(),
            });
        }
        let slice = &self.weights[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    pub fn take_matrix(
        &mut self,
        rows: usize,
        cols: usize,
    ) -> Result<ndarray::Array2<f32>, NamError> {
        let data = self.take(rows * cols)?;
        Ok(ndarray::Array2::from_shape_vec((rows, cols), data.to_vec()).unwrap())
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

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    sigmoid_auto(x)
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
    fn test_fast_tanh_toggle() {
        disable_fast_tanh();
        let std_result = tanh_auto(1.0);
        assert_eq!(std_result, 1.0f32.tanh());

        enable_fast_tanh();
        let fast_result = tanh_auto(1.0);
        assert_eq!(fast_result, fast_tanh(1.0));
        assert!(fast_result != std_result); // they should differ

        disable_fast_tanh(); // restore default
    }

    #[test]
    fn test_sigmoid_at_zero() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_sigmoid_symmetry() {
        for &x in &[0.5, 1.0, 2.0, 5.0] {
            let s_pos = sigmoid(x);
            let s_neg = sigmoid(-x);
            assert!((s_pos + s_neg - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sigmoid_bounds() {
        assert!(sigmoid(-10.0) > 0.0);
        assert!(sigmoid(-10.0) < 0.001);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(10.0) < 1.0);
    }
}
