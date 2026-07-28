use core::fmt;

pub(super) const SGEMM_MIN_SIZE: usize = 64;

#[derive(Clone, Copy, Debug)]
pub(super) struct MatrixLayout {
    rows: usize,
    inner: usize,
    left_len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MatrixLayoutError {
    rows: usize,
    inner: usize,
}

impl fmt::Display for MatrixLayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "matrix dimensions overflow usize: {} x {}",
            self.rows, self.inner
        )
    }
}

impl MatrixLayout {
    pub(super) fn new(rows: usize, inner: usize) -> Result<Self, MatrixLayoutError> {
        let left_len = rows
            .checked_mul(inner)
            .ok_or(MatrixLayoutError { rows, inner })?;
        Ok(Self {
            rows,
            inner,
            left_len,
        })
    }

    pub(super) const fn left_len(self) -> usize {
        self.left_len
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn multiply(
        self,
        columns: usize,
        alpha: f32,
        left: &[f32],
        right: &[f32],
        right_column_stride: usize,
        beta: f32,
        output: &mut [f32],
    ) -> bool {
        let Some(right_len) = right_column_stride.checked_mul(columns) else {
            return false;
        };
        let Some(output_len) = self.rows.checked_mul(columns) else {
            return false;
        };
        if right_column_stride < self.inner
            || left.len() < self.left_len
            || right.len() < right_len
            || output.len() < output_len
        {
            return false;
        }
        let left = &left[..self.left_len];
        let right = &right[..right_len];
        let output = &mut output[..output_len];

        #[cfg(feature = "faer")]
        {
            let left = faer::mat::from_column_major_slice::<f32, usize, usize>(
                left, self.rows, self.inner,
            );
            let right = faer::mat::from_column_major_slice::<f32, usize, usize>(
                right,
                right_column_stride,
                columns,
            );
            let right = right.subrows(0, self.inner);
            let output = faer::mat::from_column_major_slice_mut::<f32, usize, usize>(
                output, self.rows, columns,
            );

            faer::linalg::matmul::matmul(
                output,
                left,
                right,
                Some(beta),
                alpha,
                faer::Parallelism::None,
            );
        }

        #[cfg(not(feature = "faer"))]
        {
            // SAFETY: The validated slices cover all three column-major matrices.
            unsafe {
                matrixmultiply::sgemm(
                    self.rows,
                    self.inner,
                    columns,
                    alpha,
                    left.as_ptr(),
                    1,
                    self.rows as isize,
                    right.as_ptr(),
                    1,
                    right_column_stride as isize,
                    beta,
                    output.as_mut_ptr(),
                    1,
                    self.rows as isize,
                );
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::MatrixLayout;

    #[test]
    fn rejects_dimensions_that_overflow_storage() {
        assert!(MatrixLayout::new(usize::MAX, 2).is_err());
    }

    #[test]
    fn rejects_invalid_dynamic_buffers_without_touching_output() {
        let layout = MatrixLayout::new(2, 2).unwrap();
        let mut output = [7.0; 4];

        assert!(!layout.multiply(2, 1.0, &[1.0; 3], &[1.0; 4], 2, 0.0, &mut output));
        assert_eq!(output, [7.0; 4]);
        assert!(!layout.multiply(2, 1.0, &[1.0; 4], &[1.0; 4], 1, 0.0, &mut output));
        assert_eq!(output, [7.0; 4]);
    }
}
