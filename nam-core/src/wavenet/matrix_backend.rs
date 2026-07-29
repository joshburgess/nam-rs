use core::fmt;

pub(super) const SGEMM_MIN_SIZE: usize = 64;
const MATRIX_BLOCK: usize = 8;

#[derive(Clone, Debug)]
pub(super) struct MatrixLayout {
    rows: usize,
    inner: usize,
    left_len: usize,
    right_scratch: Vec<f32>,
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
            right_scratch: Vec::new(),
        })
    }

    pub(super) const fn left_len(&self) -> usize {
        self.left_len
    }

    pub(super) fn set_max_buffer_size(&mut self, max_buffer_size: usize) {
        let Some(padded_columns) = max_buffer_size.checked_add(MATRIX_BLOCK - 1) else {
            return;
        };
        let padded_columns = padded_columns / MATRIX_BLOCK * MATRIX_BLOCK;
        let Some(scratch_len) = self.inner.checked_mul(padded_columns) else {
            return;
        };
        self.right_scratch.resize(scratch_len, 0.0);
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn multiply(
        &mut self,
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
        let Some(padded_columns) = columns.checked_add(MATRIX_BLOCK - 1) else {
            return false;
        };
        let padded_columns = padded_columns / MATRIX_BLOCK * MATRIX_BLOCK;
        let Some(scratch_len) = self.inner.checked_mul(padded_columns) else {
            return false;
        };
        if right_column_stride < self.inner
            || left.len() < self.left_len
            || right.len() < right_len
            || output.len() < output_len
            || self.right_scratch.len() < scratch_len
        {
            return false;
        }
        let left = &left[..self.left_len];
        let right = &right[..right_len];
        let output = &mut output[..output_len];
        let right_scratch = &mut self.right_scratch[..scratch_len];

        pack_right(
            self.inner,
            columns,
            padded_columns,
            right,
            right_column_stride,
            right_scratch,
        );

        multiply_prepacked(
            self.rows,
            self.inner,
            columns,
            alpha,
            left,
            right_scratch,
            beta,
            output,
        );
        true
    }
}

fn pack_right(
    inner: usize,
    columns: usize,
    padded_columns: usize,
    right: &[f32],
    right_column_stride: usize,
    right_scratch: &mut [f32],
) {
    let full_columns = columns / MATRIX_BLOCK * MATRIX_BLOCK;
    for column_block in (0..full_columns).step_by(MATRIX_BLOCK) {
        let packed_block =
            &mut right_scratch[column_block * inner..(column_block + MATRIX_BLOCK) * inner];
        for depth in 0..inner {
            for column_offset in 0..MATRIX_BLOCK {
                packed_block[depth * MATRIX_BLOCK + column_offset] =
                    right[(column_block + column_offset) * right_column_stride + depth];
            }
        }
    }
    if full_columns < padded_columns {
        let packed_block =
            &mut right_scratch[full_columns * inner..(full_columns + MATRIX_BLOCK) * inner];
        for depth in 0..inner {
            for column_offset in 0..MATRIX_BLOCK {
                let column = full_columns + column_offset;
                packed_block[depth * MATRIX_BLOCK + column_offset] = if column < columns {
                    right[column * right_column_stride + depth]
                } else {
                    0.0
                };
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn multiply_prepacked(
    rows: usize,
    inner: usize,
    columns: usize,
    alpha: f32,
    left: &[f32],
    right_scratch: &[f32],
    beta: f32,
    output: &mut [f32],
) {
    // SAFETY: MatrixLayout validated the slices and AArch64 always supports
    // NEON. The helper keeps all block calls inside one target-feature scope.
    unsafe {
        multiply_prepacked_neon(
            rows,
            inner,
            columns,
            alpha,
            left,
            right_scratch,
            beta,
            output,
        );
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
fn multiply_prepacked(
    rows: usize,
    inner: usize,
    columns: usize,
    alpha: f32,
    left: &[f32],
    right_scratch: &[f32],
    beta: f32,
    output: &mut [f32],
) {
    for column_block in (0..columns).step_by(MATRIX_BLOCK) {
        let valid_columns = (columns - column_block).min(MATRIX_BLOCK);
        let packed_block =
            &right_scratch[column_block * inner..(column_block + MATRIX_BLOCK) * inner];
        // SAFETY: MatrixLayout validated the matrix slices. Each call writes a
        // disjoint output block and reads a complete packed right-hand block.
        unsafe {
            nano_gemm::planless::execute_f32(
                rows,
                valid_columns,
                inner,
                output[column_block * rows..].as_mut_ptr(),
                1,
                rows as isize,
                left.as_ptr(),
                1,
                rows as isize,
                packed_block.as_ptr(),
                MATRIX_BLOCK as isize,
                1,
                beta,
                alpha,
                false,
                false,
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn multiply_prepacked_neon(
    rows: usize,
    inner: usize,
    columns: usize,
    alpha: f32,
    left: &[f32],
    right_scratch: &[f32],
    beta: f32,
    output: &mut [f32],
) {
    for column_block in (0..columns).step_by(MATRIX_BLOCK) {
        let valid_columns = (columns - column_block).min(MATRIX_BLOCK);
        let packed_block =
            &right_scratch[column_block * inner..(column_block + MATRIX_BLOCK) * inner];
        for row_block in (0..rows).step_by(MATRIX_BLOCK) {
            let valid_rows = (rows - row_block).min(MATRIX_BLOCK);
            // SAFETY: The block bounds describe the readable and writable
            // tails of each validated matrix.
            unsafe {
                multiply_block_neon(
                    rows,
                    inner,
                    column_block,
                    row_block,
                    valid_columns,
                    valid_rows,
                    alpha,
                    left,
                    packed_block,
                    beta,
                    output,
                );
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn multiply_block_neon(
    rows: usize,
    inner: usize,
    column_block: usize,
    row_block: usize,
    valid_columns: usize,
    valid_rows: usize,
    alpha: f32,
    left: &[f32],
    packed_right: &[f32],
    beta: f32,
    output: &mut [f32],
) {
    use core::arch::aarch64::{
        float32x4_t, vfmaq_laneq_f32, vfmaq_n_f32, vld1q_f32, vmovq_n_f32, vmulq_n_f32, vst1q_f32,
    };

    let zero = vmovq_n_f32(0.0);
    let mut low_products: [float32x4_t; MATRIX_BLOCK] = [zero; MATRIX_BLOCK];
    let mut high_products: [float32x4_t; MATRIX_BLOCK] = [zero; MATRIX_BLOCK];

    for depth in 0..inner {
        let left_start = depth * rows + row_block;
        let (left_low, left_high) = if valid_rows == MATRIX_BLOCK {
            // SAFETY: A full block has eight readable rows.
            unsafe {
                (
                    vld1q_f32(left[left_start..].as_ptr()),
                    vld1q_f32(left[left_start + 4..].as_ptr()),
                )
            }
        } else {
            let mut padded = [0.0f32; MATRIX_BLOCK];
            padded[..valid_rows].copy_from_slice(&left[left_start..left_start + valid_rows]);
            // SAFETY: Both loads stay within the eight-element local array.
            unsafe { (vld1q_f32(padded.as_ptr()), vld1q_f32(padded[4..].as_ptr())) }
        };
        // SAFETY: The packed block contains eight values for every depth.
        let (right_low, right_high) = unsafe {
            let right = packed_right[depth * MATRIX_BLOCK..].as_ptr();
            (vld1q_f32(right), vld1q_f32(right.add(4)))
        };

        low_products[0] = vfmaq_laneq_f32(low_products[0], left_low, right_low, 0);
        low_products[1] = vfmaq_laneq_f32(low_products[1], left_low, right_low, 1);
        low_products[2] = vfmaq_laneq_f32(low_products[2], left_low, right_low, 2);
        low_products[3] = vfmaq_laneq_f32(low_products[3], left_low, right_low, 3);
        low_products[4] = vfmaq_laneq_f32(low_products[4], left_low, right_high, 0);
        low_products[5] = vfmaq_laneq_f32(low_products[5], left_low, right_high, 1);
        low_products[6] = vfmaq_laneq_f32(low_products[6], left_low, right_high, 2);
        low_products[7] = vfmaq_laneq_f32(low_products[7], left_low, right_high, 3);
        high_products[0] = vfmaq_laneq_f32(high_products[0], left_high, right_low, 0);
        high_products[1] = vfmaq_laneq_f32(high_products[1], left_high, right_low, 1);
        high_products[2] = vfmaq_laneq_f32(high_products[2], left_high, right_low, 2);
        high_products[3] = vfmaq_laneq_f32(high_products[3], left_high, right_low, 3);
        high_products[4] = vfmaq_laneq_f32(high_products[4], left_high, right_high, 0);
        high_products[5] = vfmaq_laneq_f32(high_products[5], left_high, right_high, 1);
        high_products[6] = vfmaq_laneq_f32(high_products[6], left_high, right_high, 2);
        high_products[7] = vfmaq_laneq_f32(high_products[7], left_high, right_high, 3);
    }

    for column in 0..valid_columns {
        let output_start = (column_block + column) * rows + row_block;
        let mut low = vmulq_n_f32(low_products[column], alpha);
        let mut high = vmulq_n_f32(high_products[column], alpha);
        if beta != 0.0 {
            if valid_rows == MATRIX_BLOCK {
                // SAFETY: A full block has eight readable output rows.
                unsafe {
                    low = vfmaq_n_f32(low, vld1q_f32(output[output_start..].as_ptr()), beta);
                    high = vfmaq_n_f32(high, vld1q_f32(output[output_start + 4..].as_ptr()), beta);
                }
            } else {
                let mut padded = [0.0f32; MATRIX_BLOCK];
                padded[..valid_rows]
                    .copy_from_slice(&output[output_start..output_start + valid_rows]);
                // SAFETY: Both loads stay within the eight-element local array.
                unsafe {
                    low = vfmaq_n_f32(low, vld1q_f32(padded.as_ptr()), beta);
                    high = vfmaq_n_f32(high, vld1q_f32(padded[4..].as_ptr()), beta);
                }
            }
        }
        if valid_rows == MATRIX_BLOCK {
            // SAFETY: A full block has eight writable output rows.
            unsafe {
                vst1q_f32(output[output_start..].as_mut_ptr(), low);
                vst1q_f32(output[output_start + 4..].as_mut_ptr(), high);
            }
        } else {
            let mut padded = [0.0f32; MATRIX_BLOCK];
            // SAFETY: Both stores stay within the eight-element local array.
            unsafe {
                vst1q_f32(padded.as_mut_ptr(), low);
                vst1q_f32(padded[4..].as_mut_ptr(), high);
            }
            output[output_start..output_start + valid_rows].copy_from_slice(&padded[..valid_rows]);
        }
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
        let mut layout = MatrixLayout::new(2, 2).unwrap();
        layout.set_max_buffer_size(2);
        let mut output = [7.0; 4];

        assert!(!layout.multiply(2, 1.0, &[1.0; 3], &[1.0; 4], 2, 0.0, &mut output));
        assert_eq!(output, [7.0; 4]);
        assert!(!layout.multiply(2, 1.0, &[1.0; 4], &[1.0; 4], 1, 0.0, &mut output));
        assert_eq!(output, [7.0; 4]);
    }
}
