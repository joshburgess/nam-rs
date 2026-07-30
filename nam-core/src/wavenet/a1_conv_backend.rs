const TAPS: usize = 3;

#[cfg(target_arch = "aarch64")]
const COLUMN_BLOCK: usize = 8;
#[cfg(target_arch = "x86_64")]
const COLUMN_BLOCK: usize = 4;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
const COLUMN_BLOCK: usize = 1;

#[derive(Clone, Debug)]
pub(super) struct PackedA1Conv {
    rows: usize,
    inner: usize,
    weights: Vec<f32>,
    right_scratch: Vec<f32>,
}

impl PackedA1Conv {
    pub(super) fn new(rows: usize, inner: usize, taps: &[Vec<f32>]) -> Option<Self> {
        let weight_len = rows.checked_mul(inner)?;
        if !supports_shape(rows, inner)
            || taps.len() != TAPS
            || taps.iter().any(|tap| tap.len() != weight_len)
        {
            return None;
        }

        let mut weights = Vec::with_capacity(TAPS * rows * inner);
        #[cfg(target_arch = "aarch64")]
        if rows == 16 {
            pack_weight_rows(&mut weights, taps, rows, inner, 0, 12);
            pack_weight_rows(&mut weights, taps, rows, inner, 12, 4);
        } else {
            pack_weight_rows(&mut weights, taps, rows, inner, 0, rows);
        }
        #[cfg(not(target_arch = "aarch64"))]
        pack_weight_rows(&mut weights, taps, rows, inner, 0, rows);

        Some(Self {
            rows,
            inner,
            weights,
            right_scratch: Vec::new(),
        })
    }

    pub(super) fn set_max_buffer_size(&mut self, max_buffer_size: usize) {
        let Some(padded_columns) = max_buffer_size.checked_add(COLUMN_BLOCK - 1) else {
            return;
        };
        let padded_columns = padded_columns / COLUMN_BLOCK * COLUMN_BLOCK;
        let Some(panel_len) = self.inner.checked_mul(padded_columns) else {
            return;
        };
        let Some(scratch_len) = panel_len.checked_mul(TAPS) else {
            return;
        };
        self.right_scratch.resize(scratch_len, 0.0);
    }

    pub(super) fn process(
        &mut self,
        columns: usize,
        right: &[f32],
        right_offsets: [usize; TAPS],
        right_column_stride: usize,
        bias: &[f32],
        output: &mut [f32],
    ) -> bool {
        let Some(output_len) = self.rows.checked_mul(columns) else {
            return false;
        };
        let Some(padded_columns) = columns.checked_add(COLUMN_BLOCK - 1) else {
            return false;
        };
        let padded_columns = padded_columns / COLUMN_BLOCK * COLUMN_BLOCK;
        let Some(panel_len) = self.inner.checked_mul(padded_columns) else {
            return false;
        };
        let Some(scratch_len) = panel_len.checked_mul(TAPS) else {
            return false;
        };
        if right_column_stride < self.inner
            || bias.len() < self.rows
            || output.len() < output_len
            || self.right_scratch.len() < scratch_len
        {
            return false;
        }
        let Some(right_span) = right_column_stride
            .checked_mul(columns.saturating_sub(1))
            .and_then(|span| span.checked_add(self.inner))
        else {
            return false;
        };
        if right_offsets
            .iter()
            .any(|offset| match offset.checked_add(right_span) {
                Some(end) => end > right.len(),
                None => true,
            })
        {
            return false;
        }

        for (tap, offset) in right_offsets.into_iter().enumerate() {
            let scratch_start = tap * panel_len;
            pack_right(
                self.inner,
                columns,
                padded_columns,
                &right[offset..],
                right_column_stride,
                &mut self.right_scratch[scratch_start..scratch_start + panel_len],
            );
        }

        multiply_packed(
            self.rows,
            self.inner,
            columns,
            padded_columns,
            &self.weights,
            &self.right_scratch[..scratch_len],
            &bias[..self.rows],
            &mut output[..output_len],
        )
    }
}

fn supports_shape(rows: usize, inner: usize) -> bool {
    supports_arch() && rows == inner && matches!(rows, 8 | 16)
}

#[cfg(target_arch = "aarch64")]
fn supports_arch() -> bool {
    true
}

#[cfg(target_arch = "x86_64")]
fn supports_arch() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn supports_arch() -> bool {
    false
}

fn pack_weight_rows(
    packed: &mut Vec<f32>,
    taps: &[Vec<f32>],
    rows: usize,
    inner: usize,
    row_start: usize,
    block_rows: usize,
) {
    for tap in taps {
        for depth in 0..inner {
            let start = depth * rows + row_start;
            packed.extend_from_slice(&tap[start..start + block_rows]);
        }
    }
}

fn pack_right(
    inner: usize,
    columns: usize,
    padded_columns: usize,
    right: &[f32],
    right_column_stride: usize,
    scratch: &mut [f32],
) {
    let full_columns = columns / COLUMN_BLOCK * COLUMN_BLOCK;
    for column_block in (0..full_columns).step_by(COLUMN_BLOCK) {
        let packed_block =
            &mut scratch[column_block * inner..(column_block + COLUMN_BLOCK) * inner];
        for depth in 0..inner {
            for column_offset in 0..COLUMN_BLOCK {
                packed_block[depth * COLUMN_BLOCK + column_offset] =
                    right[(column_block + column_offset) * right_column_stride + depth];
            }
        }
    }
    if full_columns < padded_columns {
        let packed_block =
            &mut scratch[full_columns * inner..(full_columns + COLUMN_BLOCK) * inner];
        for depth in 0..inner {
            for column_offset in 0..COLUMN_BLOCK {
                let column = full_columns + column_offset;
                packed_block[depth * COLUMN_BLOCK + column_offset] = if column < columns {
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
fn multiply_packed(
    rows: usize,
    inner: usize,
    columns: usize,
    padded_columns: usize,
    weights: &[f32],
    right: &[f32],
    bias: &[f32],
    output: &mut [f32],
) -> bool {
    // SAFETY: PackedA1Conv validates every matrix extent. AArch64 always
    // provides NEON, and the shape dispatch fixes each register tile.
    unsafe {
        multiply_packed_neon(
            rows,
            inner,
            columns,
            padded_columns,
            weights,
            right,
            bias,
            output,
        );
    }
    true
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn multiply_packed_neon(
    rows: usize,
    inner: usize,
    columns: usize,
    padded_columns: usize,
    weights: &[f32],
    right: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    let panel_len = inner * padded_columns;
    for column_block in (0..columns).step_by(COLUMN_BLOCK) {
        let valid_columns = (columns - column_block).min(COLUMN_BLOCK);
        if rows == 16 {
            // SAFETY: The packed 16-channel layout is a 12-row panel followed
            // by a 4-row panel, with complete bias and output blocks.
            unsafe {
                multiply_block_neon::<3>(
                    inner,
                    column_block,
                    valid_columns,
                    0,
                    &weights[..TAPS * inner * 12],
                    right,
                    panel_len,
                    &bias[..12],
                    rows,
                    output,
                );
                multiply_block_neon::<1>(
                    inner,
                    column_block,
                    valid_columns,
                    12,
                    &weights[TAPS * inner * 12..],
                    right,
                    panel_len,
                    &bias[12..],
                    rows,
                    output,
                );
            }
        } else {
            // SAFETY: Shape validation restricts this branch to one 8-row tile.
            unsafe {
                multiply_block_neon::<2>(
                    inner,
                    column_block,
                    valid_columns,
                    0,
                    weights,
                    right,
                    panel_len,
                    bias,
                    rows,
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
unsafe fn multiply_block_neon<const ROW_PACKETS: usize>(
    inner: usize,
    column_block: usize,
    valid_columns: usize,
    output_row: usize,
    weights: &[f32],
    right: &[f32],
    panel_len: usize,
    bias: &[f32],
    output_stride: usize,
    output: &mut [f32],
) {
    use core::arch::aarch64::{vaddq_f32, vfmaq_laneq_f32, vld1q_f32, vmovq_n_f32, vst1q_f32};

    let block_rows = ROW_PACKETS * 4;

    let zero = vmovq_n_f32(0.0);
    let mut accumulators = [[zero; COLUMN_BLOCK]; ROW_PACKETS];
    for tap in 0..TAPS {
        let tap_weights = &weights[tap * inner * block_rows..];
        let tap_right = &right[tap * panel_len + column_block * inner..];
        for depth in 0..inner {
            let right_values = tap_right[depth * COLUMN_BLOCK..].as_ptr();
            // SAFETY: Each packed depth contains eight input columns.
            let (right_low, right_high) =
                unsafe { (vld1q_f32(right_values), vld1q_f32(right_values.add(4))) };
            for (packet, packet_accumulators) in accumulators.iter_mut().enumerate() {
                // SAFETY: Every packed depth contains ROW_PACKETS weight vectors.
                let weight =
                    unsafe { vld1q_f32(tap_weights[depth * block_rows + packet * 4..].as_ptr()) };
                packet_accumulators[0] =
                    vfmaq_laneq_f32(packet_accumulators[0], weight, right_low, 0);
                packet_accumulators[1] =
                    vfmaq_laneq_f32(packet_accumulators[1], weight, right_low, 1);
                packet_accumulators[2] =
                    vfmaq_laneq_f32(packet_accumulators[2], weight, right_low, 2);
                packet_accumulators[3] =
                    vfmaq_laneq_f32(packet_accumulators[3], weight, right_low, 3);
                packet_accumulators[4] =
                    vfmaq_laneq_f32(packet_accumulators[4], weight, right_high, 0);
                packet_accumulators[5] =
                    vfmaq_laneq_f32(packet_accumulators[5], weight, right_high, 1);
                packet_accumulators[6] =
                    vfmaq_laneq_f32(packet_accumulators[6], weight, right_high, 2);
                packet_accumulators[7] =
                    vfmaq_laneq_f32(packet_accumulators[7], weight, right_high, 3);
            }
        }
    }

    for (packet, packet_accumulators) in accumulators.iter().enumerate() {
        // SAFETY: The bias block contains one vector per row packet.
        let bias = unsafe { vld1q_f32(bias[packet * 4..].as_ptr()) };
        for (column, accumulator) in packet_accumulators.iter().enumerate().take(valid_columns) {
            let output_start = (column_block + column) * output_stride + output_row + packet * 4;
            // SAFETY: Every dispatched row packet and valid column is in output.
            unsafe {
                vst1q_f32(
                    output[output_start..].as_mut_ptr(),
                    vaddq_f32(*accumulator, bias),
                );
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn multiply_packed(
    rows: usize,
    inner: usize,
    columns: usize,
    padded_columns: usize,
    weights: &[f32],
    right: &[f32],
    bias: &[f32],
    output: &mut [f32],
) -> bool {
    // SAFETY: PackedA1Conv is constructed only after detecting AVX2 and FMA,
    // and it validates every matrix extent.
    unsafe {
        multiply_packed_avx2(
            rows,
            inner,
            columns,
            padded_columns,
            weights,
            right,
            bias,
            output,
        );
    }
    true
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn multiply_packed_avx2(
    rows: usize,
    inner: usize,
    columns: usize,
    padded_columns: usize,
    weights: &[f32],
    right: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    let panel_len = inner * padded_columns;
    for column_block in (0..columns).step_by(COLUMN_BLOCK) {
        let valid_columns = (columns - column_block).min(COLUMN_BLOCK);
        if rows == 16 {
            // SAFETY: Shape validation guarantees two complete AVX row packets.
            unsafe {
                multiply_block_avx2::<2>(
                    inner,
                    column_block,
                    valid_columns,
                    weights,
                    right,
                    panel_len,
                    bias,
                    rows,
                    output,
                );
            }
        } else {
            // SAFETY: Shape validation restricts this branch to one AVX row packet.
            unsafe {
                multiply_block_avx2::<1>(
                    inner,
                    column_block,
                    valid_columns,
                    weights,
                    right,
                    panel_len,
                    bias,
                    rows,
                    output,
                );
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn multiply_block_avx2<const ROW_PACKETS: usize>(
    inner: usize,
    column_block: usize,
    valid_columns: usize,
    weights: &[f32],
    right: &[f32],
    panel_len: usize,
    bias: &[f32],
    output_stride: usize,
    output: &mut [f32],
) {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_storeu_ps,
    };

    let block_rows = ROW_PACKETS * 8;

    let zero = _mm256_setzero_ps();
    let mut accumulators = [[zero; COLUMN_BLOCK]; ROW_PACKETS];
    for tap in 0..TAPS {
        let tap_weights = &weights[tap * inner * block_rows..];
        let tap_right = &right[tap * panel_len + column_block * inner..];
        for depth in 0..inner {
            let right_values = &tap_right[depth * COLUMN_BLOCK..];
            for (packet, packet_accumulators) in accumulators.iter_mut().enumerate() {
                // SAFETY: Every packed depth contains ROW_PACKETS weight vectors.
                let weight = unsafe {
                    _mm256_loadu_ps(tap_weights[depth * block_rows + packet * 8..].as_ptr())
                };
                for column in 0..COLUMN_BLOCK {
                    let input = _mm256_set1_ps(right_values[column]);
                    packet_accumulators[column] =
                        _mm256_fmadd_ps(weight, input, packet_accumulators[column]);
                }
            }
        }
    }

    for (packet, packet_accumulators) in accumulators.iter().enumerate() {
        // SAFETY: The bias block contains one vector per row packet.
        let bias = unsafe { _mm256_loadu_ps(bias[packet * 8..].as_ptr()) };
        for (column, accumulator) in packet_accumulators.iter().enumerate().take(valid_columns) {
            let output_start = (column_block + column) * output_stride + packet * 8;
            // SAFETY: Every dispatched row packet and valid column is in output.
            unsafe {
                _mm256_storeu_ps(
                    output[output_start..].as_mut_ptr(),
                    _mm256_add_ps(*accumulator, bias),
                );
            }
        }
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[allow(clippy::too_many_arguments)]
fn multiply_packed(
    _rows: usize,
    _inner: usize,
    _columns: usize,
    _padded_columns: usize,
    _weights: &[f32],
    _right: &[f32],
    _bias: &[f32],
    _output: &mut [f32],
) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::{supports_arch, PackedA1Conv, COLUMN_BLOCK, TAPS};

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn assert_matches_reference(rows: usize, columns: usize) {
        if !supports_arch() {
            return;
        }
        let inner = rows;
        let taps = (0..TAPS)
            .map(|tap| {
                (0..rows * inner)
                    .map(|index| {
                        let signed = ((index * 17 + tap * 13) % 29) as f32 - 14.0;
                        signed * 0.007_812_5
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let stride = inner + 3;
        let tap_span = stride * (columns + 2);
        let offsets = [0, tap_span, tap_span * 2];
        let right = (0..tap_span * TAPS)
            .map(|index| {
                let signed = ((index * 11 + 5) % 31) as f32 - 15.0;
                signed * 0.015_625
            })
            .collect::<Vec<_>>();
        let bias = (0..rows)
            .map(|row| row as f32 * 0.003_906_25 - 0.02)
            .collect::<Vec<_>>();

        let mut packed =
            PackedA1Conv::new(rows, inner, &taps).expect("supported shape should pack");
        packed.set_max_buffer_size(columns);
        let mut actual = vec![f32::NAN; rows * columns];
        assert!(packed.process(columns, &right, offsets, stride, &bias, &mut actual));

        let mut expected = vec![0.0; rows * columns];
        for column in 0..columns {
            for row in 0..rows {
                let mut value = 0.0_f32;
                for tap in 0..TAPS {
                    for depth in 0..inner {
                        value = taps[tap][depth * rows + row]
                            .mul_add(right[offsets[tap] + column * stride + depth], value);
                    }
                }
                expected[column * rows + row] = value + bias[row];
            }
        }

        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 2.0e-6,
                "element {index}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn packed_kernels_match_reference_for_complete_and_tail_tiles() {
        for rows in [8, 16] {
            assert_matches_reference(rows, COLUMN_BLOCK * 2);
            assert_matches_reference(rows, COLUMN_BLOCK + 3);
        }
    }

    #[test]
    fn rejects_non_a1_shapes_and_malformed_weights() {
        let valid_taps = vec![vec![0.0; 8 * 8]; TAPS];
        assert!(PackedA1Conv::new(7, 7, &valid_taps).is_none());
        assert!(PackedA1Conv::new(8, 16, &valid_taps).is_none());
        assert!(PackedA1Conv::new(8, 8, &valid_taps[..2]).is_none());

        let mut malformed_taps = valid_taps;
        malformed_taps[1].pop();
        assert!(PackedA1Conv::new(8, 8, &malformed_taps).is_none());
    }

    #[test]
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn rejects_scratch_and_buffer_extents_that_are_too_small() {
        if !supports_arch() {
            return;
        }
        let taps = vec![vec![0.0; 8 * 8]; TAPS];
        let mut packed = PackedA1Conv::new(8, 8, &taps).expect("shape should pack");
        let right = vec![0.0; 8 * 4 * TAPS];
        let bias = vec![0.0; 8];
        let mut output = vec![0.0; 8 * 4];

        assert!(!packed.process(4, &right, [0, 32, 64], 8, &bias, &mut output));
        packed.set_max_buffer_size(4);
        assert!(!packed.process(4, &right[..95], [0, 32, 64], 8, &bias, &mut output));
        assert!(!packed.process(4, &right, [0, 32, 64], 7, &bias, &mut output));
        assert!(!packed.process(4, &right, [0, 32, 64], 8, &bias[..7], &mut output));
        assert!(!packed.process(4, &right, [0, 32, 64], 8, &bias, &mut output[..31]));
    }
}
