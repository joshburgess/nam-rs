pub(super) const SGEMM_MIN_SIZE: usize = 64;

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn sgemm_colmajor(
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    b_col_stride: usize,
    beta: f32,
    c: &mut [f32],
) {
    let a_len = m
        .checked_mul(k)
        .unwrap_or_else(|| panic!("matrix A dimensions overflow: {m} x {k}"));
    let b_len = b_col_stride
        .checked_mul(n)
        .unwrap_or_else(|| panic!("matrix B dimensions overflow: {b_col_stride} x {n}"));
    let c_len = m
        .checked_mul(n)
        .unwrap_or_else(|| panic!("matrix C dimensions overflow: {m} x {n}"));
    assert!(
        b_col_stride >= k,
        "matrix B stride {b_col_stride} is smaller than its active row count {k}"
    );
    assert!(
        a.len() >= a_len,
        "matrix A needs {a_len} elements but has {}",
        a.len()
    );
    assert!(
        b.len() >= b_len,
        "matrix B needs {b_len} elements but has {}",
        b.len()
    );
    assert!(
        c.len() >= c_len,
        "matrix C needs {c_len} elements but has {}",
        c.len()
    );
    let a = &a[..a_len];
    let b = &b[..b_len];
    let c = &mut c[..c_len];

    #[cfg(feature = "faer")]
    {
        let a_mat = faer::mat::from_column_major_slice::<f32, usize, usize>(a, m, k);
        let b_mat = faer::mat::from_column_major_slice::<f32, usize, usize>(b, b_col_stride, n);
        let b_mat = b_mat.subrows(0, k);
        let c_mat = faer::mat::from_column_major_slice_mut::<f32, usize, usize>(c, m, n);

        faer::linalg::matmul::matmul(
            c_mat,
            a_mat,
            b_mat,
            Some(beta),
            alpha,
            faer::Parallelism::None,
        );
    }

    #[cfg(not(feature = "faer"))]
    {
        // SAFETY: The validated slices cover the column-major matrices, and C is exclusive.
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                1,
                m as isize,
                b.as_ptr(),
                1,
                b_col_stride as isize,
                beta,
                c.as_mut_ptr(),
                1,
                m as isize,
            );
        }
    }
}
