# Optimization Plan

Last updated: 2026-03-14

## MUST DO: Block-Based Matrix Processing for WaveNet

### Problem

The `wavenet_a2_max` model (which uses every advanced feature: FiLM, grouped convolutions, gating, blending, head1x1, nested condition_dsp) diverges from C++ output by ~0.75 max diff. All 5 standard/simpler models match C++ to within 1e-6 or better.

### Root Cause

Our WaveNet processes audio **per-sample using scalar loops**. The C++ processes in **blocks of N frames using Eigen matrix operations** (GEMM). While these are mathematically equivalent, floating-point arithmetic is not associative — the order of additions in a dot product affects rounding. These per-operation rounding differences (on the order of 1e-7 each) compound exponentially through the deep network when weights are large.

This was verified exhaustively:

| Evidence | Result |
|----------|--------|
| C++(-O2) vs C++(-Ofast) | 7e-07 diff — **not** caused by FMA/fast-math |
| Rust with `f32::mul_add` (FMA) | No improvement — **not** caused by FMA |
| Same model, weights scaled to 10% | 4.4e-11 diff — **confirms** FP accumulation |
| Same features, random weights at 0.1 scale | 2.3e-10 diff — **confirms** features are correct |
| Zero input after prewarm | 0.09 diff — prewarm itself accumulates different state |
| All individual features tested in isolation | Perfect match — **no logic bugs remain** |

The error grows exponentially with weight magnitude:

| Weight Scale | Max Diff |
|-------------|----------|
| 0.01 | 0.00e+00 |
| 0.05 | 9.09e-13 |
| 0.10 | 4.37e-11 |
| 0.20 | 2.91e-09 |
| 0.50 | 5.24e-05 |
| 0.80 | 8.77e-03 |
| 1.00 | 8.37e-02 |

Standard NAM models (which use Tanh activation and no FiLM/gating) don't exhibit this because:
1. Tanh saturates, naturally bounding intermediate values
2. No FiLM means fewer chained operations per layer
3. Smaller effective channel counts in the critical paths

### Required Fix

Convert the WaveNet forward pass from per-sample scalar processing to block-based matrix processing that matches C++ Eigen's computation order. This means:

1. **`process()` receives a block of N frames** and processes them all at once (matching C++'s `process(input, output, num_frames)`)

2. **Conv1D uses batch GEMM**: Write all N input frames to ring buffer, then for each kernel tap, read an (in_channels x N) matrix from the ring buffer and compute `output += weight[k] @ tap_matrix`. This replaces per-sample dot products with matrix multiplies that use the same accumulation order as Eigen.

3. **Conv1x1 uses batch GEMM**: `output_matrix = weight @ input_matrix` over all N frames simultaneously, rather than N separate matrix-vector products.

4. **FiLM operates on matrices**: `scale_shift = conv1x1(condition_matrix)` then elementwise `output = input * scale + shift` over all frames.

5. **Activations applied elementwise over matrices**: Same as now, but on (channels x N) matrices instead of per-sample vectors.

6. **Ring buffer supports batch write/read**: Write N frames at once, read (channels x N) matrices at dilated offsets.

### Data Structures

Replace per-sample scratch buffers (`Vec<f32>`) with 2D matrices:

```rust
// Current (per-sample)
conv_out_buf: Vec<f32>,       // [max_conv_out]

// Target (block-based)
conv_out_buf: Array2<f32>,    // [max_conv_out, max_block_size]
```

Use `ndarray::Array2<f32>` with shape `(channels, num_frames)` to match Eigen's column-major layout.

### Expected Benefits

1. **Bit-identical output** to C++ for all models including a2_max
2. **Significant performance improvement** — GEMM is much faster than N separate dot products due to cache locality and SIMD
3. **Potential BLAS acceleration** — ndarray can use openblas/accelerate for the matrix multiplies
4. **Better vectorization** — batch operations auto-vectorize more easily than per-sample loops

### Impact

This change affects only `wavenet.rs` internals. The `Dsp::process()` API already takes a slice of samples, so no external interface changes are needed. The ring buffer, Conv1x1, Conv1d, and FiLM structs need block-aware methods alongside the existing per-sample ones (or replace them entirely).

### Priority

**Must do.** This is the only remaining issue preventing bit-identical output for all models. It also happens to be the single highest-impact performance optimization (WaveNet's `process()` is where 95%+ of CPU time is spent).

---

## Other Optimization Items

### Profile the WaveNet Hot Path

Use `samply` or Instruments to identify bottlenecks. Current performance:

| Model | RTF | Headroom |
|-------|-----|----------|
| Small WaveNet (test) | 0.021x | 47.7x |
| Standard WaveNet (16/8ch) | 0.74x | 1.3x |
| LSTM | 0.010x | 97.8x |
| a2_max (with condition_dsp) | 0.16x | 6.1x |

The standard WaveNet at 1.3x headroom is tight for real-time use. Block processing will help significantly.

### Optional BLAS Backend

Add optional BLAS support for the matrix multiplies:

```toml
[features]
blas = ["ndarray/blas", "blas-src/openblas"]
```

On macOS, Apple's Accelerate framework provides optimized BLAS. This would further speed up the GEMM operations after the block processing conversion.

### Zero-Allocation Verification

Use nih-plug's `assert_process_allocs` feature or a custom global allocator to verify no heap allocations occur during `process()`. Current code pre-allocates all buffers in `from_config()`, but the condition_dsp path has a `vec![]` allocation that should be eliminated.

### CI Pipeline

Create `.github/workflows/ci.yml` with:
- `cargo build` on Linux, macOS, Windows
- `cargo test` (all 117+ tests)
- `cargo clippy`
- `cargo fmt --check`
- Regression tests against C++ reference WAVs
