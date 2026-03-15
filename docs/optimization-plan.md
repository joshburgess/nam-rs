# Optimization Plan

Last updated: 2026-03-15

## Completed

### Block-Based Matrix Processing for WaveNet

Converted the WaveNet forward pass from per-sample scalar processing to block-based matrix processing matching C++ Eigen's computation order. Uses `ColMajorMatrix` and `RingBuffer2D` for batch GEMM operations.

**Results:**

| Model | Before (per-sample) | After (block) |
|-------|---------------------|---------------|
| wavenet | 8.2e-08 | 0 (bit-identical) |
| wavenet_condition_dsp | 8.9e-08 | 0 (bit-identical) |
| wavenet_a1_standard | 2.1e-06 | 1.1e-06 |
| my_model | 2.1e-06 | 1.1e-06 |
| wavenet_a2_max | **0.75** | **4.8e-06** |

**Performance improvement:**

| Model | Before | After | Improvement |
|-------|--------|-------|-------------|
| Standard WaveNet (16/8ch) | 0.74x RTF (1.3x headroom) | 0.54x RTF (1.8x headroom) | 38% faster |
| a2_max (with condition_dsp) | 0.16x RTF (6.1x headroom) | 0.14x RTF (7.0x headroom) | 12% faster |

---

## Remaining Precision Analysis

All models are now at or near the f32 precision floor. The remaining differences are fundamental to IEEE 754 single-precision arithmetic, not implementation bugs.

### wavenet_a1_standard / my_model: 1.1e-06 — At f32 Precision Floor

These models have 20 layers across 2 arrays. Each layer introduces ~1-2 ULPs (Units in the Last Place) of rounding difference. The theoretical minimum error:

```
Output magnitude: ~0.44
f32 ULP at 0.44:  2.98e-08
20 layers × 2 ULP: 1.19e-06  (theoretical floor)
Actual error:      1.13e-06  (we're AT the floor)
```

**Not improvable** without switching to f64 internal computation, which would halve performance and diverge from C++'s f32 pipeline.

### LSTM: 2.1e-07 — ndarray BLAS vs Eigen Mat-Vec

The LSTM uses `ndarray::linalg::general_mat_vec_mul` for the combined weight matrix × state vector product. ndarray's internal BLAS implementation may accumulate the dot product in a slightly different order than Eigen.

**Potentially improvable** by replacing `general_mat_vec_mul` with a manual dot product loop matching Eigen's exact iteration order (row-by-row, left-to-right accumulation). This would be a small change to `lstm.rs` but the gain is marginal (2.1e-07 → ~0, for a model with hidden_size=3).

### wavenet_a2_max: 4.8e-06 — condition_dsp Per-Sample Processing

The nested condition_dsp WaveNet is still processed per-sample via `process_sample_multi_channel()` because the `Dsp` trait's multi-channel output interface doesn't support block processing. The outer WaveNet does block processing, but its condition signal comes from per-sample processing of the nested model.

**Potentially improvable** by either:
1. Adding a `process_block_multi_channel(&mut self, input: &[Sample], output: &mut [&mut [f32]])` method to the `Dsp` trait, allowing the condition_dsp to process blocks with multi-channel output
2. Or by storing the condition_dsp as a concrete `WaveNet` (not `Box<dyn Dsp>`) so we can call its block processing methods directly

This would bring the condition_dsp processing in line with the C++ block approach and likely reduce the error to the f32 ULP floor (~1e-06 given the network depth).

---

## Future Performance Optimizations

### Optional BLAS Backend

Add optional BLAS support for the GEMM operations:

```toml
[features]
blas = ["blas-src/openblas"]
# Or on macOS:
blas = ["blas-src/accelerate"]
```

The current hand-written GEMM loops are correct but not SIMD-optimized. A BLAS backend (OpenBLAS, Accelerate, or Intel MKL) would significantly speed up the matrix multiplies, especially for larger models.

### Profiling

Use `samply` or Instruments to identify remaining bottlenecks. Current performance:

| Model | RTF | Headroom |
|-------|-----|----------|
| Small WaveNet (test) | ~0.02x | ~50x |
| Standard WaveNet (16/8ch) | 0.54x | 1.8x |
| LSTM | ~0.01x | ~98x |
| a2_max (with condition_dsp) | 0.14x | 7.0x |

The standard WaveNet at 1.8x headroom could benefit from BLAS acceleration.

### Zero-Allocation Verification

Use nih-plug's `assert_process_allocs` feature or a custom global allocator to verify no heap allocations occur during `process()`.

### Plugin GUI

Add a file browser for loading .nam files in a DAW. Currently the plugin is headless (no GUI). Options: vizia, egui, or iced via nih-plug's GUI framework support.

### Sample Rate Conversion

When the model's trained sample rate differs from the host sample rate, the plugin should resample. The C++ plugin handles this; our plugin currently does not.
