# Optimization Plan

Last updated: 2026-03-15

## Completed

### Block-Based Matrix Processing for WaveNet

Converted the WaveNet forward pass from per-sample scalar processing to block-based matrix processing matching C++ Eigen's computation order.

**Accuracy (vs C++ reference):**

| Model | Max Diff |
|-------|----------|
| wavenet | bit-identical |
| wavenet_condition_dsp | bit-identical |
| lstm | 2.09e-07 |
| wavenet_a1_standard | 1.13e-06 |
| my_model | 1.13e-06 |
| wavenet_a2_max | 4.77e-06 |

**Performance (processing 2 seconds of audio at 48kHz, buffer_size=64):**

| Model | C++ (no fast_tanh) | Rust | Ratio |
|-------|--------------------|------|-------|
| Small WaveNet | 29ms | ~20ms | Rust faster |
| LSTM | 18ms | ~20ms | Tied |
| Standard WaveNet (16/8ch) | 280ms | ~1200ms | 4.3x slower |
| a2_max | 146ms | ~390ms | 2.7x slower |

### Plugin GUI

Added egui-based GUI with native file browser for loading .nam model files, input/output gain sliders, and model status indicator.

### Strict Accuracy Guards

Per-model regression tests with exact thresholds that prevent any accuracy degradation:
- wavenet, wavenet_condition_dsp: must be bit-identical (0.0)
- lstm: <= 2.1e-07
- wavenet_a1_standard, my_model: <= 1.2e-06
- wavenet_a2_max: <= 5.0e-06

### target-cpu=native

Added to `.cargo/config.toml` for automatic SIMD optimization.

---

## Remaining Performance Gap

The standard WaveNet is ~4.3x slower than C++ (without fast_tanh). The gap comes from:

1. **Eigen's SIMD-optimized GEMM micro-kernels** — Eigen has hand-written NEON/SSE intrinsics with register blocking and cache tiling for small matrix sizes (8x8, 16x16). Our loops rely on the compiler's auto-vectorizer.

2. **C++ fast_tanh** — a polynomial approximation that's ~3x faster than `std::tanh` but introduces ~3e-04 error per call. We use `std::tanh` for accuracy. This accounts for roughly 1.5-2x of C++'s speed advantage.

### Potential Approaches (accuracy-preserving only)

**BLAS backend** — Use `matrixmultiply::sgemm` or Apple Accelerate / OpenBLAS for the GEMM operations. Previous testing showed sgemm gives ~3x speedup on the standard WaveNet but changes the floating-point accumulation order, breaking bit-identical results on small models. A careful implementation that only applies sgemm to large matrices (where accuracy is already at the f32 floor) could work, but must pass all accuracy guard tests.

**Hand-written SIMD** — Write NEON/SSE intrinsics for the specific matrix sizes used by NAM models (16x16, 16x8, 8x8, 8x4). Most complex but would give Eigen-equivalent performance without accuracy tradeoff.

**Profile-guided optimization** — Use `cargo pgo` to optimize the hot path based on actual NAM workloads.

---

## Decided Against

### FastTanh Polynomial Approximation

C++ gets ~1.5-2x speedup from replacing `std::tanh` with a rational polynomial approximation (3e-04 max error per call). This compounds through the network and degrades audio quality. Since accuracy is prioritized over performance, we use `std::tanh`.

---

## Future Items

### Sample Rate Conversion

When the model's trained sample rate differs from the host sample rate, the plugin should resample. The C++ plugin (iPlug2 wrapper) handles this; our plugin currently does not. Would require a resampling library (e.g. `rubato`).

### Criterion Benchmarks

The benchmark harness is currently a placeholder. Should have proper benchmarks for all model types at various buffer sizes with saved baselines for regression detection.
