# Performance Analysis: Rust vs C++ NAM

Last updated: 2026-07-29

## Historical Performance Numbers

Processing 2 seconds of audio at 48kHz, buffer size 64 samples (matching C++ benchmodel).

| Model | C++ (no fast_tanh) | Rust | Ratio |
|-------|--------------------|------|-------|
| Small WaveNet (3/2ch) | 29ms | 21ms | Rust 1.4x **faster** |
| LSTM (hidden=3) | 18ms | 21ms | ~Tied |
| Standard WaveNet (16/8ch) | 280ms | 399ms | Rust 1.4x slower |
| a2_max (all features) | 146ms | 257ms | Rust 1.8x slower |

C++ compiled with `-Ofast` but without fast_tanh (polynomial tanh approximation disabled) for fair comparison. Rust compiled with `target-cpu=native`.

These wall-clock measurements predate the July 2026 parity work. Criterion and
Callgrind now provide the maintained performance record.

## Current Accuracy Numbers

| Model | Max Diff vs C++ |
|-------|----------------|
| wavenet | 0.0 (bit-identical) |
| wavenet_condition_dsp | 0.0 (bit-identical) |
| lstm | 8.94e-08 |
| wavenet_a1_standard | 5.96e-08 |
| my_model | 5.96e-08 |
| wavenet_a2_max | 0.0 (bit-identical, Apple Silicon default backend) |

The A2 max result is checked against a render produced by pinned
NeuralAmpModelerCore commit
`3cde95c354d5ba6da01316cad90b05cfc4855053`. Its complete 96,000-sample
render matches bit for bit with the default matrix backend.

## Maintained Performance Gates

Criterion benchmarks A1 standard and A2 max at 32, 64, 128, and 256-sample
callbacks. It also compares direct and FFT Linear processing from 256 through
16,384 taps.

The allocation-free WaveNet run on Apple Silicon produced:

| Callback | Default A1 | Default A2 | `fast-kernels` A1 | `fast-kernels` A2 |
|---------:|-----------:|-----------:|------------------:|------------------:|
| 32 samples | 58.2 us | 31.7 us | 37.7 us | 19.5 us |
| 64 samples | 115.4 us | 62.4 us | 72.9 us | 38.0 us |
| 128 samples | 229.6 us | 119.1 us | 143.4 us | 75.6 us |
| 256 samples | 456.8 us | 245.7 us | 286.7 us | 151.2 us |

Against the immediately preceding backend, the default build stayed within
4.0% at every measured size. The `fast-kernels` build stayed within 7.0%.
The new path preserves the bit-identical A2 render on Apple Silicon.

Linux CI enforces these Callgrind budgets independently for the default,
`faer`, and `fast-kernels` builds:

- A1 and A2 instruction baselines at 64 samples.
- Per-sample scaling envelopes at 16, 32, 128, and 256 samples.
- Absolute instruction ceilings for 4,096-tap and 16,384-tap FFT Linear
  callbacks that trigger a partition transform.
- An absolute instruction ceiling for the 64-sample LSTM callback.

The budget files are pinned to Linux x86-64, and the checker rejects attempts
to apply them on another architecture. Bench profiles retain symbols so a
budget change can be traced to the functions and source lines that caused it.

## Optimization Journey

### What worked

| Optimization | Effect | Accuracy impact |
|-------------|--------|-----------------|
| Block-based processing | Per-sample → block GEMM. a2_max: 0.75 → 4.77e-06 diff | Massive improvement |
| matrixmultiply::sgemm | 3x speedup on standard WaveNet (1200ms → 399ms) | None on small models, small change on a2_max |
| Reset-sized matrix scratch and allocation-free 8x8 kernels | Removed 82 A1 and 14 A2 allocations per 128-sample callback | Preserves bit-identical A2 output on Apple Silicon |
| sgemm threshold=64 | Better than threshold=32 for a2_max (350ms → 257ms) | None |
| target-cpu=native | ~2x improvement early on | None |
| Slice-based bounds elimination | Minor improvement | None |

### What didn't work

| Attempt | Result | Why |
|---------|--------|-----|
| Profile-guided optimization (PGO) | 22-53% **slower** | Profile data caused over-specialization. LLVM's inlining decisions increased code size and hurt instruction cache. |
| Fat LTO (`lto = "fat"`) | 50% slower on standard WaveNet | Over-inlining of matrixmultiply internals blew up code size. The sgemm function is better left as an external call. |
| Thin LTO (`lto = "thin"`) | 7% slower | Same issue to a lesser degree. |
| codegen-units=1 | 6% slower | Single codegen unit forced LLVM to compile everything as one blob, hurting cache locality. |
| f32::mul_add (FMA) | No change | The compiler was already using FMA where beneficial via target-cpu=native. |
| Axpy loop without sgemm | 50% **slower** on standard WaveNet | The compiler's auto-vectorizer couldn't match matrixmultiply's hand-tuned SIMD micro-kernels for 16x16 matrices. |
| Axpy loop replacing sgemm | Faster on some, but a2_max accuracy degraded 4.77e-06 → 6.20e-06 | Different FP accumulation order compounds through a2_max's deep network with large weights. |
| fast_tanh polynomial | Only 11% speedup | tanh is ~11% of total time, not the bottleneck. GEMM dominates. |

### Small-matrix SIMD evaluation

On 2026-07-29, A1 callback profiles were collected at 32, 64, 128, and 256
frames before implementing architecture-specific small-matrix kernels.

On Apple M4, `Conv1x1::process_block_small_gemm` accounted for 5.03% to 6.68%
of callback samples. A NEON prototype was compared with the scalar build in
paired Criterion runs:

| Frames | Full callback change |
|-------:|---------------------:|
| 32 | 0.34% faster |
| 64 | 6.50% slower |
| 128 | 0.09% faster |
| 256 | 2.91% faster |

On Linux x86-64, the same Conv1x1 path accounted for 10.17% to 10.27% of
Callgrind instructions, including software `fmaf` calls. An AVX2/FMA prototype
also vectorized the single-output layer head:

| Frames | Scalar instructions | AVX2/FMA instructions | Improvement |
|-------:|--------------------:|----------------------:|------------:|
| 32 | 2,048,948 | 1,881,179 | 8.19% |
| 64 | 4,110,634 | 3,771,629 | 8.25% |
| 128 | 8,237,169 | 7,555,812 | 8.27% |
| 256 | 16,477,384 | 15,111,102 | 8.29% |

Neither prototype met the repository requirement of at least 10% improvement
for the complete callback on a supported platform. Both were removed. The
scalar fallback remains the default small-matrix implementation.

## Key Insights

### 1. The bottleneck is small-matrix GEMM

The standard WaveNet's hot path is 16x16 and 8x8 matrix-vector multiplies with
multiple frames batched. These are too small for general-purpose BLAS to be
optimal, but too large for scalar code to be competitive. The current backend
uses reset-sized packing scratch and an 8x8 NEON kernel on AArch64. Other
architectures use allocation-free `nano-gemm` microkernels.

### 2. Accuracy and performance trade off through FP accumulation order

Any change to the GEMM inner loop order changes which floating-point rounding errors occur. For models with small weights or few layers (wavenet, wavenet_condition_dsp), the differences are zero or negligible. For the a2_max model (5+ layers, 8 FiLM modules per layer, large weights), even 1 ULP difference per operation compounds to ~6e-06 through the deep network. This is still inaudible (-106 dB) but measurable.

### 3. Compiler optimizations often hurt for this workload

PGO, LTO, and codegen-units=1 all made things worse. The workload is dominated by tight numerical loops where instruction cache locality matters more than cross-function optimization. The default Rust release settings (16 codegen units, no LTO) produce the best results because they keep the hot code compact.

### 4. tanh is not the bottleneck

Despite C++ getting 1.5-2x speedup from fast_tanh, in my Rust implementation tanh only accounts for ~11% of total time. The difference is that C++ Eigen's GEMM is so fast that tanh becomes a significant fraction of the remaining time, while my GEMM is slower so it dominates.

### 5. Small models match or beat C++

For models with channels ≤ 8 (small WaveNet, LSTM), my scalar dot-product loops are competitive with or faster than C++. The gap only appears at channels ≥ 16 where Eigen's SIMD micro-kernels have a significant advantage.

## Remaining performance work

The allocation-free backend covers the batched matrix path. The measured NEON
and AVX2/FMA small-matrix prototypes did not clear the retention threshold.
Further A1 work should begin with a new full-callback profile and target a path
with more than 10% achievable headroom.

## Real-world impact of the performance gap

At 48kHz with 64-sample buffers, the DAW gives 1.33ms per buffer.

| Model | Rust per buffer | Budget used |
|-------|----------------|-------------|
| Standard WaveNet | 0.27ms | 20% |
| a2_max | 0.17ms | 13% |

Both are well within real-time requirements. The 1.4x gap means C++ uses 14%
of the CPU budget while Rust uses 20%. The difference only matters when running
multiple simultaneous NAM instances on weak hardware.
