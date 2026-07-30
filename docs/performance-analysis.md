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
| wavenet_a1_standard | 7.97e-07 ARM64, 1.02e-06 x86-64 |
| my_model | 7.97e-07 ARM64, 1.02e-06 x86-64 |
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

### Conv1d matrix-path evaluation

On 2026-07-29, symbolized native Linux x86-64 Callgrind profiles and Apple M4
Criterion measurements were used to split the standard A1 Conv1d path.

The native x86-64 profiles measured the complete callback at every maintained
size:

| Frames | Instructions | Conv1d inclusive | Matrix input packing | Bias arithmetic |
|-------:|-------------:|-----------------:|---------------------:|----------------:|
| 32 | 2,048,948 | 22.05% | 5.15% | 0.92% |
| 64 | 4,110,634 | 22.46% | 5.13% | 0.91% |
| 128 | 8,237,169 | 22.69% | 5.12% | 0.91% |
| 256 | 16,477,384 | 22.68% | 5.12% | 0.91% |

The remaining Conv1d cost consists of direct `nano-gemm` multiplication,
output accumulation, ring-buffer writes, tap selection, validation, and loop
overhead. The immutable column-major weights are not repacked in the callback.
Inputs are transposed once per tap into reset-sized 8-column panels.
`nano-gemm` creates a small dispatch plan per panel, accounting for about 1.3%
of the complete 64-frame callback.

A prototype concatenated the three immutable tap matrices at model load,
packed all three input views into one scratch layout, and evaluated one
combined-inner GEMM per convolution. It remained allocation-free and passed
the matrix oracle, exact callback-partition and reset tests, and both upstream
A2 render fixtures. It increased the complete A1 callback time:

| Frames | Apple M4 baseline | Fused taps | Change |
|-------:|------------------:|-----------:|-------:|
| 32 | 58.591 us | 62.377 us | 6.46% slower |
| 64 | 115.30 us | 123.74 us | 7.32% slower |
| 128 | 228.87 us | 245.32 us | 7.19% slower |
| 256 | 454.67 us | 479.47 us | 5.45% slower |

The same x86-64 binary was measured under Rosetta to exercise the non-AArch64
backend before rejection:

| Frames | Baseline | Fused taps | Change |
|-------:|---------:|-----------:|-------:|
| 32 | 151.38 us | 152.39 us | 0.67% slower |
| 64 | 298.63 us | 300.35 us | 0.58% slower |
| 128 | 595.34 us | 597.74 us | 0.40% slower |
| 256 | 1.1895 ms | 1.2103 ms | 1.75% slower |

The longer accumulator dependency chain and more complex tap-aware packing
cost more than the saved output passes. The prototype was removed. Packing,
plan construction, and bias together provide less than 10% theoretical
full-callback headroom, so isolated changes to those operations cannot meet
the retention gate. A future attempt would need a complete architecture-
specific packed-weight kernel that also replaces most of the multiplication.

### Packed A1 Conv1d kernels

The complete packed-weight design was evaluated against
NeuralAmpModelerCore commit
`3cde95c354d5ba6da01316cad90b05cfc4855053` and its Eigen commit
`bc3b39870ecb690a623a3f49149a358b95c5781d`. The model loader now packs the
three convolution taps into the native register-tile layout. The callback
packs input panels into reset-sized scratch, accumulates all three taps while
the output tile remains in registers, adds bias, and stores once.

The ARM64 layout follows Eigen's 12x8 plus 4x8 decomposition for a 16x16
matrix and its 8x8 decomposition for an 8x8 matrix. The x86-64 AVX2/FMA
layout uses 16x4 and 8x4 tiles. Unsupported shapes and x86-64 processors
without AVX2/FMA continue through the existing backend.

Fresh Apple M4 Criterion runs measured the complete A1 callback:

| Frames | Previous backend | Packed kernel | Improvement |
|-------:|-----------------:|--------------:|------------:|
| 32 | 76.361 us | 56.204 us | 26.4% |
| 64 | 149.98 us | 109.71 us | 26.9% |
| 128 | 308.98 us | 219.47 us | 29.0% |
| 256 | 589.10 us | 435.02 us | 26.2% |

ARM64 clears the repository's 10% full-callback retention gate at every
maintained size. Native Linux x86-64 instruction counts are enforced by CI;
Rosetta does not expose the AVX2/FMA features required by the x86-64 kernel.

The native Linux x86-64 Callgrind run measured:

| Frames | Previous backend | Packed kernel | Improvement |
|-------:|-----------------:|--------------:|------------:|
| 32 | 2,048,948 | 1,900,766 | 7.2% |
| 64 | 4,110,634 | 3,784,468 | 7.9% |
| 128 | 8,237,169 | 7,555,533 | 8.3% |
| 256 | 16,477,384 | 15,121,451 | 8.2% |

The x86-64 result does not independently clear 10%, but the implementation is
retained because the ARM64 full-callback result clears the cross-platform
retention rule. A dedicated native harness asserts AVX2 and FMA before loading
the model. Its measurements agree within 0.02% with the portable CI harness.
The portable harness's glibc capability mask stabilizes library dispatch, but
does not suppress Rust's CPUID-based packed-kernel detection.

### Post-kernel A1 callback profile and fusion gate

After the packed kernel landed, fresh symbolized Apple M4 profiles measured
complete A1 callbacks at every maintained buffer size:

| Frames | Activation | Conv1d | Conv1x1 | Layer vector operations | Head accumulation |
|-------:|-----------:|-------:|--------:|------------------------:|------------------:|
| 32 | 39.5% | 22.5% | 15.1% | 18.4% | 4.4% |
| 64 | 39.6% | 23.2% | 14.7% | 18.2% | 4.2% |
| 128 | 39.3% | 22.5% | 15.3% | 18.7% | 4.1% |
| 256 | 39.6% | 22.7% | 14.2% | 18.9% | 4.4% |

Native Linux x86-64 Callgrind profiles from the same post-kernel revision
provided instruction-level attribution:

| Frames | Activation | Packed multiply | Input packing | Mixin add | Residual add | Ring buffer | Head accumulation |
|-------:|-----------:|----------------:|--------------:|----------:|-------------:|------------:|------------------:|
| 32 | 43.66% | 10.00% | 6.08% | 6.07% | 5.16% | 1.26% | 3.65% |
| 64 | 43.85% | 10.01% | 6.05% | 6.09% | 5.18% | 1.22% | 3.66% |
| 128 | 43.92% | 10.01% | 6.03% | 6.10% | 5.19% | 1.20% | 3.66% |
| 256 | 43.88% | 9.99% | 6.02% | 6.10% | 5.18% | 1.29% | 3.66% |

Activation includes the activation loops and their `tanhf` and `expm1f`
callees. Ring-buffer attribution includes the inlined write path and its
libc copy instructions. The remaining instructions are primarily Conv1x1
matrix work, packed-kernel dispatch, output handling, and loop overhead.

Temporary symbol boundaries split the 256-frame Conv1d path into 10.9% packed
multiplication, 7.2% input-panel packing, and 4.4% ring-buffer handling,
dispatch, and call overhead. Source-line attribution split the layer vector
operations into 10.4% for the convolution-plus-mixin pass and about 8.4% for
the residual pass. The proportions remained stable across callback sizes.

Three exact, allocation-free fusion designs were evaluated:

| Prototype | Complete callback result |
|-----------|--------------------------|
| Convolution-plus-mixin, accurate tanh, and head accumulation in one loop | Neutral at 32 and 64 frames, slower at 128 frames, and 8.1% slower at 256 frames |
| Convolution-plus-mixin and accurate tanh in one loop | 22% slower at 32 frames, increasing to 66% slower at 256 frames |
| Residual addition in the 1x1 matrix store | 8.7%, 7.5%, 7.4%, and 6.5% faster at 32, 64, 128, and 256 frames |

Putting scalar `tanhf` calls inside the vector addition loop prevents that
loop from vectorizing. Fusing the residual into the matrix store was
profitable, but did not meet the 10% complete-callback retention gate.
Batching all ten layer heads and reducing them with explicit SIMD also failed
the gate in matched alternating runs: median improvements were 6.0% at 64
frames and 3.8% at 256 frames, with no statistically significant change.
Every prototype was removed.

Keeping each output tile in registers changes the rounding boundary between
convolution taps, while preserving tap order, depth order, and bias-last
evaluation. The measured complete-render differences are 7.97e-07 on ARM64
and 1.02e-06 on x86-64. Direct kernel oracles cover 8x8, 16x16, full, and
partial column tiles. Complete model tests cover render accuracy,
callback-partition invariance, reset behavior, and allocation freedom.

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

The packed A1 Conv1d backend covers the dominant 16x16 and 8x8 matrix shapes
on ARM64 and AVX2/FMA x86-64. Post-kernel profiles show that another retained
A1 optimization needs either an accurate vector-math activation backend or a
broader design that removes more than one layer pass without inhibiting SIMD.

## Real-world impact of the performance gap

At 48kHz with 64-sample buffers, the DAW gives 1.33ms per buffer.

| Model | Rust per buffer | Budget used |
|-------|----------------|-------------|
| Standard WaveNet | 0.27ms | 20% |
| a2_max | 0.17ms | 13% |

Both are well within real-time requirements. The 1.4x gap means C++ uses 14%
of the CPU budget while Rust uses 20%. The difference only matters when running
multiple simultaneous NAM instances on weak hardware.
