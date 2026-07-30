# Performance Analysis: Rust vs C++ NAM

Last updated: 2026-07-30

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
| 32 samples | 58.2 us | 31.7 us | 24.2 us | 19.5 us |
| 64 samples | 115.4 us | 62.4 us | 47.9 us | 38.0 us |
| 128 samples | 229.6 us | 119.1 us | 95.8 us | 75.6 us |
| 256 samples | 456.8 us | 245.7 us | 189.5 us | 151.2 us |

The default and A2 columns come from the preceding maintained run. The vForce
change affects only `fast-kernels` A1. It preserves the bit-identical A2 render
on Apple Silicon.

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
| Accelerate vForce accurate tanh on Apple platforms | 31% to 33% faster complete A1 callbacks | No measurable regression |
| glibc libmvec accurate tanh on x86-64 Linux | 22% to 28% faster complete A1 callbacks | No measurable regression |

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
| fast_tanh polynomial before packed kernels | Only 11% speedup | GEMM still dominated that revision. |
| Batched vForce LSTM gates and state tanh | 48% slower complete LSTM callback | Library-call overhead dominates at hidden size 3. |

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

### Accurate Apple activation backend

After the packed convolution kernels moved activation to roughly 40% of A1
callback samples, the accurate fused add-and-tanh kernel was changed to use
Accelerate vForce on Apple platforms. The kernel adds into fixed 256-element
stack chunks and applies `vvtanhf` without allocating.

Matched Criterion runs measured the complete A1 callback:

| Frames | Scalar accurate tanh | vForce accurate tanh | Improvement |
|-------:|---------------------:|---------------------:|------------:|
| 32 | 36.37 us | 24.23 us | 33.2% |
| 64 | 72.02 us | 47.87 us | 33.1% |
| 128 | 139.39 us | 95.79 us | 31.1% |
| 256 | 277.48 us | 189.45 us | 31.6% |

The direct kernel test covers lengths on both sides of the 256-element chunk
boundary. Complete A1 render, callback-partition, reset, and plugin allocation
tests pass. The maximum render difference remains 7.97e-07 on ARM64.

A post-change Apple M4 profile attributed 18.5% of complete callback leaf
samples to vForce and 8.1% to its platform-vector helpers. Scalar libm fell to
0.02%, and the fused kernel's own addition and dispatch code accounted for
2.1%. The remaining activation cost is inside the retained vector math
implementation.

The same profiling pass found no activation target in A2: activation kernels
accounted for 3.3% of its complete callback, while Conv1d and Conv1x1 accounted
for 55.1%. LSTM scalar math accounted for 32.3%, but batching its four gate
vectors and cell-state tanh through vForce made the complete callback 48.2%
slower. That prototype was removed.

### A2 convolution kernels

Native Linux x86-64 Callgrind profiles separated A2 convolution self
instructions from their wrappers and matrix callees:

| Frames | Conv1d | Conv1x1 | Combined |
|-------:|-------:|--------:|---------:|
| 32 | 29.69% | 51.48% | 81.17% |
| 64 | 29.96% | 51.58% | 81.54% |
| 128 | 30.07% | 51.57% | 81.64% |
| 256 | 30.12% | 51.56% | 81.68% |

The largest target was Conv1x1. Current Core commit `3cde95c` retains Eigen
for broad A2 block convolution and adds explicit small convolution paths in
[`dd972d6`](https://github.com/sdatkinson/NeuralAmpModelerCore/commit/dd972d6a45574aa4abff3a22487aebe7998aa5c7).
Its earlier portable A2Fast change
([`baf1bf8`](https://github.com/sdatkinson/NeuralAmpModelerCore/commit/baf1bf8e6a83691804ed23bbc176483d3f20b661))
was reverted in
[`b5a68c3`](https://github.com/sdatkinson/NeuralAmpModelerCore/commit/b5a68c3ebed5035a91d9207219346c81e8e3ce8e)
after AppleClang testing found target-dependent regressions.

The retained nam-rs path specializes 1, 4, and 8-input Conv1x1 products and
fuses the common rank-1 and 8-channel FiLM projections with their scale and
shift application. This removes the intermediate FiLM parameter-buffer pass
without changing the accumulation order. Matched Apple M4 Criterion runs
measured complete A2 callbacks:

| Frames | Before | After | Improvement |
|-------:|-------:|------:|------------:|
| 32 | 18.528 us | 16.172 us | 11.26% |
| 64 | 36.269 us | 32.324 us | 11.11% |
| 128 | 71.333 us | 62.661 us | 12.36% |
| 256 | 141.82 us | 123.91 us | 13.43% |

The conservative confidence bound exceeded 10% at every maintained size.
Direct tests cover strided and in-place FiLM variants plus every specialized
Conv1x1 input width. Complete model tests preserve callback partition and
reset equivalence, report zero plugin-callback allocations, and keep the
maximum A2 render difference at 5.72e-06 against the pinned Core fixture.

The follow-up profile attributed 40.20% of the 64-frame callback and 40.46%
of the 256-frame callback to Conv1d. Source-line attribution placed about
89.5% of the kernel's instructions in the generic matrix inner loop.
Shape-specific benchmarks cover the upstream 8x4/k4 path and the maintained
A2 fixture's 4x4/k4, 4x4/k3, and grouped 12x3/k2 paths at 16, 32, 64, 128,
and 256 frames.

Explicit 8x4 and 4x4 kernels retain the input-channel, tap, and bias
accumulation order of the generic implementation. The 8x4 kernel reduced
shape-level time by 77.6% to 78.2%. The 4x4/k3 path improved by 79.4% to
81.3%.

The upstream fused 4x4/k3 design was also evaluated. It improved the retained
per-tap kernel by 6.8% to 12.3% at 16 through 64 frames, but regressed it by
20.9% at 128 frames and 45.0% at 256 frames. The fused prototype was removed.
The per-tap specialization improved the complete Apple M4 A2 callback:

| Frames | Before | After | Improvement |
|-------:|-------:|------:|------------:|
| 16 | 8.773 us | 7.296 us | 16.1% |
| 32 | 16.734 us | 13.833 us | 17.1% |
| 64 | 32.807 us | 26.991 us | 17.9% |
| 128 | 64.210 us | 53.085 us | 17.4% |
| 256 | 128.06 us | 105.62 us | 17.2% |

The complete callback clears the 10% retention gate at every measured size.
Direct scalar oracles cover all benchmarked shapes and short-buffer
rejection. Complete model tests preserve render accuracy, streaming and reset
behavior, and allocation freedom.

The next profile found that the grouped 12x3/k2 condition convolution still
expanded its weights into a dense block-diagonal matrix. Its generic inner
loop accounted for 14.9% to 16.2% of the complete callback from 16 through
256 frames.

The loader now retains tap-major compact weights for grouped convolutions.
The common 12x3/k2 path evaluates three independent 4x1 groups, avoiding the
zero blocks, output-clear pass, and separate bias pass. A corrected
microbenchmark uses the fixture's grouped sparsity and weight order. The
compact kernel reduced its time by 97.4% to 97.6%.

Matched Apple M4 Criterion runs measured the complete A2 callback:

| Frames | Dense grouped path | Compact grouped path | Improvement |
|-------:|-------------------:|---------------------:|------------:|
| 16 | 7.184 us | 6.343 us | 11.7% |
| 32 | 14.074 us | 12.085 us | 14.1% |
| 64 | 27.106 us | 23.670 us | 12.7% |
| 128 | 53.812 us | 46.916 us | 12.8% |
| 256 | 106.58 us | 93.364 us | 12.4% |

The conservative confidence bound exceeds 10% at every size. Direct tests
cover compact loader order, scalar equivalence, and invalid buffer extents.
Complete render, streaming, reset, and plugin allocation tests also pass.

The following CI profile attributed 22.8% to 24.6% of the callback to
Conv1x1. Its generic inner loop accounted for 15.9% to 17.4%. Grouped
Conv1x1 weights were still expanded into dense block-diagonal matrices,
including the grouped FiLM projections.

The loader now retains the upstream group-major weights. Grouped products
route through fixed 1-, 2-, and 4-input-per-group kernels before the dense
or matrix backends. Direct four-output kernels preserve vectorization for
the 4x4/g2 and 4x8/g4 layouts. The 16x8/g4 scale-and-shift projection uses
a compact fused FiLM kernel.

Criterion benchmarks exercise every grouped Conv1x1 layout in the maintained
A2 fixture through the production loader and dispatch:

| Output x input | Groups | Improvement across 16–256 frames |
|---------------:|-------:|---------------------------------:|
| 3x6 | 3 | 61.6% to 63.4% |
| 6x6 | 3 | 68.9% to 71.6% |
| 4x2 | 2 | 27.7% to 33.6% |
| 4x4 | 2 | 61.8% to 64.7% |
| 4x8 | 4 | 21.5% to 27.3% |
| 8x8 | 2 | 31.2% to 32.8% |
| 8x8 | 4 | 37.2% to 41.1% |
| 8x8 | 8 | 29.1% to 32.0% |
| 16x8 | 4 | 7.6% to 12.7% |

Matched Apple M4 runs measured the complete A2 callback:

| Frames | Dense grouped path | Compact grouped path | Improvement |
|-------:|-------------------:|---------------------:|------------:|
| 16 | 6.296 us | 5.417 us | 14.0% |
| 32 | 12.225 us | 10.392 us | 15.0% |
| 64 | 23.738 us | 20.068 us | 15.5% |
| 128 | 46.908 us | 39.569 us | 15.6% |
| 256 | 93.727 us | 80.142 us | 14.5% |

The conservative confidence bound exceeds 13.7% at every size. Scalar
oracles cover every grouped shape, optional bias, strided input, and the
out-of-place and in-place fused FiLM paths. Complete render, streaming,
reset, and plugin allocation tests pass.

### Accurate Linux activation backend

The GNU x86-64 backend resolves glibc's four-lane SSE2 `tanhf` entry point
during WaveNet construction. glibc 2.35 and newer use the vector function.
Older glibc versions, musl, and non-x86 Linux targets retain the scalar path.
This avoids a hard GLIBC 2.35 symbol dependency and keeps dynamic loading out
of the audio callback.

Matched Criterion runs in an Ubuntu 22.04 x86-64 container measured the
complete A1 callback:

| Frames | Scalar accurate tanh | libmvec accurate tanh | Improvement |
|-------:|---------------------:|----------------------:|------------:|
| 32 | 201.90 us | 158.45 us | 21.8% |
| 64 | 406.52 us | 289.97 us | 27.9% |
| 128 | 786.56 us | 575.52 us | 27.0% |
| 256 | 1.5693 ms | 1.1752 ms | 24.9% |

The x86-64 container ran under Rosetta, so CI's native Linux instruction
profile remains the architecture-native performance gate. The same Ubuntu
22.04 environment passed boundary, special-value, and randomized-equivalence
tests. A glibc 2.31 build resolved no vector entry point and selected the
scalar fallback.

## Key Insights

### 1. A1 bottlenecks shift after each retained kernel

The standard WaveNet's hot path is 16x16 and 8x8 matrix-vector multiplies with
multiple frames batched. These are too small for general-purpose BLAS to be
optimal, but too large for scalar code to be competitive. The current backend
uses reset-sized packing scratch and an 8x8 NEON kernel on AArch64. Other
architectures use allocation-free `nano-gemm` microkernels. Once those kernels
landed, accurate tanh became the dominant A1 target and warranted a separate
vector-math backend.

### 2. Accuracy and performance trade off through FP accumulation order

Any change to the GEMM inner loop order changes which floating-point rounding errors occur. For models with small weights or few layers (wavenet, wavenet_condition_dsp), the differences are zero or negligible. For the a2_max model (5+ layers, 8 FiLM modules per layer, large weights), even 1 ULP difference per operation compounds to ~6e-06 through the deep network. This is still inaudible (-106 dB) but measurable.

### 3. Compiler optimizations often hurt for this workload

PGO, LTO, and codegen-units=1 all made things worse. The workload is dominated by tight numerical loops where instruction cache locality matters more than cross-function optimization. The default Rust release settings (16 codegen units, no LTO) produce the best results because they keep the hot code compact.

### 4. Accurate tanh became the post-kernel A1 bottleneck

Before the packed kernels, the fast-tanh approximation improved the callback
by only 11%. After the matrix work was reduced, accurate activation accounted
for roughly 40% of Apple samples and 44% of Linux instructions. Accelerate
vForce reduced the complete Apple callback by 31% to 33% without switching to
the approximate activation mode.

### 5. Small models match or beat C++

For models with channels ≤ 8 (small WaveNet, LSTM), my scalar dot-product loops are competitive with or faster than C++. The gap only appears at channels ≥ 16 where Eigen's SIMD micro-kernels have a significant advantage.

## Remaining performance work

The Apple and GNU x86-64 accurate activation backends and the A2 Conv1x1
specializations clear the complete-callback retention gate. A2 activation is
too small a fraction of its callback, and the LSTM vForce prototype regressed.
Windows and non-glibc Linux retain scalar accurate tanh until a portable
vector-math implementation can clear the same accuracy and complete-callback
gates.

## Real-world impact of the performance gap

At 48kHz with 64-sample buffers, the DAW gives 1.33ms per buffer.

| Model | Rust per buffer | Budget used |
|-------|----------------|-------------|
| Standard WaveNet | 0.27ms | 20% |
| a2_max | 0.17ms | 13% |

Both are well within real-time requirements. The 1.4x gap means C++ uses 14%
of the CPU budget while Rust uses 20%. The difference only matters when running
multiple simultaneous NAM instances on weak hardware.
