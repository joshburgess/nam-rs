# nam-rs

A Rust reimplementation of [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore) — the real-time DSP inference engine for [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler).

Loads `.nam` model files produced by the NAM Python trainer and processes audio identically to the C++ implementation.

## Features

- **All model architectures**: WaveNet, LSTM, ConvNet, Linear
- **All advanced WaveNet features**: FiLM conditioning, gated/blended activations, grouped convolutions, head1x1, nested condition_dsp, slimmable models
- **C++ accuracy parity**: bit-identical output on small models, within f32 precision floor on larger models (verified against C++ reference)
- **Fast tanh toggle**: switch between accurate `std::tanh` and C++-compatible polynomial approximation for performance
- **Audio plugin**: VST3/CLAP plugin with egui GUI, file browser, and gain controls
- **Training GUI**: native desktop app for training NAM models, with one-click Python/environment setup
- **CLI tools**: offline rendering and benchmarking

## Project Structure

```
nam-rs/
├── nam-core/       # DSP inference library
├── nam-cli/        # Command-line render and benchmark tools
├── nam-plugin/     # VST3/CLAP audio plugin (nih-plug + egui)
├── nam-trainer/    # GUI for training NAM models
└── xtask/          # Plugin bundling tool
```

## Building

```bash
# Build everything
cargo build --release

# Build the plugin (VST3 + CLAP)
cargo xtask bundle nam-plugin --release
```

The plugin bundles are output to `target/bundled/`.

### Recommended: enable native CPU optimizations

Add to `.cargo/config.toml` (already included in this repo):

```toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

## CLI Usage

### Render audio through a model

```bash
cargo run --release -p nam-cli -- render model.nam input.wav output.wav
```

### Benchmark a model

```bash
cargo run --release -p nam-cli -- bench model.nam 64 1500
cargo run --release -p nam-cli -- bench model.nam 64 1500 --fast  # with fast tanh
```

## NAM Trainer

The `nam-trainer` crate is a native desktop GUI for training Neural Amp Models. The upstream NAM project provides training via a [Google Colab notebook](https://github.com/sdatkinson/neural-amp-modeler) or a command-line Python script, but both require manual Python environment setup and offer no visual feedback during training. The trainer GUI solves this by providing a self-contained desktop experience.

```bash
cargo run --release -p nam-trainer
```

**Features:**
- **One-click environment setup** — auto-discovers Python installations, installs Miniforge and `neural-amp-modeler` with a single button click, detects NVIDIA/Apple GPU hardware
- **Training workflow** — select input/output audio files, configure model architecture (Standard/Lite/Feather/Nano), epochs, batch size, and advanced parameters, then train with a live loss curve plot and streaming log
- **Device selection** — automatically detects available training devices (CPU, CUDA GPUs, Apple MPS) and selects the best one
- **Model metadata** — set model name, gear type, tone type, and other metadata fields embedded in the `.nam` file
- **Cross-platform** — builds and runs on macOS, Linux, and Windows with platform-specific Python discovery and Miniforge installation

The trainer drives the upstream [neural-amp-modeler](https://github.com/sdatkinson/neural-amp-modeler) Python package via a JSON-based worker protocol over stdin/stdout. The Rust GUI handles all environment management and progress visualization while the Python process does the actual PyTorch training.

## Accuracy

Compared against C++ NeuralAmpModelerCore reference output (2 seconds of audio at 48kHz):

| Model | Max Diff vs C++ |
|-------|----------------|
| wavenet | bit-identical |
| wavenet_condition_dsp | bit-identical |
| lstm | 2.09e-07 |
| wavenet_a1_standard (16/8ch, 20 layers) | 1.13e-06 |
| wavenet_a2_max (all advanced features) | 4.77e-06 |

All differences are at the f32 precision floor.

## Performance

Processing 2 seconds of audio at 48kHz, buffer size 64, 1500 iterations.
Benchmarked on a MacBook Pro 16-inch (Nov 2024) with Apple M4 Max and 48 GB RAM, running macOS Sequoia 15.4.1.

All benchmarks below were measured with interleaved C++/Rust runs (alternating single runs, not batches) to equalize thermal conditions on the CPU.

### Default build (pure Rust, stable toolchain, no C compiler needed)

| Model | C++ (fast_tanh) | Rust (fast_tanh) | vs C++ |
|-------|----------------|-----------------|--------|
| Small WaveNet | 5ms | 7ms | 1.4x |
| LSTM | 6ms | 6ms | ~tied |
| Standard WaveNet (16/8ch, 20 layers) | 63ms | 167ms | 2.7x slower |
| a2_max (all advanced features) | 58ms | 132ms | 2.3x slower |

### With `fast-kernels` feature (requires C compiler)

```bash
cargo build --release --features fast-kernels
```

| Model | C++ (fast_tanh) | Rust (fast_tanh) | vs C++ |
|-------|----------------|-----------------|--------|
| Small WaveNet | 5ms | 7ms | 1.4x |
| LSTM | 6ms | 6ms | ~tied |
| Standard WaveNet (16/8ch, 20 layers) | 63ms | 74ms | 1.17x |
| a2_max (all advanced features) | 58ms | 67ms | 1.16x |

### With `fast-kernels` + `faer` features (best performance)

```bash
cargo build --release --features fast-kernels,faer
```

| Model | C++ (fast_tanh) | Rust (fast_tanh) | vs C++ |
|-------|----------------|-----------------|--------|
| Small WaveNet | 5ms | 7ms | 1.4x |
| LSTM | 6ms | 6ms | ~tied |
| Standard WaveNet (16/8ch, 20 layers) | 63ms | 66ms | **1.05x** |
| a2_max (all advanced features) | 58ms | 66ms | 1.14x |

The `faer` feature replaces the default `matrixmultiply` GEMM with [faer](https://github.com/sarah-ek/faer-rs)'s GEMM kernel, which is better tuned for the small matrix sizes (8x16) used in WaveNet's SGEMM path. It closes the Standard WaveNet gap from 17% to 5%. It has minimal effect on a2_max since that model's matrices are too small for the SGEMM path.

All models run well within real-time at any buffer size.

### Optimization journey

The pure Rust implementation started at 2.3-2.7x slower than C++ on large models. Through algorithmic fixes and C fast-math kernels, the gap was closed to 5-16%:

**Algorithmic fixes (pure Rust, no feature flags needed):**

- **Block condition_dsp processing.** The nested condition_dsp WaveNet was processed sample-by-sample (96,000 individual calls per benchmark), while C++ processes the entire buffer in one block. Switching to block processing was the single largest win for the a2_max model. *(a2_max: 132ms → 106ms)*
- **Bulk ring buffer rewind.** The ring buffer's rewind operation copied data element-by-element in a double loop. Replaced with a single `copy_within` call. *(~2% improvement across all models)*
- **Fused bias initialization.** Conv1d output was zeroed, accumulated via GEMM, then bias was added in a separate pass. Fused bias into the output initialization to eliminate the extra pass. *(~2% improvement)*
- **Eliminated redundant zeroing.** The condition input buffer was being zeroed per-frame for unused channels. The buffer is already pre-zeroed by allocation, so only the active channel needs to be written.
- **Hoisted fast_tanh flag.** The atomic `USE_FAST_TANH` flag was being loaded inside the per-element activation loop. Hoisted the check out of the loop so it's read once per buffer, not once per sample.

**C fast-math kernels (`fast-kernels` feature):**

The remaining gap after algorithmic fixes was caused by Rust's strict IEEE 754 float semantics preventing the compiler from reordering float operations, using FMA instructions freely, or vectorizing loops as aggressively as C with `-ffast-math`. The `fast-kernels` feature compiles a small C file (`csrc/fast_kernels.c`) with `-O3 -ffast-math` via the `cc` crate, providing optimized versions of:

- **Conv1d inner loops** — depthwise multiply-accumulate and small matrix-vector products, processed as one coarse-grained call per Conv1d block to amortize FFI overhead
- **Conv1x1 GEMM** — generic small matrix multiply with fused bias for all channel configurations
- **FiLM operations** — scale+shift and scale-only, both out-of-place and in-place variants
- **Fused add+activate** — combines `z = conv_output + mixin_output` and `z = tanh(z)` into a single pass, eliminating an intermediate buffer read/write
- **Gated/blended activations** — applies two different activation functions and combines them (multiply for gated, linear blend for blended) in one pass
- **Element-wise operations** — vector add, vector add-in-place, bias addition, all compiled with fast-math for better vectorization
- **Full SGEMM** — column-major matrix multiply for the large-matrix path

The C code is trivial — the same loops that exist in the Rust implementation, just compiled with different flags. The entire file is under 300 lines. The performance comes from the compiler flag, not the language.

**What we tried that didn't help:**

- **LLVM `-enable-unsafe-fp-math` flag** — Rust's frontend generates more conservative IR than C's frontend, so the backend flag alone doesn't replicate `-ffast-math`. No measurable improvement.
- **Apple Accelerate / BLAS** — dispatch overhead for BLAS routines exceeded the compute time for NAM's small matrices (8x16, 4x1). Pure Rust `matrixmultiply` was faster.
- **Nightly `core::intrinsics::fadd_fast`/`fmul_fast`** — per-operation fast-math intrinsics didn't help because LLVM doesn't propagate them to enable loop-level vectorization the way a global `-ffast-math` flag does.
- **Hand-written NEON SIMD kernels** — per-element SIMD kernels called per kernel tap had higher overhead than letting the compiler auto-vectorize the scalar loops. A hand-tuned 8x16 GEMM kernel matched `matrixmultiply` but didn't beat it.
- **Fused Conv1d loop (frame-outer)** — restructuring the Conv1d from tap-outer to frame-outer (to keep accumulators in registers across all taps) hurt cache performance because it accessed multiple tap data arrays per frame instead of streaming through one at a time.
- **Fused layer pipeline (mixin + add + activation in one pass)** — prevented the compiler from auto-vectorizing each simple pass independently, resulting in slower code than separate passes.

### Where the remaining 5-16% gap comes from

Nearly every float operation in the hot path now goes through C code compiled with `-ffast-math`, so compiler flags are no longer the primary difference. The remaining gap is architectural:

1. **Eigen expression templates eliminate intermediate buffers.** In C++, Eigen can write `z = conv_output + mixin_output` and the compiler sees it as one fused operation — no intermediate buffer is ever materialized. In our code, even with C kernels, we write the Conv1d output to one buffer, write the mixin output to another, then call a C function to combine them into a third. That's three separate buffers and two memory passes where Eigen does one. This applies to every step in the pipeline.

2. **Monolithic compilation vs. compositional architecture.** Our code has separate structs for Conv1d, Conv1x1, and FiLM, with method calls between them. Each method call is a function boundary the optimizer can't see across, especially across the Rust/C FFI boundary. Eigen compiles the entire layer pipeline as one function body where the optimizer sees everything and can keep intermediate values in registers across operations.

3. **FFI call overhead compounds.** Each C kernel call has a small fixed cost: save caller-saved registers, branch to the function, restore on return. For the a2_max model with its nested condition_dsp WaveNet and 8 FiLM modules per dilation, there are roughly 200+ C kernel calls per 64-sample buffer. At ~5-10ns each, that's 1-2 microseconds per buffer, compounding to several milliseconds over the full benchmark.

### What it would take to close the gap completely

The only way to eliminate the remaining overhead would be to compile the entire layer processing pipeline as a single C function — one call per layer per buffer instead of dozens of small kernel calls. But this would mean reimplementing the Rust layer processing logic (Conv1d + Conv1x1 mixin + FiLM + activation + gating + residual connection + head output) as monolithic C code, which defeats the purpose of having a Rust codebase.

Alternatively, a future Rust `#[fast_math]` function attribute (discussed in RFCs but not on the language roadmap) would allow the Rust compiler to apply the same float optimizations that `-ffast-math` enables in C, without needing FFI at all. This would close the gap entirely while keeping everything in pure Rust.

## Supported .nam File Features

| Feature | Status |
|---------|--------|
| WaveNet (standard/lite/feather/nano presets) | Supported |
| LSTM (all presets) | Supported |
| ConvNet | Supported |
| Linear | Supported |
| FiLM conditioning (all 8 positions) | Supported |
| Gated/blended activations | Supported |
| Grouped convolutions | Supported |
| head1x1 output projection | Supported |
| Nested condition_dsp | Supported |
| SlimmableWaveNet | Supported |
| SlimmableContainer | Supported |
| All activations (Tanh, ReLU, Sigmoid, SiLU, Softsign, HardSwish, PReLU, etc.) | Supported |
| Version 0.5-0.7 model files | Supported |

## Testing

```bash
cargo test -p nam-core
```

135+ tests including:
- Unit tests for all components
- C++ regression tests comparing output against reference WAV files
- Strict accuracy guards that prevent any degradation

## License

MIT
