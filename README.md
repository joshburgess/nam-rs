# nam-rs

A Rust reimplementation of [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore) — the real-time DSP inference engine for [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler).

Loads `.nam` model files produced by the NAM Python trainer and processes audio identically to the C++ implementation.

## Features

- **All model architectures**: WaveNet, LSTM, ConvNet, Linear
- **All advanced WaveNet features**: FiLM conditioning, gated/blended activations, grouped convolutions, head1x1, nested condition_dsp, slimmable models
- **C++ accuracy parity**: bit-identical output on small models, within f32 precision floor on larger models (verified against C++ reference)
- **Fast tanh toggle**: switch between accurate `std::tanh` and C++-compatible polynomial approximation for performance
- **Audio plugin**: VST3/CLAP plugin with egui GUI, file browser, and gain controls
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

### Default build (pure Rust, stable toolchain, no C compiler needed)

| Model | C++ (fast_tanh) | Rust (fast_tanh) | vs C++ |
|-------|----------------|-----------------|--------|
| Small WaveNet | 6ms | 6ms | ~tied |
| LSTM | 6ms | 5ms | Rust faster |
| Standard WaveNet (16/8ch, 20 layers) | 61ms | 167ms | 2.7x slower |
| a2_max (all advanced features) | 56ms | 132ms | 2.4x slower |

### With `fast-kernels` feature (requires C compiler)

```bash
cargo build --release --features fast-kernels
```

| Model | C++ (fast_tanh) | Rust (fast_tanh) | vs C++ |
|-------|----------------|-----------------|--------|
| Small WaveNet | 6ms | 6ms | ~tied |
| LSTM | 6ms | 5ms | Rust faster |
| Standard WaveNet (16/8ch, 20 layers) | 61ms | 71ms | 1.16x |
| a2_max (all advanced features) | 56ms | 63ms | 1.13x |

All models run well within real-time at any buffer size.

### Optimization journey

The pure Rust implementation started at 2.4-2.7x slower than C++ on large models. Through a series of optimizations, the `fast-kernels` feature closes the gap to ~15%:

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

The C code is trivial — the same loops that exist in the Rust implementation, just compiled with different flags. The entire file is ~250 lines. The "secret sauce" is the compiler flag, not the language.

### Where the remaining ~15% gap comes from

Nearly every float operation in the hot path now goes through C code compiled with `-ffast-math`, so compiler flags are no longer the difference. The remaining gap is architectural:

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
