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

Processing 2 seconds of audio at 48kHz, buffer size 64 (matching C++ benchmodel):

| Model | C++ | Rust | |
|-------|-----|------|-|
| Small WaveNet | 29ms | 25ms | Rust faster |
| LSTM | 18ms | 20ms | Tied |
| Standard WaveNet (16/8ch) | 280ms | 384ms | 1.4x slower |
| a2_max | 146ms | 336ms | 2.3x slower |

C++ times are without fast_tanh (fair comparison). With fast_tanh enabled on both sides, the gap narrows further.

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

120+ tests including:
- Unit tests for all components
- C++ regression tests comparing output against reference WAV files
- Strict accuracy guards that prevent any degradation

## License

MIT
