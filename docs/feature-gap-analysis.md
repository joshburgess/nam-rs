# Feature Gap Analysis: nam-rs vs C++ NeuralAmpModelerCore

Last updated: 2026-06-26

## Feature Support Matrix

| Feature | Status | Notes |
|---------|--------|-------|
| **WaveNet (standard)** | Fully supported | All 4 official presets (Standard/Lite/Feather/Nano) |
| **WaveNet (gated)** | Fully supported | `gated: true` and `gating_mode: "gated"` |
| **WaveNet (blended)** | Fully supported | `gating_mode: "blended"` |
| **WaveNet (per-layer gating)** | Fully supported | Array of gating modes per layer |
| **WaveNet (FiLM)** | Fully supported | All 8 FiLM positions with shift and groups |
| **WaveNet (grouped convs)** | Fully supported | groups_input, groups_input_mixin, layer1x1 groups, head1x1 groups |
| **WaveNet (head1x1)** | Fully supported | Optional head output projection |
| **WaveNet (top-level head)** | Supported | Upstream A2 `config.head` with activation and kernel sizes |
| **WaveNet (bottleneck != channels)** | Fully supported | |
| **WaveNet (condition_dsp)** | Fully supported | Nested model as condition processor |
| **WaveNet A2 config aliases** | Supported | `layers_configs`, `head`, `head_1x1_config`, `layer_1x1_config`, `film_params` |
| **Sequential** | Supported | Runtime chaining for exported submodels with concatenated weights, including biased Linear submodels |
| **LSTM** | Fully supported | All 4 official presets |
| **Linear** | Fully supported | FIR filter with optional upstream `bias` export |
| **ConvNet** | Fully supported | With optional batch normalization |
| **Tanh, ReLU, Sigmoid, SiLU** | Supported | |
| **HardTanh, LeakyReLU** | Supported | |
| **Softsign, Softsigmoid, HardSwish** | Supported | |
| **LeakyHardTanh** | Supported | With configurable bounds and slopes |
| **PReLU** | Supported | Per-channel slopes |
| **Metadata** | Supported | Preserves raw metadata and exposes common upstream fields: loudness, gain, sample_rate, user gear fields, dBu levels, validation ESR |
| **Version 0.5-0.7** | Supported | Warns on versions beyond 0.7 |
| SlimmableWaveNet | Supported | Standalone channel-width selection is exposed through `Dsp::set_slimming` |
| SlimmableContainer | Supported | Loads embedded models, defaults to the highest-quality submodel, and supports `Dsp::set_slimming` selection |
| Packed A2 trainer option | Partial | Rust trainer exposes a Packed A2 option, sends explicit data-check and `ny` settings, upgrades installed NAM packages, filters worker kwargs against the installed upstream NAM signature, and includes an opt-in upstream full-config backend for `config_data`/`config_model`/`config_learning` packed training; upstream-generated packed export fixtures validate runtime export shape and render parity |
| FastTanh / fast sigmoid approximation | Supported | Global performance-mode toggle applies to WaveNet activations and LSTM gates/state tanh |

## Models That Load and Process Correctly

| Model File | Features Used | Status |
|------------|---------------|--------|
| `wavenet.nam` | Standard WaveNet | Matches C++ to 8.2e-08 |
| `wavenet_a1_standard.nam` | Standard (16/8ch, 20 layers) | Matches C++ to 2.1e-06 |
| `my_model.nam` | Standard WaveNet | Matches C++ to 2.1e-06 |
| `lstm.nam` | Standard LSTM | Matches C++ to 2.1e-07 |
| `wavenet_condition_dsp.nam` | Nested condition DSP | Matches C++ to 8.9e-08 |
| `wavenet_a2_max.nam` | All advanced features | Loads and runs; FP accumulation divergence (see optimization-plan.md) |
| `slimmable_wavenet.nam` | Slimmable architecture | Loads, processes, and supports runtime width selection |
| `slimmable_container.nam` | Multi-model container | Loads, processes, and supports runtime submodel selection |
| `a2_slimmable_container_upstream_style.nam` | Upstream-style packed A2 container | Loads from disk, defaults to the largest submodel, and supports runtime submodel selection |
| `upstream_packed_a2_export.nam` | Upstream `PackedWaveNet.export_container` output | Loads from disk, validates exact `SlimmableContainer` shape, and matches upstream reference render through the largest submodel |
| `upstream_full_packed_a2_trained.nam` | Upstream `nam.train.full.main` packed-training output | Loads from disk and validates trained packed export shape, including compensated `head_scale` weights |
| `sequential_linear.nam` | Sequential compositional model | Loads from disk, chains exported submodels, and covers biased Linear weight splitting |

## Remaining Gaps

### Slimmable Runtime Selection

Packed A2 training exports runtime `.nam` files as `SlimmableContainer` objects containing ordinary `WaveNet` submodels. This path is supported: all embedded submodels are loaded, the highest-quality submodel is used by default, and `Dsp::set_slimming` selects the active submodel by `max_value`.

Standalone `SlimmableWaveNet` channel slicing is also supported through `Dsp::set_slimming`. The standalone path follows upstream's restricted slimmable contract and rejects unsupported feature combinations at load time, including `condition_dsp`, FiLM, grouped convolutions, head1x1, multi-array configs, and non-1x1 layer-array heads.

### Block-Based Matrix Processing

See `optimization-plan.md`. The per-sample scalar processing produces mathematically equivalent but floating-point-divergent results from C++ Eigen block processing for the a2_max model. Converting to block-based processing would achieve bit-identical output and improve performance.
