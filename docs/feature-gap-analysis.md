# Feature Gap Analysis: nam-rs vs C++ NeuralAmpModelerCore

Last updated: 2026-03-14

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
| **WaveNet (bottleneck != channels)** | Fully supported | |
| **WaveNet (condition_dsp)** | Fully supported | Nested model as condition processor |
| **LSTM** | Fully supported | All 4 official presets |
| **Linear** | Fully supported | FIR filter |
| **ConvNet** | Fully supported | With optional batch normalization |
| **Tanh, ReLU, Sigmoid, SiLU** | Supported | |
| **HardTanh, LeakyReLU** | Supported | |
| **Softsign, HardSwish** | Supported | |
| **LeakyHardTanh** | Supported | With configurable bounds and slopes |
| **PReLU** | Supported | Per-channel slopes |
| **Metadata** | Fully supported | loudness, sample_rate, dBu levels |
| **Version 0.5-0.7** | Supported | Warns on versions beyond 0.7 |
| SlimmableWaveNet | Not supported | Experimental model compression |
| SlimmableContainer | Not supported | Multi-model container |
| FastTanh (LUT) | Not supported | Performance optimization only |

## Models That Load and Process Correctly

| Model File | Features Used | Status |
|------------|---------------|--------|
| `wavenet.nam` | Standard WaveNet | Matches C++ to 8.2e-08 |
| `wavenet_a1_standard.nam` | Standard (16/8ch, 20 layers) | Matches C++ to 2.1e-06 |
| `my_model.nam` | Standard WaveNet | Matches C++ to 2.1e-06 |
| `lstm.nam` | Standard LSTM | Matches C++ to 2.1e-07 |
| `wavenet_condition_dsp.nam` | Nested condition DSP | Matches C++ to 8.9e-08 |
| `wavenet_a2_max.nam` | All advanced features | Loads and runs; FP accumulation divergence (see optimization-plan.md) |
| `slimmable_wavenet.nam` | Slimmable architecture | Does not load (unsupported) |
| `slimmable_container.nam` | Multi-model container | Does not load (unsupported) |

## Remaining Gaps

### SlimmableWaveNet / SlimmableContainer

These are experimental model compression architectures for on-device inference. They allow dynamically reducing channel counts at runtime. Not used by any official trainer preset. Implementation would require:
- A `SlimmableWaveNet` that wraps a regular WaveNet with channel-slicing logic
- A `SlimmableContainer` that holds multiple model variants and switches between them
- ~200+ lines of additional code

### Block-Based Matrix Processing

See `optimization-plan.md`. The per-sample scalar processing produces mathematically equivalent but floating-point-divergent results from C++ Eigen block processing for the a2_max model. Converting to block-based processing would achieve bit-identical output and improve performance.
