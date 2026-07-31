# Feature Gap Analysis: nam-rs vs C++ NeuralAmpModelerCore

Comparison date: 2026-07-31

## Comparison Basis

This audit compares nam-rs against:

- `neural-amp-modeler` A2 release
  [`v0.13.0`](https://github.com/sdatkinson/neural-amp-modeler/tree/f26112906de06ec6b796ad6d1982e29eed83144e)
- `neural-amp-modeler` main at
  [`7527d0224b6110a2336819ba37f3422f2c15db3c`](https://github.com/sdatkinson/neural-amp-modeler/tree/7527d0224b6110a2336819ba37f3422f2c15db3c)
- `NeuralAmpModelerCore` main at
  [`3cde95c354d5ba6da01316cad90b05cfc4855053`](https://github.com/sdatkinson/NeuralAmpModelerCore/tree/3cde95c354d5ba6da01316cad90b05cfc4855053)

The post-A2 Core changes covered by this audit are the LSTM real-time fix
(`c9ac48e`), A2 prewarming fix (`763a079`), slimmable breakpoint API
(`1108b60`), WaveNet head dilation (`4c0ee78`), and FFT Linear processing
(`b352966`). Core also added small convolution specializations (`dd972d6`).
The broader portable A2Fast optimization (`baf1bf8`) was reverted
(`b5a68c3`) after target-dependent regressions, so it is not part of current
upstream parity.

Training changes after v0.13.0 add packed breakpoint validation
([`3cafd2a`](https://github.com/sdatkinson/neural-amp-modeler/commit/3cafd2a81f5299a9c3aba373a91e70074ce4d891),
[`faccd89`](https://github.com/sdatkinson/neural-amp-modeler/commit/faccd895769d9ea608acdb92aaca58d44c9ec731)),
mean packed validation metrics
([`bb56b2e`](https://github.com/sdatkinson/neural-amp-modeler/commit/bb56b2e8c84d8dbf035922dedd7d14bef64228fd)),
centralized packed Lightning module resolution
([`4f42495`](https://github.com/sdatkinson/neural-amp-modeler/commit/4f4249572b69b06ad973ed038b9a59d1427e80bd)),
and safer interrupted-training export
([`7527d02`](https://github.com/sdatkinson/neural-amp-modeler/commit/7527d0224b6110a2336819ba37f3422f2c15db3c)).
These commits do not add an inference architecture or exported model field.

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
| **WaveNet (top-level head)** | Supported | Upstream A2 `config.head` with activation, kernel sizes, and `head_dilation` |
| **WaveNet (bottleneck != channels)** | Fully supported | |
| **WaveNet (condition_dsp)** | Fully supported | Nested model as condition processor |
| **WaveNet A2 config aliases** | Supported | `layers_configs`, `head`, `head_1x1_config`, `layer_1x1_config`, `film_params` |
| **Sequential** | Supported | Runtime chaining for exported submodels with concatenated weights, including biased Linear submodels and fixed-capacity real-time scratch storage |
| **LSTM** | Fully supported | All 4 official presets, including allocation-free multi-layer inference |
| **Linear** | Fully supported | FIR filter with optional upstream `bias` export and zero-latency partitioned FFT processing for long filters |
| **ConvNet** | Fully supported | With optional batch normalization |
| **Tanh, ReLU, Sigmoid, SiLU** | Supported | |
| **HardTanh, LeakyReLU** | Supported | |
| **Softsign, Softsigmoid, HardSwish** | Supported | |
| **LeakyHardTanh** | Supported | With configurable bounds and slopes |
| **PReLU** | Supported | Per-channel slopes |
| **Metadata** | Supported | Preserves raw metadata and exposes common upstream fields: loudness, gain, sample_rate, user gear fields, dBu levels, validation ESR |
| **Version 0.5-0.7** | Supported | Warns on versions beyond 0.7 |
| SlimmableWaveNet | Supported | `Dsp::set_slimming` selects channel width and `Dsp::slimming_breakpoints` exposes every valid transition |
| SlimmableContainer | Supported | Loads embedded models, defaults to the highest-quality submodel, validates full breakpoint coverage, and exposes runtime transitions |
| Packed A2 plugin control | Supported | Persisted model-size parameter applies supported breakpoints during loading, parameter changes, reset, and allocation-free audio processing |
| Packed A2 trainer option | Supported | Requires `neural-amp-modeler >= 0.13.0`, reports actionable dependency errors, supports the upstream full-config packed training path, and normalizes per-submodel metrics across release and main semantics |
| FastTanh / fast sigmoid approximation | Supported | Global performance-mode toggle applies to WaveNet activations and LSTM gates/state tanh |

## Models That Load and Process Correctly

| Model File | Features Used | Status |
|------------|---------------|--------|
| `wavenet.nam` | Standard WaveNet | Bit-identical to the pinned Core render |
| `wavenet_a1_standard.nam` | Standard (16/8ch, 20 layers) | Matches C++ to 5.96e-08 |
| `my_model.nam` | Standard WaveNet | Matches C++ to 5.96e-08 |
| `lstm.nam` | Standard LSTM | Matches C++ to 8.94e-08 |
| `wavenet_condition_dsp.nam` | Nested condition DSP | Bit-identical to the pinned Core render |
| `wavenet_a2_max.nam` | All advanced features | Bit-identical to the pinned Apple Silicon Core render with the default matrix backend |
| `slimmable_wavenet.nam` | Slimmable architecture | Loads, processes, and supports runtime width selection |
| `slimmable_container.nam` | Multi-model container | Loads, processes, and supports runtime submodel selection |
| `a2_slimmable_container_upstream_style.nam` | Upstream-style packed A2 container | Loads from disk, defaults to the largest submodel, and supports runtime submodel selection |
| `upstream_packed_a2_export.nam` | Upstream `PackedWaveNet.export_container` output | Loads from disk, validates exact `SlimmableContainer` shape, and matches upstream reference render through the largest submodel |
| `upstream_full_packed_a2_trained.nam` | Upstream `nam.train.full.main` packed-training output | Loads from disk and validates trained packed export shape, including compensated `head_scale` weights |
| `sequential_linear.nam` | Sequential compositional model | Loads from disk, chains exported submodels, and covers biased Linear weight splitting |
| `upstream_head_dilation.nam` | Post-A2 WaveNet layer-head dilation | Matches a render produced by Core commit `4c0ee78` within 2e-06 |

## Real-Time and Performance Parity

Complete plugin-callback allocation tests exercise real two-layer LSTM,
Sequential, standard A1 WaveNet, maximum A2 WaveNet, packed A2, and FFT Linear
models. WaveNet matrix scratch is sized during `reset()` and reused by every
convolution in the callback. Model construction, buffer-size changes, and
prewarming remain outside the audio callback.

At 128 samples, the old SGEMM backends produced these allocation counts per
complete callback:

| Build | A1 standard | A2 max |
|-------|------------:|-------:|
| Default | 82 | 14 |
| `fast-kernels` | 82 | 14 |
| `faer` | 77 | 0 |
| `faer,fast-kernels` | 77 | 0 |

All four builds now report zero allocations for both models. CI runs the same
callback tests for every backend feature combination.

The Linear implementation follows Core's zero-latency hybrid design: the first
partition is evaluated directly and longer tails use partitioned FFT
convolution. Automatic selection retains direct processing through 256 taps.
On an Apple Silicon development machine, Criterion measured these 64-sample
callback times:

| Filter taps | Direct | FFT |
|------------:|-------:|----:|
| 512 | 28.4 us | 13.6 us |
| 1,536 | 89.9 us | 13.5 us |
| 4,096 | 240.5 us | 28.7 us |
| 16,384 | 967.2 us | 59.3 us |

## Repeatable Upstream Audit

The fixture manifest at `scripts/upstream_compatibility.json` records the
comparison date, training release and main commits, Core commit, and upstream
render fixtures. Bit-exact fixtures reject any differing sample. Packed A2 and
head-dilation fixtures enforce their recorded maximum absolute error. With the
pinned Core checkout built, reproduce the audit with:

```sh
python3 scripts/audit_upstream_compatibility.py \
  --core /path/to/NeuralAmpModelerCore
```

Use `--update` only when intentionally regenerating fixtures from that exact
commit. The script rejects any other Core revision before rendering.

## Remaining Gap

No known post-A2 model-format or inference capability gap remains at the
comparison commits. Performance work remains ongoing, especially small-matrix
throughput on standard A1 models, but it does not block model compatibility.
