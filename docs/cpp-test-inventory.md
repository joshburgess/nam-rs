# C++ Upstream Test Inventory

Last updated: 2026-03-14

Source: `/upstream-core/tools/test/`

This document catalogs every test function in the C++ NeuralAmpModelerCore test suite for reference when expanding the Rust test suite.

## test_activations.cpp (24 tests)

### FastTanh
| Test | Behavior | Inputs | Assertions |
|------|----------|--------|-----------|
| `test_core_function()` | Fast tanh approximation | `0.0f` | `fast_tanh(0.0) == 0.0` |
| `test_get_by_init()` | Constructor + singleton | Create `ActivationLeakyReLU` | Applies to `[0.0]` -> `[0.0]` |
| `test_get_by_str()` | String-based lookup | "Fasttanh" | Process `[0.0]` -> `[0.0]` |

### LeakyReLU
| Test | Behavior | Inputs | Assertions |
|------|----------|--------|-----------|
| `test_core_function()` | Scalar function | `[0.0, 1.0, -1.0]` | `leaky_relu(0.0, 0.01) == 0.0`, `(1.0) == 1.0`, `(-1.0, 0.01) == -0.01` |
| `test_get_by_init()` | Constructor | slope=0.01 | Applies to `[0.0, 1.0, -1.0]` |
| `test_get_by_str()` | Singleton lookup | "LeakyReLU" | Process array |

### Softsign
| Test | Behavior | Inputs | Assertions |
|------|----------|--------|-----------|
| `test_core_function()` | `x / (1 + |x|)` | `[0.0, 1.0, -1.0, 2.0, -2.0]` | `softsign(0.0) == 0.0`, `(1.0) ~= 0.5`, `(2.0) ~= 0.667`, tol `1e-6` |
| `test_get_by_init()` / `test_get_by_str()` | Singleton access | Create/lookup | Apply to array |

### PReLU
| Test | Behavior | Inputs | Assertions |
|------|----------|--------|-----------|
| `test_core_function()` | Parametric ReLU | input=-1.0, slopes=[0.01, 0.05] | Expects -0.01, -0.05 |
| `test_per_channel_behavior()` | Per-channel slopes on matrix | 2x3 matrix, slopes=[0.01, 0.05] | Channel-wise application, tol `1e-6` |
| `test_wrong_number_of_channels_matrix()` | Channel mismatch validation | 3 channels, 2 slopes | Throws `std::invalid_argument` |
| `test_wrong_size_array()` | Array size validation | Size 5, 2 channels | Throws `std::invalid_argument` |
| `test_valid_array_size()` | Per-channel on flat array | Size 6, 2 channels, slopes=[0.1, 0.2] | Alternating application |

### ActivationConfig
| Test | Behavior | Inputs | Assertions |
|------|----------|--------|-----------|
| `test_simple_config()` | Config factory for ReLU | `ActivationConfig::simple(ReLU)` | Type matches |
| `test_all_simple_types()` | All 8 activation types | Tanh, Hardtanh, Fasttanh, ReLU, Sigmoid, SiLU, Hardswish, Softsign | All construct |
| `test_leaky_relu_config()` | Config with custom slope | slope=0.2 | Apply to `[-1, 0, 1]` -> `[-0.2, 0, 1]`, tol `1e-6` |
| `test_prelu_single_slope_config()` | PReLU from config | slope=0.25 | Constructs |
| `test_prelu_multi_slope_config()` | Per-channel slopes | 3 slopes on 3x2 matrix | Channel-wise, tol `1e-6` |
| `test_leaky_hardtanh_config()` | Hardtanh with slopes | min=-2, max=2, slopes=0.1 | Constructs |
| `test_softsign_config()` | Softsign config | No params | Apply to `[-1, 0, 1]` -> `[-0.5, 0, 0.5]`, tol `1e-6` |
| `test_from_json_string()` | Parse JSON string | `"ReLU"` | Type == ReLU |
| `test_from_json_object()` | Parse JSON object | `{"type": "LeakyReLU", "negative_slope": 0.15}` | Correct type and slope |
| `test_from_json_prelu_multi()` | Parse PReLU from JSON | `{"type": "PReLU", "negative_slopes": [0.1, 0.2, 0.3, 0.4]}` | 4 slopes parsed |
| `test_from_json_softsign_string()` | JSON softsign | `"Softsign"` | Type == Softsign |
| `test_from_json_softsign_object()` | JSON softsign with alpha | `{"type": "Softsign", "alpha": 0.5}` | Type == Softsign (alpha ignored) |
| `test_unknown_activation_throws()` | Invalid name | `"UnknownActivation"` | Throws `std::runtime_error` |

## test_conv1d.cpp (19 tests)

| Test | Behavior | Inputs | Assertions |
|------|----------|--------|-----------|
| `test_construct()` | Default constructor | None | `dilation == 1`, `in_channels == 0`, etc. |
| `test_construct_with_shape()` | Constructor with params | in=2, out=3, kernel=5, bias=true, dilation=7 | All getters match |
| `test_set_size()` | `set_size_()` | in=2, out=4, kernel=3, bias=true, dilation=2 | All getters match |
| `test_reset()` | Buffer allocation | 64 samples | Output shape correct |
| `test_process_basic()` | Basic 1D convolution | kernel=[1, 2], input=[1, 2, 3, 4] | output=[2, 5, 8, 11], tol 0.01 |
| `test_process_with_bias()` | Convolution + bias | kernel=[1, 0], bias=5, input=[2, 3] | output=[5, 7], tol 0.01 |
| `test_process_multichannel()` | Multi-channel (2 in, 3 out) | Specific kernel, input=[[1,3],[2,4]] | Expected output per channel, tol 0.01 |
| `test_process_dilation()` | Dilated convolution | kernel=[1, 2], dilation=2, input=[1,2,3,4] | output=[2, 4, 7, 10], tol 0.01 |
| `test_process_multiple_calls()` | Ring buffer persistence | kernel=[1, 1], 3 calls with input=[1, 2] | History maintained |
| `test_get_output_different_sizes()` | Variable output sizes | Identity kernel, process 4 then read 2 | Both work |
| `test_set_size_and_weights_()` | Combined init | kernel=[1, 2], iterator consumed | Iterator at `.end()` |
| `test_get_num_weights()` | Weight count | in=2, out=3, kernel=2, bias=true | Expected = 15 |
| `test_reset_multiple()` | Repeated resets | 64 then 128 | Both work |
| `test_process_grouped_basic()` | Grouped conv (2 groups) | in=4, out=4, groups=2 | Group isolation verified |
| `test_process_grouped_with_bias()` | Grouped + bias | 4 channels, 2 groups | output = input + bias |
| `test_process_grouped_multiple_groups()` | 4 groups | 8 channels, different scales | Each group independent |
| `test_process_grouped_kernel_size()` | Grouped with kernel>1 | kernel=2, dilation=1, 2 groups | Multi-tap history |
| `test_process_grouped_dilation()` | Grouped + dilated | kernel=2, dilation=2, 2 groups | Spacing applied per group |
| `test_process_grouped_channel_isolation()` | Group independence | 3 groups, different inputs | No cross-contamination |
| `test_get_num_weights_grouped()` | Weight count (grouped) | in=8, out=6, kernel=3, groups=2 | Expected = 78 |

## test_conv_1x1.cpp (14 tests)

| Test | Behavior | Inputs | Assertions |
|------|----------|--------|-----------|
| `test_construct()` | Constructor | in=2, out=3, bias=false | Getters match |
| `test_construct_with_groups()` | With groups | in=4, out=6, groups=2 | Getters match |
| `test_construct_validation_in_channels()` | Validation | in=5, out=6, groups=2 | Throws (not divisible) |
| `test_construct_validation_out_channels()` | Validation | in=4, out=5, groups=2 | Throws (not divisible) |
| `test_process_basic()` | Matrix multiply | weights=[1..6] (3x2), input=[[1,3],[2,4]] | Expected output, tol 0.01 |
| `test_process_with_bias()` | With bias | identity + bias=[10,20] | output=input+bias, tol 0.01 |
| `test_process_underscore()` | `process_()` stores internally | identity weights | `GetOutput()` matches |
| `test_process_grouped_basic()` | Grouped (2 groups) | in=4, out=4 | Group isolation |
| `test_process_grouped_with_bias()` | Grouped + bias | 4 channels, 2 groups | output = input + bias |
| `test_process_grouped_multiple_groups()` | 4 groups | 8 channels | Independent scaling |
| `test_process_grouped_channel_isolation()` | Isolation | 3 groups: zero, identity, identity | No cross-group leak |
| `test_process_underscore_grouped()` | `process_()` with groups | Grouped weights | Internal buffer correct |
| `test_set_max_buffer_size()` | Buffer allocation | 128 | Output shape matches |
| `test_process_multiple_calls()` | Multiple calls | identity weights | Successive calls work |

## test_convnet.cpp (7 tests)

| Test | Behavior | Inputs | Assertions |
|------|----------|--------|-----------|
| `test_convnet_basic()` | Basic ConvNet | 1 in/out, 2 channels, dilations=[1,2], ReLU | 4 frames, finite output |
| `test_convnet_batchnorm()` | With BatchNorm | 1 channel, batchnorm=true | Finite output |
| `test_convnet_multiple_blocks()` | Multiple blocks | dilations=[1,2,4], Tanh | 3 blocks, finite |
| `test_convnet_zero_input()` | Zero input | All zeros | No NaN/Inf |
| `test_convnet_different_buffer_sizes()` | Variable sizes | Reset 64 then 128 | Both work |
| `test_convnet_prewarm()` | Prewarm + process | Multiple blocks | Works after prewarm |
| `test_convnet_multiple_calls()` | State persistence | 5 calls, 2 frames each | All finite |

## test_dsp.cpp (9 tests)

| Test | Behavior | Inputs | Assertions |
|------|----------|--------|-----------|
| `test_construct()` | Constructor | in=1, out=1, sr=48000 | Constructs |
| `test_channels()` | Channel getters | in=2, out=3 | Match |
| `test_get_input_level()` | Input level | Set to 19.0 dB | Returns 19.0 |
| `test_get_output_level()` | Output level | Set to 12.0 dB | Returns 12.0 |
| `test_has_input_level()` | Presence check | Initially false, set, then true | State correct |
| `test_has_output_level()` | Presence check | Initially false, set, then true | State correct |
| `test_set_input_level()` | Setter | 19.0 | No errors |
| `test_set_output_level()` | Setter | 19.0 | No errors |
| `test_process_multi_channel()` | Multi-channel | 2 in, 2 out, 64 frames | Copies in->out |

## test_get_dsp.cpp (8 tests)

| Test | Behavior | Inputs | Assertions |
|------|----------|--------|-----------|
| `test_gets_input_level()` | Load LSTM, check level | Real LSTM JSON | `HasInputLevel()` == true |
| `test_gets_output_level()` | Load LSTM, check level | Real LSTM JSON | `HasOutputLevel()` == true |
| `test_null_input_level()` | Null field handling | `"input_level_dbu": null` | `HasInputLevel()` == false |
| `test_null_output_level()` | Null field handling | `"output_level_dbu": null` | `HasOutputLevel()` == false |
| `test_load_and_process_nam_files()` | Load all fixture .nam files | wavenet, lstm, condition_dsp, a2_max | 3 buffers x 64 frames, all finite |
| `test_version_patch_one_beyond_supported()` | Version compat | Patch beyond supported | Loads with warning |
| `test_version_minor_one_beyond_supported()` | Version rejection | Minor beyond supported | Throws |
| `test_version_too_early()` | Version rejection | Earlier than earliest | Throws |

## test_lstm.cpp (11 tests)

| Test | Behavior | Inputs | Assertions |
|------|----------|--------|-----------|
| `test_lstm_basic()` | Basic LSTM | 1 layer, hidden=4 | 4 frames, finite |
| `test_lstm_multiple_layers()` | Stacked | 2 layers, hidden=4 | 8 frames, finite |
| `test_lstm_zero_input()` | Zero input | All zeros | Finite |
| `test_lstm_different_buffer_sizes()` | Variable sizes | 64 then 128 | Both work |
| `test_lstm_prewarm()` | Prewarm | Reset + prewarm() | Process works after |
| `test_lstm_multiple_calls()` | State persistence | 5 calls, 2 frames each | State maintained |
| `test_lstm_multichannel()` | Multi-channel I/O | in=2, out=2, hidden=4 | Both channels finite |
| `test_lstm_large_hidden_size()` | Scaling | hidden=16 | Finite |
| `test_lstm_different_input_size()` | Variable input | in=3 | Works |
| `test_lstm_state_evolution()` | State change | Sine wave, 10 frames | Shows variation |
| `test_lstm_no_layers()` | Edge case | 0 LSTM layers (head only) | Finite |

## test_ring_buffer.cpp (10 tests)

| Test | Behavior | Inputs | Assertions |
|------|----------|--------|-----------|
| `test_construct()` | Default | None | Size=0, channels=0 |
| `test_reset()` | Allocation | channels=2, max=64 | Getters match |
| `test_reset_with_receptive_field()` | Zeroes history | lookback=10 | `Read(10, 10)` is zero |
| `test_write()` | Write data | [1..8], advance | Read matches written |
| `test_read_with_lookback()` | History offset | Write [1,2,3], write [4,5] | Can read past data |
| `test_advance()` | Advance pointer | Write, advance 10, read | History accessible |
| `test_rewind()` | Overflow rewind | 25 writes (>storage) | No crash, history preserved |
| `test_multiple_writes_reads()` | Alternating | Two writes | Lookback reads correct |
| `test_reset_zeros_history_area()` | History zeroing | Fill with 42.0, reset | Behind start is zero |
| `test_rewind_preserves_history()` | Rewind preserves | 3 writes of 32 | History matches last write |

## test_gating_activations.cpp (7 tests)

| Test | Behavior | Inputs | Assertions |
|------|----------|--------|-----------|
| `GatingActivation::test_basic_functionality()` | Identity + sigmoid gating | 2x3 input | Output 1x3, element-wise multiply |
| `GatingActivation::test_with_custom_activations()` | LeakyReLU gates | Custom config | Output shape correct |
| `BlendingActivation::test_basic_functionality()` | Identity blending | 2x3 input | Output 1x3 |
| `BlendingActivation::test_blending_behavior()` | Linear blending | Linear + sigmoid | alpha in [0,1] |
| `BlendingActivation::test_with_custom_activations()` | LeakyReLU + LeakyReLU | Custom | Shape correct |
| `BlendingActivation::test_error_handling()` | Error conditions | 1 row input | Assertion in debug |
| `BlendingActivation::test_edge_cases()` | Extremes | Zero, 1000, -1000 | Finite, no crashes |

## test_input_buffer_verification.cpp (2 tests)

| Test | Behavior | Inputs | Assertions |
|------|----------|--------|-----------|
| `test_buffer_stores_pre_activation_values()` | Blending pre-act storage | ReLU, input=-2.0, blend=0.5 | output = -1.0, tol `1e-6` |
| `test_buffer_with_different_activations()` | LeakyReLU buffer | LeakyReLU(0.1), input=-1.0, blend=0.8 | output = -0.28, tol `1e-6` |

## test_wavenet/ (6 files)

### test_layer.cpp (3 tests)

| Test | Behavior | Inputs | Assertions |
|------|----------|--------|-----------|
| `test_gated()` | Gated layer | 1ch, kernel=1, ReLU+sigmoid | Per-frame gated output |
| `test_layer_getters()` | Property getters | channels=4, kernel=3, dilation=2 | All match |
| `test_non_gated_layer()` | Non-gated with known weights | identity-like, input=1, cond=1 | Exact output traced through |

### test_layer_array.cpp (4 tests)

| Test | Behavior | Inputs | Assertions |
|------|----------|--------|-----------|
| `test_layer_array_basic()` | 2 layers [1,2] | 1 in/out, channels=1, ReLU | Correct shape, finite |
| `test_layer_array_receptive_field()` | RF calculation | kernel=3, dilations=[1,2,4] | RF = 14 |
| `test_layer_array_with_head_input()` | Head from prev array | Process with head_inputs | Works |
| `test_layer_array_different_activations()` | Per-layer activations | Tanh, ReLU | Different per layer |

### test_full.cpp (2 tests)

| Test | Behavior | Inputs | Assertions |
|------|----------|--------|-----------|
| `test_wavenet_model()` | Single array | 1ch, dilations=[1], ReLU | 4 frames, finite |
| `test_wavenet_multiple_arrays()` | 2 arrays chained | Array 0->1 | State propagates |

### test_head1x1.cpp (2 tests)

| Test | Behavior | Inputs | Assertions |
|------|----------|--------|-----------|
| `test_head1x1_inactive()` | head1x1=false | 2ch, identity | Head output = activated z |
| `test_head1x1_active()` | head1x1=true | Same + head1x1 weights | Transformation applied |

### test_condition_processing.cpp (1 test)

| Test | Behavior | Inputs | Assertions |
|------|----------|--------|-----------|
| `test_with_condition_dsp()` | Nested WaveNets | Parent with child condition DSP | Output uses conditioned input |

### test_real_time_safe.cpp (3 tests)

| Test | Behavior | Inputs | Assertions |
|------|----------|--------|-----------|
| `test_allocation_tracking_pass()` | Pre-allocated matmul | 10x20 matrices | Zero allocations |
| `test_allocation_tracking_fail()` | New allocation (negative test) | Create 10x20 matrix | Allocations detected |
| `test_conv1d_process_realtime_safe()` | Conv1D zero-alloc | 7 buffer sizes: 1..256 | All process() zero-alloc |

### test_factory.cpp (1 test)

| Test | Behavior | Inputs | Assertions |
|------|----------|--------|-----------|
| `test_factory_without_head_key()` | Load from JSON (no "head") | Minimal config | Loads, processes 4 frames finite |

## Testing Patterns Used in C++

1. **Synthetic weights**: Hand-crafted weights (identity, scaling, zeros) for deterministic output verification
2. **Tolerance**: Most float assertions use `0.01` or `1e-6`
3. **Ring buffer history**: Conv and WaveNet tests verify state persistence across multiple `process()` calls
4. **Grouped conv isolation**: Tests that groups don't leak into each other
5. **JSON parsing**: `get_dsp()` and factory tests load from JSON strings
6. **Real-time safety**: Custom allocation tracker verifies zero heap ops in hot paths
7. **Metadata handling**: Input/output levels tested as optional (can be null)
8. **Edge cases**: Zero input, large values, variable buffer sizes
9. **Prewarm**: Models pre-process silence to initialize state
