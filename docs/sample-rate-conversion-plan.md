# Sample Rate Conversion Plan

## Problem

When a DAW runs at a different sample rate than the model was trained at (e.g. DAW at 44.1kHz, model trained at 48kHz), the plugin currently processes audio at the wrong rate. This causes pitch shifting and incorrect frequency response — the model's learned characteristics are tuned to a specific sample rate.

## How C++ NAM Handles It

The C++ NeuralAmpModelerCore itself does NOT resample. The iPlug2 plugin wrapper handles it at a higher level using iPlug2's built-in resampling. The model always processes audio at its trained sample rate.

## Architecture

```
Host audio (host_rate, N samples)
  │
  ▼
Input gain
  │
  ▼
Resample host_rate → model_rate (if needed)
  │  produces M samples (M ≈ N * model_rate / host_rate)
  ▼
NAM model.process() at model_rate
  │  produces M samples
  ▼
Resample model_rate → host_rate (if needed)
  │  produces N samples
  ▼
Output gain
  │
  ▼
Host output (host_rate, N samples)
```

## Challenges

### Variable buffer sizes

DAWs call `process()` with different buffer sizes each time (could be 32, 64, 128, 256, etc.). The resampler needs to handle arbitrary input lengths cleanly.

### Non-integer ratios

44100/48000 = 0.91875 — the resampler must handle non-integer ratios. Feeding N input samples produces a non-integer number of output samples. Over time, the resampler accumulates fractional samples that need to be buffered.

### Latency

Resampling introduces latency. The plugin should report this to the DAW via `set_latency_samples()` so the DAW can compensate.

### No allocations in process()

All buffers must be pre-allocated. The resampler state and intermediate buffers are set up in `initialize()` / `reset()`, never in `process()`.

## Implementation Plan

### Step 1: Choose resampler

Use `rubato` crate with `SincFixedIn` — a sinc interpolation resampler that accepts variable-length input and produces variable-length output. This matches the DAW's variable buffer size pattern.

Key rubato types:
- `SincFixedIn<f64>`: accepts fixed-size input chunks, produces variable output
- `SincFixedOut<f64>`: accepts variable input, produces fixed-size output chunks

For our case, `SincFixedOut` is wrong (we need variable output matching the host buffer). `SincFixedIn` is also wrong (DAW gives variable input).

Better: use `FastFixedIn<f64>` or `SincInterpolationType::Linear` with `SincFixedIn` where we pad/trim to fit the fixed chunk size. Or use rubato's `process_partial_into` which handles variable input lengths.

Actually, the simplest approach that avoids all chunk-size issues:

### Step 2: Ring buffer approach

Instead of fighting rubato's chunk requirements, use a simple ring-buffer approach:

1. **Input ring buffer** (host rate): accumulate incoming samples from `process()` calls
2. When enough samples are available, resample a chunk to model rate
3. **Model input buffer** (model rate): feed resampled audio to the NAM model
4. **Model output buffer** (model rate): collect model output
5. Resample model output back to host rate
6. **Output ring buffer** (host rate): feed back to the DAW

This decouples the DAW's variable buffer sizes from the resampler's preferred chunk size.

### Step 3: Data structures

```rust
struct ResamplerState {
    /// Host rate -> model rate resampler
    to_model: rubato::SincFixedIn<f64>,
    /// Model rate -> host rate resampler
    to_host: rubato::SincFixedIn<f64>,

    /// Input ring buffer: accumulates host-rate samples
    input_ring: Vec<f64>,
    input_ring_len: usize,

    /// Resampled model-rate input (ready to feed to model)
    model_in: Vec<nam_core::Sample>,
    model_in_len: usize,

    /// Model output at model rate
    model_out: Vec<nam_core::Sample>,

    /// Resampled host-rate output ring buffer
    output_ring: Vec<f64>,
    output_ring_read: usize,
    output_ring_write: usize,

    /// Resampler chunk size (fixed for rubato)
    chunk_size: usize,

    /// Ratio: model_rate / host_rate
    ratio: f64,

    /// Latency in host-rate samples (reported to DAW)
    latency: usize,
}
```

### Step 4: Process flow

```
fn process(buffer):
    N = buffer.samples()

    // Apply input gain, copy to input_ring
    for each sample in buffer:
        input_ring.push(sample * in_gain)

    // Resample chunks from input_ring to model_in
    while input_ring has >= chunk_size samples:
        resampled = to_model.process(input_ring[..chunk_size])
        input_ring.drain(..chunk_size)
        model_in.extend(resampled)

    // Process all available model_in through NAM
    if model_in_len > 0:
        model.process(model_in[..model_in_len], model_out[..model_in_len])

        // Resample model output back to host rate
        // Feed model_out chunks to to_host resampler
        while model_out has >= chunk_size samples:
            resampled = to_host.process(model_out[..chunk_size])
            output_ring.extend(resampled)
            model_out.drain(..chunk_size)

        model_in_len = 0

    // Copy from output_ring to buffer
    for i in 0..N:
        if output_ring has data:
            buffer[i] = output_ring.pop_front() * out_gain
        else:
            buffer[i] = 0.0  // underrun (shouldn't happen in steady state)
```

### Step 5: Initialization

In `initialize()`:
1. Compare `buffer_config.sample_rate` with model's `expected_sample_rate`
2. If they match (within 0.5 Hz), set `resampler = None` — direct processing
3. If they differ:
   - Create `SincFixedIn` resamplers for both directions
   - Pre-allocate all ring buffers with generous sizes (e.g. 4 * max_buffer_size)
   - Calculate and report latency via `context.set_latency_samples()`
4. In `reset()`: flush all ring buffers, reset resamplers

### Step 6: Latency reporting

The sinc resampler introduces latency equal to its filter length. rubato reports this via `input_delay()`. Total plugin latency = `to_model.input_delay() + to_host.input_delay()` converted to host-rate samples.

Report to DAW: `context.set_latency_samples(total_latency as u32)` in `initialize()`.

### Step 7: Edge cases

- **Model loaded after initialize()**: When a model is loaded via the GUI (background task), the resampler needs to be set up. The task_executor runs off the audio thread, so it can create resamplers. Signal the audio thread to swap in the new resampler state.
- **Buffer underrun**: If the output ring doesn't have enough samples (startup transient), output silence. This resolves after a few process() calls.
- **Model with no expected_sample_rate**: Some models don't specify a rate. Treat as matching host rate (no resampling).

### Step 8: Testing

- Unit test: verify resampling preserves signal characteristics (sine wave at known frequency, check output frequency matches)
- Integration test: load a 48kHz model, process at 44.1kHz, verify output is equivalent to processing at native rate
- Latency test: verify reported latency matches actual delay

## Estimated Complexity

Medium-high. The core resampling math is handled by rubato, but the ring buffer management, chunk alignment, latency reporting, and thread-safe resampler swapping add significant plumbing. Roughly 200-300 lines of new code in the plugin.

## Dependencies

- `rubato = "1.0"` (already evaluated, works with f64 samples)
