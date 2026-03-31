# WASM Compilation Audit — nam-core

## 1. Dependency Audit

### Runtime dependencies

| Crate | Links C? | OS-specific? | WASM compatible? |
|---|---|---|---|
| `matrixmultiply` 0.3 | No | No | Yes |
| `ndarray` 0.16 | No | No | Yes |
| `serde` 1.0 | No | No | Yes |
| `serde_json` 1.0 | No | No | Yes |
| `thiserror` 2.0 | No | No | Yes |
| `cc` (build-dep) | N/A | N/A | Only used when `fast-kernels` feature enabled |
| `faer` 0.20 (optional) | No | No | Untested — likely works but not needed for WASM |

**No blocking dependencies.** All runtime crates are pure Rust and compile cleanly to `wasm32-unknown-unknown`.

### Build dependency: `cc`

The `cc` crate is an unconditional build-dependency but only invokes the C compiler when `CARGO_FEATURE_FAST_KERNELS` is set (see `build.rs`). Since `fast-kernels` is not a default feature, the WASM build works without modification.

## 2. OS-Dependent API Usage

### `std::fs` (file I/O)

Used in `get_dsp.rs:12` — the `get_dsp(path)` public function reads a `.nam` file from disk.

**Not a blocker.** There is already a `get_dsp_from_json(json_str: &str)` function that accepts a JSON string directly. The WASM wrapper should use this instead — the caller (JS) will provide the file contents as bytes/string.

### `std::sync::atomic::AtomicBool`

Used in `util.rs` for the `USE_FAST_TANH` toggle. Atomics are supported in WASM (single-threaded mode uses non-atomic fallbacks). No issue.

### No usage of:
- `std::thread`
- `std::net`
- `std::time`
- `std::process`
- `std::arch::x86_64` or any platform-specific SIMD intrinsics

## 3. Test Compilation Result

```
$ RUSTFLAGS="" cargo build -p nam-core --target wasm32-unknown-unknown
Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.91s
```

**Clean build. No errors, no warnings** (beyond the `target-cpu=native` ignore, which is expected and harmless).

Full output saved to `wasm_build_audit.txt`.

## 4. API Surface for `wasm-bindgen` Wrappers

The WASM crate needs to expose:

| Method | Source | Notes |
|---|---|---|
| `load(data: &[u8])` | `get_dsp_from_json()` | Parse bytes as UTF-8 string, call `get_dsp_from_json` |
| `process(input, output)` | `dsp.process()` | Input/output are `&[f32]` / `&mut [f32]`. Use `float_io` feature so `Sample = f32` matches Web Audio |
| `sample_rate()` | `dsp.metadata().expected_sample_rate` | Return `Option<f32>` |
| `reset(sample_rate, buffer_size)` | `dsp.reset()` | Required before first `process()` call |
| `prewarm()` | `dsp.prewarm()` | Stabilizes model state |

### `Sample` type consideration

By default, `Sample = f64`. Web Audio uses `f32`. Two options:
1. **Use `float_io` feature** — sets `Sample = f32`, zero-cost match with Web Audio
2. **Convert at the boundary** — keep `f64` internally, cast in the wrapper

Option 1 is preferred — simpler and avoids per-sample conversion overhead in the audio thread.

## 5. Features to Exclude in WASM Builds

- `fast-kernels` — requires C compiler, not available for `wasm32-unknown-unknown`
- `faer` — untested on WASM, not needed (small matrix sizes don't benefit from it)

## 6. Estimated Complexity

**Green.** nam-core compiles to WASM with zero changes. The wrapper crate is straightforward — the entire public API surface is 5 methods, all operating on simple types (`&[f32]`, `&str`, `f32`).
