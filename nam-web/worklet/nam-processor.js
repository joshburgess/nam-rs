// nam-processor.js — AudioWorklet processor for NAM inference
//
// Uses C-style FFI exports (nam_alloc, nam_load, nam_process, etc.)
// The wasm-bindgen imports are stubbed out since we only use the FFI path.

class NamProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._wasm = null;
    this._ready = false;
    this._inPtr = 0;
    this._outPtr = 0;
    this._bufLen = 0;

    this.port.onmessage = (e) => {
      const { type } = e.data;
      if (type === 'init-wasm') {
        this._initWasm(e.data.wasmBytes);
      } else if (type === 'load-model') {
        this._loadModel(e.data.modelBytes);
      }
    };
  }

  async _initWasm(wasmBytes) {
    try {
      // Stub all wasm-bindgen imports — we only use the C-style nam_* exports.
      const noop = () => {};
      const imports = {
        './nam_wasm_bg.js': {
          __wbindgen_object_drop_ref: noop,
          __wbg_new_227d7c05414eb861: () => 0,
          __wbg_stack_3b0d974bbf31e44f: noop,
          __wbg_error_a6fa202b58aa1cd3: noop,
          __wbg___wbindgen_throw_6ddd609b62940d55: (ptr, len) => {
            throw new Error('WASM panic');
          },
          __wbg___wbindgen_copy_to_typed_array_d2f20acdab8e0740: noop,
          __wbindgen_cast_0000000000000001: () => 0,
        },
      };

      const { instance } = await WebAssembly.instantiate(wasmBytes, imports);
      this._wasm = instance.exports;

      // Do NOT call __wbindgen_start — it sets up console_error_panic_hook
      // which requires console.error (unavailable in AudioWorklet scope).

      this.port.postMessage({ type: 'wasm-ready' });
    } catch (err) {
      this.port.postMessage({ type: 'error', message: `WASM init failed: ${err.message}` });
    }
  }

  _loadModel(modelBytes) {
    try {
      if (!this._wasm) {
        this.port.postMessage({ type: 'error', message: 'WASM not initialized' });
        return;
      }

      const wasm = this._wasm;
      const data = new Uint8Array(modelBytes);

      // Allocate and copy model JSON into WASM memory
      const jsonPtr = wasm.nam_alloc(data.length, 1);
      new Uint8Array(wasm.memory.buffer).set(data, jsonPtr);

      // Load model
      const result = wasm.nam_load(jsonPtr, data.length);

      // Free the JSON buffer
      wasm.nam_dealloc(jsonPtr, data.length, 1);

      if (result !== 0) {
        this.port.postMessage({ type: 'error', message: 'Model load failed in WASM' });
        return;
      }

      // Reset and prewarm
      const sr = wasm.nam_sample_rate() || sampleRate;
      wasm.nam_reset(sr, 128);
      wasm.nam_prewarm();

      // Pre-allocate audio buffers (128 samples = AudioWorklet render quantum)
      this._allocBuffers(128);

      this._ready = true;
      this.port.postMessage({ type: 'model-ready', sampleRate: sr });
    } catch (err) {
      this.port.postMessage({ type: 'error', message: `Model load failed: ${err.message}` });
    }
  }

  _allocBuffers(len) {
    const wasm = this._wasm;
    if (this._inPtr) wasm.nam_dealloc(this._inPtr, this._bufLen * 4, 4);
    if (this._outPtr) wasm.nam_dealloc(this._outPtr, this._bufLen * 4, 4);
    this._bufLen = len;
    this._inPtr = wasm.nam_alloc(len * 4, 4);
    this._outPtr = wasm.nam_alloc(len * 4, 4);
  }

  process(inputs, outputs) {
    if (!this._ready || !this._wasm) return true;

    const input = inputs[0]?.[0];
    const output = outputs[0]?.[0];
    if (!input || !output) return true;

    const wasm = this._wasm;
    const len = input.length;

    if (len !== this._bufLen) {
      this._allocBuffers(len);
    }

    // Copy input into WASM memory
    new Float32Array(wasm.memory.buffer, this._inPtr, len).set(input);

    // Process
    wasm.nam_process(this._inPtr, this._outPtr, len);

    // Copy output back
    output.set(new Float32Array(wasm.memory.buffer, this._outPtr, len));

    return true;
  }
}

registerProcessor('nam-processor', NamProcessor);
