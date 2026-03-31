// nam-processor.js — AudioWorklet processor for NAM inference
//
// Flow:
// 1. Main thread adds this module via audioWorklet.addModule()
// 2. importScripts loads the wasm-bindgen JS glue (defines wasm_bindgen global)
// 3. Main thread sends compiled WebAssembly.Module via postMessage
// 4. Worklet calls wasm_bindgen.initSync(module) to instantiate WASM synchronously
// 5. Main thread sends .nam model bytes
// 6. Worklet creates NamModel and starts processing

importScripts('../pkg-no-modules/nam_wasm.js');

class NamProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._model = null;
    this._ready = false;
    this._wasmReady = false;

    this.port.onmessage = (e) => {
      const { type } = e.data;
      if (type === 'init-wasm') {
        this._initWasm(e.data.module);
      } else if (type === 'load-model') {
        this._loadModel(e.data.modelBytes);
      }
    };
  }

  _initWasm(compiledModule) {
    try {
      wasm_bindgen.initSync({ module: compiledModule });
      this._wasmReady = true;
      this.port.postMessage({ type: 'wasm-ready' });
    } catch (err) {
      this.port.postMessage({ type: 'error', message: `WASM init failed: ${err.message}` });
    }
  }

  _loadModel(modelBytes) {
    try {
      if (!this._wasmReady) {
        this.port.postMessage({ type: 'error', message: 'WASM not initialized' });
        return;
      }

      // Free previous model
      if (this._model) {
        this._model.free();
        this._model = null;
        this._ready = false;
      }

      const model = new wasm_bindgen.NamModel(new Uint8Array(modelBytes));
      const modelSampleRate = model.sample_rate() || sampleRate;
      model.reset(modelSampleRate, 128);
      model.prewarm();

      this._model = model;
      this._ready = true;

      this.port.postMessage({
        type: 'model-ready',
        sampleRate: modelSampleRate,
      });
    } catch (err) {
      this.port.postMessage({ type: 'error', message: `Model load failed: ${err.message}` });
    }
  }

  process(inputs, outputs) {
    if (!this._ready || !this._model) return true;

    const input = inputs[0]?.[0];
    const output = outputs[0]?.[0];
    if (!input || !output) return true;

    this._model.process(input, output);
    return true;
  }
}

registerProcessor('nam-processor', NamProcessor);
