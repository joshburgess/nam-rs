// nam-processor.js — AudioWorklet processor for NAM inference
//
// Flow:
// 1. Main thread compiles WASM module and sends it via postMessage
// 2. Worklet instantiates WASM synchronously with minimal import stubs
// 3. Main thread sends .nam model bytes
// 4. Worklet creates NamModel and processes audio blocks

class NamProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._wasm = null;
    this._modelPtr = 0;
    this._ready = false;

    // wasm-bindgen heap object management
    this._heapObjects = [undefined, null, true, false];
    this._heapFreeSlots = [];

    this.port.onmessage = (e) => {
      const { type } = e.data;
      if (type === 'init-wasm') {
        this._initWasm(e.data.module);
      } else if (type === 'load-model') {
        this._loadModel(e.data.modelBytes);
      }
    };
  }

  // -- Heap object helpers (mirror wasm-bindgen's JS glue) --

  _addHeapObject(obj) {
    if (this._heapFreeSlots.length > 0) {
      const idx = this._heapFreeSlots.pop();
      this._heapObjects[idx] = obj;
      return idx;
    }
    this._heapObjects.push(obj);
    return this._heapObjects.length - 1;
  }

  _getObject(idx) {
    return this._heapObjects[idx];
  }

  _takeObject(idx) {
    const obj = this._heapObjects[idx];
    this._heapObjects[idx] = undefined;
    this._heapFreeSlots.push(idx);
    return obj;
  }

  _getStringFromWasm(ptr, len) {
    return new TextDecoder().decode(
      new Uint8Array(this._wasm.memory.buffer, ptr, len)
    );
  }

  _getDataView() {
    return new DataView(this._wasm.memory.buffer);
  }

  _initWasm(compiledModule) {
    try {
      const self = this;
      let wasm;
      let WASM_VECTOR_LEN = 0;

      function passStringToWasm(str, alloc, realloc) {
        const encoder = new TextEncoder();
        const encoded = encoder.encode(str);
        const ptr = alloc(encoded.length, 1) >>> 0;
        new Uint8Array(wasm.memory.buffer).set(encoded, ptr);
        WASM_VECTOR_LEN = encoded.length;
        return ptr;
      }

      const imports = {
        './nam_wasm_bg.js': {
          __wbindgen_object_drop_ref: (idx) => {
            self._takeObject(idx);
          },
          __wbg_new_227d7c05414eb861: () => {
            return self._addHeapObject(new Error());
          },
          __wbg_stack_3b0d974bbf31e44f: (retPtr, objIdx) => {
            const stack = self._getObject(objIdx).stack || '';
            const ptr = passStringToWasm(stack, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
            const dv = self._getDataView();
            dv.setInt32(retPtr + 4, WASM_VECTOR_LEN, true);
            dv.setInt32(retPtr + 0, ptr, true);
          },
          __wbg_error_a6fa202b58aa1cd3: (ptr, len) => {
            const msg = self._getStringFromWasm(ptr, len);
            console.error('WASM error:', msg);
          },
          __wbg___wbindgen_throw_6ddd609b62940d55: (ptr, len) => {
            throw new Error(self._getStringFromWasm(ptr, len));
          },
          __wbg___wbindgen_copy_to_typed_array_d2f20acdab8e0740: (ptr, len, objIdx) => {
            const target = self._getObject(objIdx);
            new Uint8Array(target.buffer, target.byteOffset, target.byteLength)
              .set(new Uint8Array(wasm.memory.buffer, ptr, len));
          },
          __wbindgen_cast_0000000000000001: (ptr, len) => {
            return self._addHeapObject(self._getStringFromWasm(ptr, len));
          },
        },
      };

      const instance = new WebAssembly.Instance(compiledModule, imports);
      wasm = instance.exports;
      this._wasm = wasm;

      // Call start function (sets panic hook)
      if (wasm.__wbindgen_start) {
        wasm.__wbindgen_start();
      }

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

      // Free previous model
      if (this._modelPtr) {
        wasm.__wbg_nammodel_free(this._modelPtr, 0);
        this._modelPtr = 0;
        this._ready = false;
      }

      const data = new Uint8Array(modelBytes);

      // Allocate and copy model bytes into WASM memory
      const dataPtr = wasm.__wbindgen_export2(data.length, 1) >>> 0;
      new Uint8Array(wasm.memory.buffer).set(data, dataPtr);

      // Call NamModel::load(retptr, data_ptr, data_len)
      const retPtr = wasm.__wbindgen_add_to_stack_pointer(-16);
      wasm.nammodel_load(retPtr, dataPtr, data.length);

      const dv = new DataView(wasm.memory.buffer);
      const modelHandle = dv.getInt32(retPtr + 0, true);
      const errFlag = dv.getInt32(retPtr + 8, true);
      wasm.__wbindgen_add_to_stack_pointer(16);

      if (errFlag) {
        const errPtr = dv.getInt32(retPtr + 0, true);
        const errLen = dv.getInt32(retPtr + 4, true);
        this.port.postMessage({ type: 'error', message: 'Model load failed in WASM' });
        return;
      }

      this._modelPtr = modelHandle;

      // Reset and prewarm
      wasm.nammodel_reset(this._modelPtr, sampleRate, 128);
      wasm.nammodel_prewarm(this._modelPtr);

      this._ready = true;
      this.port.postMessage({
        type: 'model-ready',
        sampleRate: sampleRate,
      });
    } catch (err) {
      this.port.postMessage({ type: 'error', message: `Model load failed: ${err.message}` });
    }
  }

  process(inputs, outputs) {
    if (!this._ready || !this._wasm) return true;

    const input = inputs[0]?.[0];
    const output = outputs[0]?.[0];
    if (!input || !output) return true;

    const wasm = this._wasm;
    const len = input.length;

    // Allocate buffers in WASM linear memory
    const inPtr = wasm.__wbindgen_export2(len * 4, 4) >>> 0;
    const outPtr = wasm.__wbindgen_export2(len * 4, 4) >>> 0;

    // Copy input
    new Float32Array(wasm.memory.buffer, inPtr, len).set(input);
    new Float32Array(wasm.memory.buffer, outPtr, len).fill(0);

    // Process
    const retPtr = wasm.__wbindgen_add_to_stack_pointer(-16);
    this._wasm.nammodel_process(this._modelPtr, inPtr, len, outPtr, len, retPtr);
    wasm.__wbindgen_add_to_stack_pointer(16);

    // Copy output
    output.set(new Float32Array(wasm.memory.buffer, outPtr, len));

    // Free temp buffers
    wasm.__wbindgen_export(inPtr, len * 4, 1);
    wasm.__wbindgen_export(outPtr, len * 4, 1);

    return true;
  }
}

registerProcessor('nam-processor', NamProcessor);
