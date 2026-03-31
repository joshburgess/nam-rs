// main.js — Main thread NAM Web Audio controller
// Prefers AudioWorklet (off main thread), falls back to ScriptProcessorNode.

import initWasm, { NamModel } from './pkg/nam_wasm.js';

let audioContext = null;
let workletNode = null;
let scriptNode = null;
let sourceNode = null;
let micStream = null;
let mainThreadModel = null; // only used for ScriptProcessorNode fallback
let useWorklet = false;
let wasmInitialized = false;
let compiledModule = null;

const status = document.getElementById('status');
const modelInfo = document.getElementById('model-info');
const modelInput = document.getElementById('model-input');
const audioInput = document.getElementById('audio-input');
const micButton = document.getElementById('mic-toggle');

function setStatus(msg, isError = false) {
  status.textContent = msg;
  status.className = isError ? 'error' : '';
}

// Initialize wasm-bindgen module on main thread (for fallback + model loading UI)
setStatus('Loading WASM...');
initWasm().then(() => {
  wasmInitialized = true;
  setStatus('WASM loaded. Select a .nam model file.');
}).catch((err) => {
  setStatus(`WASM init failed: ${err.message}`, true);
});

function getProcessorNode() {
  return useWorklet ? workletNode : scriptNode;
}

async function ensureAudioContext() {
  if (audioContext) return;
  audioContext = new AudioContext({ latencyHint: 'interactive' });

  // Try AudioWorklet first
  try {
    console.log('Adding worklet module...');
    await audioContext.audioWorklet.addModule('worklet/nam-processor.js');
    console.log('Worklet module added successfully');

    workletNode = new AudioWorkletNode(audioContext, 'nam-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    workletNode.connect(audioContext.destination);

    // Compile and send WASM module to worklet
    const wasmResponse = await fetch('pkg/nam_wasm_bg.wasm');
    const wasmBytes = await wasmResponse.arrayBuffer();
    compiledModule = await WebAssembly.compile(wasmBytes);

    // Wait for wasm-ready with a timeout
    const workletReady = await new Promise((resolve) => {
      const timeout = setTimeout(() => resolve(false), 3000);

      workletNode.port.onmessage = (e) => {
        console.log('Worklet message:', e.data);
        if (e.data.type === 'wasm-ready') {
          clearTimeout(timeout);
          resolve(true);
        } else if (e.data.type === 'error') {
          clearTimeout(timeout);
          console.warn('Worklet error:', e.data.message);
          resolve(false);
        } else if (e.data.type === 'model-ready') {
          modelInfo.textContent = `Model sample rate: ${e.data.sampleRate} Hz | AudioWorklet`;
          setStatus('Model ready. Load audio or enable mic input.');
          audioInput.disabled = false;
          micButton.disabled = false;
        }
      };

      workletNode.onprocessorerror = (err) => {
        console.error('Worklet processor error:', err);
        clearTimeout(timeout);
        resolve(false);
      };

      workletNode.port.postMessage({ type: 'ping' });
      workletNode.port.postMessage(
        { type: 'init-wasm', wasmBytes },
        [wasmBytes],
      );
    });

    if (workletReady) {
      useWorklet = true;
      console.log('Using AudioWorklet');
      return;
    }

    // Worklet failed — clean up
    workletNode.disconnect();
    workletNode = null;
  } catch (err) {
    console.warn('AudioWorklet failed:', err, err.message, err.stack);
  }

  // Fallback to ScriptProcessorNode
  console.log('Falling back to ScriptProcessorNode');
  useWorklet = false;
  scriptNode = audioContext.createScriptProcessor(512, 1, 1);
  scriptNode.onaudioprocess = (e) => {
    if (!mainThreadModel) return;
    const input = e.inputBuffer.getChannelData(0);
    const output = e.outputBuffer.getChannelData(0);
    mainThreadModel.process(input, output);
  };
  scriptNode.connect(audioContext.destination);
}

// Model file loading
modelInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  if (!wasmInitialized) {
    setStatus('WASM not ready yet, please wait...', true);
    return;
  }

  setStatus(`Loading model: ${file.name}...`);
  await ensureAudioContext();

  const bytes = await file.arrayBuffer();

  if (useWorklet) {
    // Send model to worklet
    workletNode.port.postMessage(
      { type: 'load-model', modelBytes: bytes },
      [bytes],
    );
  } else {
    // Load on main thread
    try {
      if (mainThreadModel) {
        mainThreadModel.free();
        mainThreadModel = null;
      }

      const data = new Uint8Array(bytes);
      mainThreadModel = new NamModel(data);

      const sr = mainThreadModel.sample_rate() || audioContext.sampleRate;
      mainThreadModel.reset(sr, 512);
      mainThreadModel.prewarm();

      setStatus('Model ready. Load audio or enable mic input.');
      modelInfo.textContent = `Model sample rate: ${sr} Hz | ScriptProcessor`;
      audioInput.disabled = false;
      micButton.disabled = false;
    } catch (err) {
      setStatus(`Model load failed: ${err.message}`, true);
    }
  }
});

// Audio file playback
audioInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  await ensureAudioContext();
  stopSource();

  setStatus('Decoding audio...');
  try {
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    sourceNode = audioContext.createBufferSource();
    sourceNode.buffer = audioBuffer;
    sourceNode.connect(getProcessorNode());
    sourceNode.onended = () => {
      setStatus('Playback finished.');
      sourceNode = null;
    };
    sourceNode.start();
    setStatus(`Playing: ${file.name}`);
  } catch (err) {
    setStatus(`Audio decode failed: ${err.message}`, true);
  }
});

// Mic input
micButton.addEventListener('click', async () => {
  await ensureAudioContext();

  if (micStream) {
    stopSource();
    micStream.getTracks().forEach((t) => t.stop());
    micStream = null;
    micButton.textContent = 'Enable Mic';
    micButton.classList.remove('active');
    setStatus('Mic disabled.');
    return;
  }

  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        latency: 0,
      },
    });

    stopSource();
    sourceNode = audioContext.createMediaStreamSource(micStream);
    sourceNode.connect(getProcessorNode());
    micButton.textContent = 'Disable Mic';
    micButton.classList.add('active');
    setStatus('Mic active — playing through model.');
  } catch (err) {
    setStatus(`Mic access denied: ${err.message}`, true);
  }
});

function stopSource() {
  if (sourceNode) {
    try { sourceNode.disconnect(); } catch (_) {}
    if (sourceNode.stop) {
      try { sourceNode.stop(); } catch (_) {}
    }
    sourceNode = null;
  }
}
