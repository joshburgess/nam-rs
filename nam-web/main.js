// main.js — Main thread NAM Web Audio controller
// Uses ScriptProcessorNode for initial demo (runs WASM on main thread).
// Will migrate to AudioWorklet once the raw WASM instantiation is sorted out.

import init, { NamModel } from './pkg/nam_wasm.js';

let audioContext = null;
let processorNode = null;
let sourceNode = null;
let micStream = null;
let model = null;

const status = document.getElementById('status');
const modelInfo = document.getElementById('model-info');
const modelInput = document.getElementById('model-input');
const audioInput = document.getElementById('audio-input');
const micButton = document.getElementById('mic-toggle');

function setStatus(msg, isError = false) {
  status.textContent = msg;
  status.className = isError ? 'error' : '';
}

// Initialize WASM on page load
setStatus('Loading WASM...');
init().then(() => {
  setStatus('WASM loaded. Select a .nam model file.');
}).catch((err) => {
  setStatus(`WASM init failed: ${err.message}`, true);
});

function ensureAudioContext() {
  if (audioContext) return audioContext;
  audioContext = new AudioContext({ latencyHint: 'interactive' });
  return audioContext;
}

function ensureProcessor() {
  if (processorNode) return processorNode;

  const ctx = ensureAudioContext();
  const bufferSize = 512;

  // ScriptProcessorNode runs on the main thread — not ideal for production
  // but reliable for initial testing
  processorNode = ctx.createScriptProcessor(bufferSize, 1, 1);
  processorNode.onaudioprocess = (e) => {
    if (!model) return;

    const input = e.inputBuffer.getChannelData(0);
    const output = e.outputBuffer.getChannelData(0);
    model.process(input, output);
  };
  processorNode.connect(ctx.destination);

  return processorNode;
}

// Model file loading
modelInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  setStatus(`Loading model: ${file.name}...`);

  try {
    const bytes = await file.arrayBuffer();
    const data = new Uint8Array(bytes);

    // Free previous model
    if (model) {
      model.free();
      model = null;
    }

    model = new NamModel(data);

    const ctx = ensureAudioContext();
    const sr = model.sample_rate() || ctx.sampleRate;
    model.reset(sr, 512);
    model.prewarm();

    setStatus('Model ready. Load audio or enable mic input.');
    modelInfo.textContent = `Model sample rate: ${sr} Hz`;
    audioInput.disabled = false;
    micButton.disabled = false;
  } catch (err) {
    setStatus(`Model load failed: ${err.message}`, true);
  }
});

// Audio file playback
audioInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  ensureAudioContext();
  ensureProcessor();
  stopSource();

  setStatus('Decoding audio...');
  try {
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    sourceNode = audioContext.createBufferSource();
    sourceNode.buffer = audioBuffer;
    sourceNode.connect(processorNode);
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
  ensureAudioContext();
  ensureProcessor();

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
    sourceNode.connect(processorNode);
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
