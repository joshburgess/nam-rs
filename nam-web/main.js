// main.js — Main thread NAM Web Audio controller

let audioContext = null;
let workletNode = null;
let sourceNode = null;
let micStream = null;

const status = document.getElementById('status');
const modelInfo = document.getElementById('model-info');
const modelInput = document.getElementById('model-input');
const audioInput = document.getElementById('audio-input');
const micButton = document.getElementById('mic-toggle');

function setStatus(msg, isError = false) {
  status.textContent = msg;
  status.className = isError ? 'error' : '';
}

async function ensureAudioContext() {
  if (audioContext) return;

  audioContext = new AudioContext({ latencyHint: 'interactive' });

  // Add worklet module (this loads the JS glue via importScripts inside the worklet)
  await audioContext.audioWorklet.addModule('worklet/nam-processor.js');

  // Create worklet node
  workletNode = new AudioWorkletNode(audioContext, 'nam-processor', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
  });
  workletNode.connect(audioContext.destination);

  // Listen for messages from the worklet
  workletNode.port.onmessage = (e) => {
    const { type } = e.data;
    if (type === 'wasm-ready') {
      setStatus('WASM loaded. Select a .nam model file.');
    } else if (type === 'model-ready') {
      setStatus('Model ready. Load audio or enable mic input.');
      modelInfo.textContent = `Model sample rate: ${e.data.sampleRate} Hz`;
      audioInput.disabled = false;
      micButton.disabled = false;
    } else if (type === 'error') {
      setStatus(e.data.message, true);
    }
  };

  // Compile and send WASM module to worklet
  setStatus('Loading WASM...');
  const wasmResponse = await fetch('pkg-no-modules/nam_wasm_bg.wasm');
  const wasmBytes = await wasmResponse.arrayBuffer();
  const compiledModule = await WebAssembly.compile(wasmBytes);
  workletNode.port.postMessage({ type: 'init-wasm', module: compiledModule });
}

// Model file loading
modelInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  setStatus(`Loading model: ${file.name}...`);
  await ensureAudioContext();

  const bytes = await file.arrayBuffer();
  workletNode.port.postMessage(
    { type: 'load-model', modelBytes: bytes },
    [bytes],
  );
});

// Audio file playback
audioInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  await ensureAudioContext();
  stopSource();

  setStatus('Decoding audio...');
  const arrayBuffer = await file.arrayBuffer();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

  sourceNode = audioContext.createBufferSource();
  sourceNode.buffer = audioBuffer;
  sourceNode.connect(workletNode);
  sourceNode.onended = () => {
    setStatus('Playback finished.');
    sourceNode = null;
  };
  sourceNode.start();
  setStatus(`Playing: ${file.name}`);
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
    sourceNode.connect(workletNode);
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
