// main.js — Main thread NAM Web Audio controller

import initWasm, { NamModel } from './pkg/nam_wasm.js';
import { saveProfile, listProfiles, loadProfile, deleteProfile } from './lib/profile-store.js';

let audioContext = null;
let workletNode = null;
let scriptNode = null;
let sourceNode = null;
let micStream = null;
let mainThreadModel = null;
let useWorklet = false;
let wasmInitialized = false;
let activeProfileId = null;
let currentModelName = null;

const statusEl = document.getElementById('status');
const modelInput = document.getElementById('model-input');
const audioInput = document.getElementById('audio-input');
const audioLabel = document.getElementById('audio-label');
const micButton = document.getElementById('mic-toggle');
const profileListEl = document.getElementById('profile-list');
const modelInfoPanel = document.getElementById('model-info-panel');
const infoName = document.getElementById('info-name');
const infoSr = document.getElementById('info-sr');
const infoMode = document.getElementById('info-mode');

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.className = isError ? 'error' : '';
}

function enableAudioControls() {
  audioInput.disabled = false;
  audioLabel.classList.remove('disabled');
  micButton.disabled = false;
}

function showModelInfo(name, sampleRate, mismatch = false, contextRate = 0) {
  currentModelName = name;
  modelInfoPanel.classList.add('visible');
  infoName.textContent = name;
  let srText = `${sampleRate} Hz`;
  if (mismatch) {
    srText += ` (browser: ${contextRate} Hz)`;
  }
  infoSr.textContent = srText;
  const badge = useWorklet ? 'AudioWorklet' : 'ScriptProcessor';
  const cls = useWorklet ? 'badge worklet' : 'badge';
  infoMode.innerHTML = `<span class="${cls}">${badge}</span>`;
}

// ── WASM init ──

initWasm().then(() => {
  wasmInitialized = true;
  setStatus('Select a model from the library or import a .nam file.');
  renderProfileList();
}).catch((err) => {
  setStatus(`WASM init failed: ${err.message}`, true);
});

// ── Audio context + worklet ──

function getProcessorNode() {
  return useWorklet ? workletNode : scriptNode;
}

async function ensureAudioContext() {
  if (audioContext) return;
  audioContext = new AudioContext({ latencyHint: 'interactive' });

  try {
    await audioContext.audioWorklet.addModule('worklet/nam-processor.js');

    workletNode = new AudioWorkletNode(audioContext, 'nam-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    workletNode.connect(audioContext.destination);

    const wasmResponse = await fetch('pkg/nam_wasm_bg.wasm');
    const wasmBytes = await wasmResponse.arrayBuffer();

    const workletReady = await new Promise((resolve) => {
      const timeout = setTimeout(() => resolve(false), 3000);

      workletNode.port.onmessage = (e) => {
        if (e.data.type === 'wasm-ready') {
          clearTimeout(timeout);
          resolve(true);
        } else if (e.data.type === 'error') {
          clearTimeout(timeout);
          console.warn('Worklet error:', e.data.message);
          resolve(false);
        } else if (e.data.type === 'model-ready') {
          showModelInfo(currentModelName, e.data.sampleRate, e.data.sampleRateMismatch, e.data.contextSampleRate);
          if (e.data.sampleRateMismatch) {
            setStatus(`Model ready. Warning: model expects ${e.data.sampleRate} Hz but browser is running at ${e.data.contextSampleRate} Hz.`);
          } else {
            setStatus('Model ready. Load audio or enable mic input.');
          }
          enableAudioControls();
        }
      };

      workletNode.onprocessorerror = () => {
        clearTimeout(timeout);
        resolve(false);
      };

      workletNode.port.postMessage(
        { type: 'init-wasm', wasmBytes },
        [wasmBytes],
      );
    });

    if (workletReady) {
      useWorklet = true;
      return;
    }

    workletNode.disconnect();
    workletNode = null;
  } catch (err) {
    console.warn('AudioWorklet unavailable:', err.message);
  }

  // Fallback
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

// ── Model loading ──

async function loadModelFromBytes(bytes, name) {
  if (!wasmInitialized) {
    setStatus('WASM not ready yet...', true);
    return;
  }

  setStatus(`Loading model: ${name}...`);
  currentModelName = name;
  await ensureAudioContext();

  if (useWorklet) {
    const copy = bytes.slice(0);
    workletNode.port.postMessage(
      { type: 'load-model', modelBytes: copy },
      [copy],
    );
  } else {
    try {
      if (mainThreadModel) {
        mainThreadModel.free();
        mainThreadModel = null;
      }

      mainThreadModel = new NamModel(new Uint8Array(bytes));
      const sr = mainThreadModel.sample_rate() || audioContext.sampleRate;
      mainThreadModel.reset(sr, 512);
      mainThreadModel.prewarm();

      const mismatch = Math.abs(sr - audioContext.sampleRate) > 1;
      showModelInfo(name, sr, mismatch, audioContext.sampleRate);
      if (mismatch) {
        setStatus(`Model ready. Warning: model expects ${sr} Hz but browser is running at ${audioContext.sampleRate} Hz.`);
      } else {
        setStatus('Model ready. Load audio or enable mic input.');
      }
      enableAudioControls();
    } catch (err) {
      setStatus(`Model load failed: ${err.message}`, true);
    }
  }
}

// ── Profile list ──

async function renderProfileList() {
  try {
    const profiles = await listProfiles();

    if (profiles.length === 0) {
      profileListEl.innerHTML = '<div class="profile-empty">No models saved yet.<br>Import a .nam file above.</div>';
      return;
    }

    profileListEl.innerHTML = profiles
      .map((p) => {
        const sizeKB = (p.size / 1024).toFixed(0);
        const isActive = p.id === activeProfileId;
        return `
          <div class="profile-item ${isActive ? 'active' : ''}" data-id="${p.id}">
            <div class="profile-item-info">
              <div class="profile-item-name">${escapeHtml(p.name)}</div>
              <div class="profile-item-meta">${sizeKB} KB</div>
            </div>
            <button class="profile-item-delete" data-delete-id="${p.id}" title="Delete">&times;</button>
          </div>
        `;
      })
      .join('');

    // Click to load
    profileListEl.querySelectorAll('.profile-item').forEach((el) => {
      el.addEventListener('click', async (e) => {
        if (e.target.closest('.profile-item-delete')) return;
        const id = Number(el.dataset.id);
        activeProfileId = id;
        renderProfileList();
        const bytes = await loadProfile(id);
        const name = profiles.find((p) => p.id === id)?.name || 'Unknown';
        await loadModelFromBytes(bytes, name);
      });
    });

    // Delete buttons
    profileListEl.querySelectorAll('.profile-item-delete').forEach((btn) => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const id = Number(btn.dataset.deleteId);
        await deleteProfile(id);
        if (activeProfileId === id) activeProfileId = null;
        renderProfileList();
      });
    });
  } catch (err) {
    console.error('Failed to load profile list:', err);
  }
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ── File import ──

modelInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const bytes = await file.arrayBuffer();
  const name = file.name.replace(/\.nam$/i, '');

  // Save to IndexedDB
  await saveProfile(name, bytes);
  await renderProfileList();

  // Load it
  activeProfileId = (await listProfiles()).slice(-1)[0]?.id;
  renderProfileList();
  await loadModelFromBytes(bytes, name);

  // Reset input so same file can be re-imported
  modelInput.value = '';
});

// ── Audio file playback ──

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

// ── Mic input ──

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
