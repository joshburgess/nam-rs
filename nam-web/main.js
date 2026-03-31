// main.js — Main thread NAM Web Audio controller

import initWasm, { NamModel } from './pkg/nam_wasm.js';
import { saveProfile, listProfiles, loadProfile, deleteProfile } from './lib/profile-store.js';

let audioContext = null;
let workletNode = null;
let scriptNode = null;
let sourceNode = null;
let micStream = null;
let mainThreadModel = null;
let convolverNode = null;
let eqLow = null;
let eqMid = null;
let eqHigh = null;
let delayNode = null;
let delayFeedback = null;
let delayDry = null;
let delayWet = null;
let reverbConvolver = null;
let reverbDry = null;
let reverbWet = null;
let eqEnabled = false;
let delayEnabled = false;
let reverbEnabled = false;
let useWorklet = false;
let wasmInitialized = false;
let activeProfileId = null;
let currentModelName = null;

const statusEl = document.getElementById('status');
const modelInput = document.getElementById('model-input');
const audioInput = document.getElementById('audio-input');
const audioLabel = document.getElementById('audio-label');
const irInput = document.getElementById('ir-input');
const irStatusEl = document.getElementById('ir-status');
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

function ensureEffectNodes() {
  if (eqLow) return; // already created

  const ctx = audioContext;

  // 3-band EQ
  eqLow = ctx.createBiquadFilter();
  eqLow.type = 'lowshelf';
  eqLow.frequency.value = 320;
  eqLow.gain.value = 0;

  eqMid = ctx.createBiquadFilter();
  eqMid.type = 'peaking';
  eqMid.frequency.value = 1000;
  eqMid.Q.value = 0.7;
  eqMid.gain.value = 0;

  eqHigh = ctx.createBiquadFilter();
  eqHigh.type = 'highshelf';
  eqHigh.frequency.value = 3200;
  eqHigh.gain.value = 0;

  // Delay (feedback delay with wet/dry mix)
  delayNode = ctx.createDelay(2.0);
  delayNode.delayTime.value = 0.35;
  delayFeedback = ctx.createGain();
  delayFeedback.gain.value = 0.4;
  delayDry = ctx.createGain();
  delayDry.gain.value = 1.0;
  delayWet = ctx.createGain();
  delayWet.gain.value = 0.3;

  // Delay feedback loop: delay → feedback → delay
  delayNode.connect(delayFeedback);
  delayFeedback.connect(delayNode);
  delayNode.connect(delayWet);

  // Reverb (synthetic IR via offline rendering)
  reverbConvolver = ctx.createConvolver();
  reverbConvolver.normalize = true;
  reverbDry = ctx.createGain();
  reverbDry.gain.value = 1.0;
  reverbWet = ctx.createGain();
  reverbWet.gain.value = 0.25;

  generateReverbIR(2.0);
}

function generateReverbIR(decay) {
  const ctx = audioContext;
  const rate = ctx.sampleRate;
  const length = Math.floor(rate * decay);
  const buffer = ctx.createBuffer(2, length, rate);

  for (let ch = 0; ch < 2; ch++) {
    const data = buffer.getChannelData(ch);
    for (let i = 0; i < length; i++) {
      data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, 2);
    }
  }

  reverbConvolver.buffer = buffer;
}

// Reconnect the full output chain:
// processor → [cab IR] → [EQ] → [delay dry/wet] → [reverb dry/wet] → destination
function reconnectOutput() {
  const processor = getProcessorNode();
  if (!processor || !audioContext) return;

  ensureEffectNodes();

  // Disconnect everything first
  processor.disconnect();
  if (convolverNode) convolverNode.disconnect();
  eqLow.disconnect();
  eqMid.disconnect();
  eqHigh.disconnect();
  delayDry.disconnect();
  delayWet.disconnect();
  reverbDry.disconnect();
  reverbWet.disconnect();
  reverbConvolver.disconnect();

  // Build chain
  let current = processor;

  // Cab IR (optional)
  if (convolverNode) {
    current.connect(convolverNode);
    current = convolverNode;
  }

  // EQ (bypass = connect through, enabled = chain filters)
  if (eqEnabled) {
    current.connect(eqLow);
    eqLow.connect(eqMid);
    eqMid.connect(eqHigh);
    current = eqHigh;
  }

  // Delay (bypass = dry only, enabled = dry + wet parallel)
  if (delayEnabled) {
    current.connect(delayDry);
    current.connect(delayNode); // feeds the delay line → delayWet

    // Merge dry + wet into reverb or destination
    if (reverbEnabled) {
      delayDry.connect(reverbDry);
      delayDry.connect(reverbConvolver);
      delayWet.connect(reverbDry);
      delayWet.connect(reverbConvolver);
      reverbConvolver.connect(reverbWet);
      reverbDry.connect(audioContext.destination);
      reverbWet.connect(audioContext.destination);
    } else {
      delayDry.connect(audioContext.destination);
      delayWet.connect(audioContext.destination);
    }
  } else if (reverbEnabled) {
    current.connect(reverbDry);
    current.connect(reverbConvolver);
    reverbConvolver.connect(reverbWet);
    reverbDry.connect(audioContext.destination);
    reverbWet.connect(audioContext.destination);
  } else {
    current.connect(audioContext.destination);
  }
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
    // Connected via reconnectOutput() after init
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
      reconnectOutput();
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
  reconnectOutput();
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

// ── Effects controls ──

function setupFxToggle(checkboxId, controlsId, onToggle) {
  const checkbox = document.getElementById(checkboxId);
  const controls = document.getElementById(controlsId);
  checkbox.addEventListener('change', () => {
    const enabled = checkbox.checked;
    controls.classList.toggle('visible', enabled);
    onToggle(enabled);
  });
}

function setupSlider(sliderId, valueId, format, onChange) {
  const slider = document.getElementById(sliderId);
  const display = document.getElementById(valueId);
  const update = () => {
    const v = parseFloat(slider.value);
    display.textContent = format(v);
    onChange(v);
  };
  slider.addEventListener('input', update);
  update(); // set initial display
}

// EQ
setupFxToggle('eq-enabled', 'eq-controls', (on) => {
  eqEnabled = on;
  reconnectOutput();
});

setupSlider('eq-low', 'eq-low-val', (v) => `${v > 0 ? '+' : ''}${v} dB`, (v) => {
  if (eqLow) eqLow.gain.value = v;
});

setupSlider('eq-mid', 'eq-mid-val', (v) => `${v > 0 ? '+' : ''}${v} dB`, (v) => {
  if (eqMid) eqMid.gain.value = v;
});

setupSlider('eq-high', 'eq-high-val', (v) => `${v > 0 ? '+' : ''}${v} dB`, (v) => {
  if (eqHigh) eqHigh.gain.value = v;
});

// Delay
setupFxToggle('delay-enabled', 'delay-controls', (on) => {
  delayEnabled = on;
  reconnectOutput();
});

setupSlider('delay-time', 'delay-time-val', (v) => `${Math.round(v * 1000)} ms`, (v) => {
  if (delayNode) delayNode.delayTime.value = v;
});

setupSlider('delay-feedback', 'delay-feedback-val', (v) => `${Math.round(v * 100)}%`, (v) => {
  if (delayFeedback) delayFeedback.gain.value = v;
});

setupSlider('delay-mix', 'delay-mix-val', (v) => `${Math.round(v * 100)}%`, (v) => {
  if (delayWet) delayWet.gain.value = v;
  if (delayDry) delayDry.gain.value = 1.0 - v * 0.5; // keep dry present
});

// Reverb
setupFxToggle('reverb-enabled', 'reverb-controls', (on) => {
  reverbEnabled = on;
  reconnectOutput();
});

setupSlider('reverb-decay', 'reverb-decay-val', (v) => `${v.toFixed(1)} s`, (v) => {
  if (audioContext && reverbConvolver) generateReverbIR(v);
});

setupSlider('reverb-mix', 'reverb-mix-val', (v) => `${Math.round(v * 100)}%`, (v) => {
  if (reverbWet) reverbWet.gain.value = v;
  if (reverbDry) reverbDry.gain.value = 1.0 - v * 0.5;
});

// ── Impulse Response ──

irInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  await ensureAudioContext();

  irStatusEl.textContent = `Loading: ${file.name}...`;

  try {
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    // Create or replace convolver
    if (convolverNode) {
      convolverNode.disconnect();
    }
    convolverNode = audioContext.createConvolver();
    convolverNode.normalize = true;
    convolverNode.buffer = audioBuffer;

    reconnectOutput();
    irStatusEl.textContent = `IR: ${file.name} (${audioBuffer.duration.toFixed(2)}s, ${audioBuffer.sampleRate} Hz)`;
  } catch (err) {
    irStatusEl.textContent = `IR load failed: ${err.message}`;
  }

  irInput.value = '';
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
