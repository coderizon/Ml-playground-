import {
  AUDIO_OVERLAP_FACTOR,
  AUDIO_PREVIEW_PROBABILITY_THRESHOLD,
  AUDIO_SAMPLE_DURATION_MS,
  AUDIO_VIZ_FFT_SIZE,
  STOP_DATA_GATHER,
} from '../constants.js';
import { STATUS } from '../domRefs.js';
import { getState, mutateState } from '../state.js';
import { updateExampleCounts } from '../ui/classes.js';

let speechCommandsLib = null;
let visualizerCanvas = null;
let visualizerCtx = null;
let audioContext = null;
let analyser = null;
let analyserData;
let visualizerHandle = null;

async function ensureSpeechCommands() {
  if (speechCommandsLib) return speechCommandsLib;
  speechCommandsLib = await import('https://cdn.jsdelivr.net/npm/@tensorflow-models/speech-commands');
  return speechCommandsLib;
}

export async function ensureAudioBackend() {
  const state = getState();
  if (state.audioTransferRecognizer) return state.audioTransferRecognizer;

  const speechCommands = await ensureSpeechCommands();
  const recognizer = speechCommands.create('BROWSER_FFT');
  await recognizer.ensureModelLoaded();
  const transfer = recognizer.createTransfer('custom');

  mutateState((draft) => {
    draft.audioRecognizer = recognizer;
    draft.audioTransferRecognizer = transfer;
  });

  if (STATUS) {
    STATUS.innerText = 'Audio-Backend geladen. Sammle Beispiele.';
  }

  return transfer;
}

export async function startAudioCollection(labelId) {
  const transfer = await ensureAudioBackend();
  if (!transfer) return;
  const label = getLabelForClass(labelId);

  mutateState((draft) => {
    draft.audioCollecting = true;
    draft.gatherDataState = labelId;
  });

  const collectOnce = async () => {
    const state = getState();
    if (!state.audioCollecting || state.gatherDataState !== labelId) return;
    try {
      await transfer.collectExample(label, { durationSec: AUDIO_SAMPLE_DURATION_MS / 1000 });
      mutateState((draft) => {
        if (draft.examplesCount[labelId] === undefined) {
          draft.examplesCount[labelId] = 0;
        }
        draft.examplesCount[labelId]++;
      });
      updateExampleCounts();
    } catch (error) {
      console.error(error);
      if (STATUS) {
        STATUS.innerText = 'Audioaufnahme fehlgeschlagen.';
      }
    }
    if (getState().audioCollecting && getState().gatherDataState === labelId) {
      window.setTimeout(collectOnce, 120);
    }
  };

  collectOnce();
}

export function stopAudioCollection() {
  mutateState((draft) => {
    draft.audioCollecting = false;
    draft.gatherDataState = STOP_DATA_GATHER;
  });
}

export async function trainAudioModel({ epochs, batchSize, onEpochEnd }) {
  const transfer = await ensureAudioBackend();
  const exampleCount = transfer.countExamples();
  const hasExamples = exampleCount && Object.keys(exampleCount).length > 0;
  if (!hasExamples) {
    throw new Error('Bitte sammle zuerst Audio-Beispiele.');
  }

  await transfer.train({
    epochs,
    batchSize,
    callback: {
      onEpochEnd: async (epoch, logs) => {
        if (typeof onEpochEnd === 'function') {
          onEpochEnd(epoch, logs);
        }
      },
    },
  });

  return transfer;
}

export async function startAudioPreview(onPrediction) {
  const state = getState();
  if (state.audioPreviewActive) return state.audioTransferRecognizer;
  const transfer = await ensureAudioBackend();
  if (!transfer) return null;

  await transfer.listen(
    (result) => {
      const probabilities = Array.from(result?.scores ?? []);
      const bestIndex =
        probabilities.length > 0
          ? probabilities.reduce(
              (bestIdx, value, idx, arr) => (value > arr[bestIdx] ? idx : bestIdx),
              0
            )
          : -1;
      if (typeof onPrediction === 'function') {
        onPrediction(probabilities, bestIndex, getState().classNames);
      }
      if (result?.spectrogram?.data) {
        drawFromSpectrogram(result.spectrogram.data, result.spectrogram.frameSize);
      }
    },
    {
      includeSpectrogram: true,
      probabilityThreshold: AUDIO_PREVIEW_PROBABILITY_THRESHOLD,
      overlapFactor: AUDIO_OVERLAP_FACTOR,
    }
  );

  mutateState((draft) => {
    draft.audioPreviewActive = true;
  });
  return transfer;
}

export function stopAudioPreview() {
  const { audioTransferRecognizer } = getState();
  if (audioTransferRecognizer) {
    try {
      audioTransferRecognizer.stopListening();
    } catch (error) {
      console.warn('Stop listening failed', error);
    }
  }
  mutateState((draft) => {
    draft.audioPreviewActive = false;
  });
}

export function resetAudioState() {
  stopAudioCollection();
  stopAudioPreview();
  stopVisualizer();
  const { audioVisualizerStream } = getState();
  if (audioVisualizerStream) {
    audioVisualizerStream.getTracks().forEach((track) => track.stop());
  }
  mutateState((draft) => {
    draft.audioVisualizerStream = null;
    draft.audioCollecting = false;
  });
}

export function clearAudioExamples() {
  const { audioTransferRecognizer } = getState();
  if (audioTransferRecognizer && typeof audioTransferRecognizer.clearExamples === 'function') {
    audioTransferRecognizer.clearExamples();
  }
}

export function attachAudioVisualizer(slot) {
  if (!slot) return;
  if (!visualizerCanvas) {
    visualizerCanvas = document.createElement('canvas');
    visualizerCanvas.id = 'audioVisualizer';
  }
  if (visualizerCanvas.parentElement !== slot) {
    slot.innerHTML = '';
    slot.appendChild(visualizerCanvas);
  }
  resizeVisualizer(slot);
  startVisualizer();
}

function getLabelForClass(idx) {
  const { classNames } = getState();
  return classNames[idx] || `Class ${idx + 1}`;
}

function resizeVisualizer(slot) {
  if (!visualizerCanvas || !slot) return;
  const { width, height } = slot.getBoundingClientRect();
  visualizerCanvas.width = Math.max(320, Math.floor(width));
  visualizerCanvas.height = Math.max(120, Math.floor(height || 180));
}

async function startVisualizer() {
  if (!visualizerCanvas) return;
  if (visualizerHandle) return;

  const stream = await ensureAudioStream();
  if (!stream) return;

  visualizerCtx = visualizerCanvas.getContext('2d');
  if (!visualizerCtx) return;
  analyserData = new Uint8Array(analyser.frequencyBinCount);

  const render = () => {
    if (!visualizerCtx || !analyser) return;
    visualizerHandle = requestAnimationFrame(render);
    analyser.getByteFrequencyData(analyserData);
    visualizerCtx.clearRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
    drawBars(visualizerCtx, analyserData, visualizerCanvas.width, visualizerCanvas.height);
  };

  render();
}

function stopVisualizer() {
  if (visualizerHandle) {
    cancelAnimationFrame(visualizerHandle);
    visualizerHandle = null;
  }
  if (visualizerCtx && visualizerCanvas) {
    visualizerCtx.clearRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
  }
  analyser = null;
  analyserData = null;
  audioContext = null;
}

async function ensureAudioStream() {
  const current = getState().audioVisualizerStream;
  if (current) {
    setupAnalyser(current);
    return current;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    mutateState((draft) => {
      draft.audioVisualizerStream = stream;
    });
    setupAnalyser(stream);
    return stream;
  } catch (error) {
    console.error('Mikrofon konnte nicht geöffnet werden', error);
    if (STATUS) {
      STATUS.innerText = 'Mikrofon konnte nicht geöffnet werden.';
    }
    return null;
  }
}

function setupAnalyser(stream) {
  if (analyser) return;
  audioContext = audioContext || new AudioContext();
  const source = audioContext.createMediaStreamSource(stream);
  analyser = audioContext.createAnalyser();
  analyser.fftSize = AUDIO_VIZ_FFT_SIZE;
  source.connect(analyser);
}

function drawBars(ctx, data, width, height) {
  const barWidth = Math.max(2, Math.floor(width / data.length));
  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, '#22d3ee');
  gradient.addColorStop(1, '#0ea5e9');
  for (let i = 0; i < data.length; i++) {
    const value = data[i] / 255;
    const barHeight = value * height;
    ctx.fillStyle = gradient;
    ctx.fillRect(i * barWidth, height - barHeight, barWidth - 1, barHeight);
  }
}

function drawFromSpectrogram(values = [], frameSize = 0) {
  if (!visualizerCtx || !visualizerCanvas || !values.length) return;
  const numFrames = Math.floor(values.length / frameSize);
  visualizerCtx.clearRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
  const maxFrames = Math.min(numFrames, 40);
  for (let i = 0; i < maxFrames; i++) {
    const start = i * frameSize;
    const slice = values.slice(start, start + frameSize);
    const avg =
      slice.reduce((sum, val) => sum + Math.abs(val || 0), 0) / Math.max(1, slice.length);
    const height = (avg / 10) * visualizerCanvas.height;
    visualizerCtx.fillStyle = `rgba(34, 211, 238, ${0.2 + (i / maxFrames) * 0.8})`;
    visualizerCtx.fillRect(
      (i / maxFrames) * visualizerCanvas.width,
      visualizerCanvas.height - height,
      visualizerCanvas.width / maxFrames - 4,
      height
    );
  }
}
