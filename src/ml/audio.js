import { getState, mutateState } from '../state.js';
import { updateExampleCounts } from '../ui/classes.js';

const BACKGROUND_LABEL = '_background_noise_';

function getSpeechCommands() {
  return window?.speechCommands;
}

export async function initAudio() {
  const state = getState();
  if (state.audioBaseRecognizer && state.audioTransferRecognizer) return;

  const api = getSpeechCommands();
  if (!api?.create) {
    console.error('speech-commands Bibliothek nicht geladen.');
    return;
  }

  try {
    const baseRecognizer = api.create('BROWSER_FFT');
    await baseRecognizer.ensureModelLoaded();
    const transferRecognizer = baseRecognizer.createTransfer(state.audioModelName);

    mutateState((draft) => {
      draft.audioBaseRecognizer = baseRecognizer;
      draft.audioTransferRecognizer = transferRecognizer;
    });
    console.log('Audio Modell geladen.');
  } catch (err) {
    console.error('Audio Init Fehler:', err);
  }
}

export async function collectAudioSample(label) {
  await initAudio();
  const state = getState();
  const recognizer = state.audioTransferRecognizer;
  const trimmed = (label || '').trim();
  if (!recognizer || !trimmed) return;

  const isBackground = trimmed === BACKGROUND_LABEL;
  const duration = isBackground ? 20 : 2;
  const samplesCount = isBackground ? 1 : 2;

  showProgress(true);
  toggleMidMarker(!isBackground);

  console.log(`Nehme auf für: ${trimmed}`);

  try {
    if (isBackground) {
      animateProgressBar(duration);
      await recognizer.collectExample(BACKGROUND_LABEL, { durationSec: duration });
    } else {
      for (let i = 0; i < samplesCount; i++) {
        animateSegment(i, samplesCount);
        await recognizer.collectExample(trimmed);
      }
    }

    syncExampleCounts(recognizer);
  } catch (error) {
    console.error('Fehler bei Audioaufnahme:', error);
  } finally {
    showProgress(false);
    toggleMidMarker(false);
  }
}

export async function trainAudioModel(epochs = 25, callback) {
  await initAudio();
  const state = getState();
  const recognizer = state.audioTransferRecognizer;
  if (!recognizer) return;

  await recognizer.train({
    epochs,
    callback:
      callback ||
      {
        onEpochEnd: async (epoch, logs) => {
          console.log(`Epoch ${epoch}: Acc=${logs.acc?.toFixed(3) ?? 'n/a'}`);
        },
      },
  });
}

export async function startAudioListening(onResult) {
  await initAudio();
  const state = getState();
  const recognizer = state.audioTransferRecognizer;
  if (!recognizer || state.isAudioListening) return;

  mutateState((draft) => {
    draft.isAudioListening = true;
  });

  try {
    await recognizer.listen(
      (result) => {
        const words = recognizer.wordLabels();
        let maxScore = 0;
        let maxWord = '';
        for (let i = 0; i < words.length; i++) {
          if (result.scores[i] > maxScore) {
            maxScore = result.scores[i];
            maxWord = words[i];
          }
        }
        if (onResult) {
          onResult({ label: maxWord, score: maxScore, scores: result.scores, labels: words });
        }
      },
      {
        probabilityThreshold: 0.75,
        invokeCallbackOnNoiseAndUnknown: true,
      }
    );
  } catch (error) {
    console.error(error);
    mutateState((draft) => {
      draft.isAudioListening = false;
    });
  }
}

export async function stopAudioListening() {
  const state = getState();
  const recognizer = state.audioTransferRecognizer;
  try {
    if (recognizer && state.isAudioListening) {
      await recognizer.stopListening();
    }
  } finally {
    mutateState((draft) => {
      draft.isAudioListening = false;
    });
  }
}

export function clearAudioExamples() {
  const state = getState();
  if (state.audioTransferRecognizer?.clearExamples) {
    state.audioTransferRecognizer.clearExamples();
  }
}

function syncExampleCounts(recognizer) {
  const counts = recognizer?.countExamples() || {};
  mutateState((draft) => {
    draft.classNames.forEach((name, idx) => {
      if (counts[name] !== undefined) {
        draft.examplesCount[idx] = counts[name];
      }
    });
  });
  updateExampleCounts();
}

function showProgress(show) {
  const container = document.getElementById('audio-progress-container');
  const bar = document.getElementById('audio-progress-bar');
  if (container) {
    container.style.display = show ? 'block' : 'none';
    if (!show && bar) {
      bar.style.width = '0%';
      bar.innerText = '0%';
    }
  }
}

function toggleMidMarker(show) {
  const marker = document.getElementById('audio-mid-marker');
  if (marker) {
    marker.style.display = show ? 'block' : 'none';
  }
}

function animateProgressBar(seconds) {
  const bar = document.getElementById('audio-progress-bar');
  if (!bar) return;

  bar.style.transition = `width ${seconds}s linear`;
  setTimeout(() => {
    bar.style.width = '100%';
  }, 50);

  let remaining = seconds;
  bar.innerText = `Noch ${remaining}s`;

  const interval = setInterval(() => {
    remaining--;
    if (remaining <= 0) {
      clearInterval(interval);
    }
    bar.innerText = `Noch ${remaining}s`;
  }, 1000);
}

function animateSegment(index, total) {
  const bar = document.getElementById('audio-progress-bar');
  if (!bar) return;

  const segmentSize = 100 / total;
  const endPos = (index + 1) * segmentSize;

  bar.style.transition = 'width 1s linear';
  bar.style.width = `${endPos}%`;
  bar.innerText = `Sprechen! (${index + 1}/${total})`;
}
