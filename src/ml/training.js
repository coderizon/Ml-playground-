import {
  MOBILE_NET_INPUT_HEIGHT,
  MOBILE_NET_INPUT_WIDTH,
  STOP_DATA_GATHER,
  GESTURE_FEATURE_LENGTH,
} from '../constants.js';
import {
  CAPTURE_VIDEO,
  PREVIEW_VIDEO,
  STATUS,
  TRAIN_BUTTON,
  trainingProgress,
  trainingProgressFill,
  trainingProgressLabel,
  trainingProgressValue,
} from '../domRefs.js';
import { state } from '../state.js';
import { lockCapturePanels, updateExampleCounts } from '../ui/classes.js';
import { renderProbabilities } from '../ui/probabilities.js';
import { setMobileStep } from '../ui/steps.js';
import { ensureGestureRecognizer, normalizeGestureLandmarks, runGestureStep } from './gesture.js';
import { runFaceStep } from './face.js';

const defaultTrainLabel = TRAIN_BUTTON ? TRAIN_BUTTON.textContent : 'Modell trainieren';

export async function trainAndPredict() {
  if (state.currentMode === 'face') {
    if (STATUS) {
      STATUS.innerText = 'Face Landmarker ist reines Inferenz-Modell.';
    }
    return;
  }
  if (state.trainingCompleted || state.trainingInProgress) return;
  if (!state.trainingDataInputs.length) {
    if (STATUS) {
      STATUS.innerText = 'Bitte sammle zuerst Beispiele.';
    }
    return;
  }

  const { batchSize, epochs } = getTrainingHyperparams();
  state.predict = false;
  state.trainingInProgress = true;
  setTrainingButtonState(true);
  startTrainingUi(epochs);

  let outputsAsTensor;
  let oneHotOutputs;
  let inputsAsTensor;

  try {
    tf.util.shuffleCombo(state.trainingDataInputs, state.trainingDataOutputs);
    outputsAsTensor = tf.tensor1d(state.trainingDataOutputs, 'int32');
    oneHotOutputs = tf.oneHot(outputsAsTensor, state.classNames.length);
    inputsAsTensor = tf.stack(state.trainingDataInputs);

    await state.model.fit(inputsAsTensor, oneHotOutputs, {
      shuffle: true,
      batchSize,
      epochs,
      callbacks: {
        onEpochEnd: (epoch, logs) => updateTrainingProgressUi(epoch + 1, epochs, logs),
      },
    });

    state.predict = true;
    state.trainingCompleted = true;
    completeTrainingUi(epochs);
    lockCapturePanels();
    showPreview();
    setMobileStep('preview');
    predictLoop();
  } catch (error) {
    handleTrainingError(error);
  } finally {
    outputsAsTensor?.dispose();
    oneHotOutputs?.dispose();
    inputsAsTensor?.dispose();
    state.trainingInProgress = false;
    setTrainingButtonState(false);
  }
}

export function resetTrainingProgress() {
  resetTrainingUi();
}

function setTrainingButtonState(isTraining) {
  if (!TRAIN_BUTTON) return;
  TRAIN_BUTTON.disabled = isTraining;
  TRAIN_BUTTON.textContent = isTraining ? 'Trainiert...' : defaultTrainLabel;
}

function startTrainingUi(totalEpochs) {
  if (trainingProgress) {
    trainingProgress.classList.remove('hidden');
  }
  if (trainingProgressLabel) {
    trainingProgressLabel.textContent = `Training wird vorbereitet (${totalEpochs} Epochen)...`;
  }
  if (trainingProgressValue) {
    trainingProgressValue.textContent = '0%';
  }
  if (trainingProgressFill) {
    trainingProgressFill.style.width = '0%';
  }
  if (STATUS) {
    STATUS.innerText = `Training wird vorbereitet (${totalEpochs} Epochen)...`;
  }
}

function updateTrainingProgressUi(currentEpoch, totalEpochs, logs) {
  const percent = Math.min(100, Math.round((currentEpoch / totalEpochs) * 100));
  const acc = logs?.acc ?? logs?.accuracy;
  const accValue = typeof acc === 'number' && Number.isFinite(acc) ? (acc * 100).toFixed(1) : null;
  const accText = accValue ? ` · acc ${accValue}%` : '';
  const accStatus = accValue ? ` Accuracy: ${accValue}%.` : '';

  if (trainingProgressFill) {
    trainingProgressFill.style.width = `${percent}%`;
  }
  if (trainingProgressValue) {
    trainingProgressValue.textContent = `${percent}%`;
  }
  if (trainingProgressLabel) {
    trainingProgressLabel.textContent = `Training läuft (${currentEpoch}/${totalEpochs})${accText}`;
  }
  if (STATUS) {
    STATUS.innerText = `Training läuft. Epoche ${currentEpoch} von ${totalEpochs}.${accStatus}`;
  }
}

function completeTrainingUi(totalEpochs) {
  updateTrainingProgressUi(totalEpochs, totalEpochs);
  if (trainingProgressLabel) {
    trainingProgressLabel.textContent = 'Training abgeschlossen';
  }
  if (STATUS) {
    STATUS.innerText = 'Training abgeschlossen.';
  }
}

function handleTrainingError(error) {
  console.error(error);
  if (trainingProgressLabel) {
    trainingProgressLabel.textContent = 'Training fehlgeschlagen';
  }
  if (STATUS) {
    STATUS.innerText = 'Training fehlgeschlagen. Details in der Konsole.';
  }
}

function resetTrainingUi() {
  state.trainingInProgress = false;
  if (trainingProgress) {
    trainingProgress.classList.add('hidden');
  }
  if (trainingProgressLabel) {
    trainingProgressLabel.textContent = 'Bereit zum Trainieren';
  }
  if (trainingProgressValue) {
    trainingProgressValue.textContent = '0%';
  }
  if (trainingProgressFill) {
    trainingProgressFill.style.width = '0%';
  }
  setTrainingButtonState(false);
}

function getTrainingHyperparams() {
  const safeBatchSize = sanitizeInteger(state.trainingBatchSize, 5);
  const safeEpochs = sanitizeInteger(state.trainingEpochs, 10);
  return { batchSize: safeBatchSize, epochs: safeEpochs };
}

function sanitizeInteger(value, fallback) {
  const parsed = parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
  return parsed;
}

export function showPreview() {
  if (STATUS) {
    STATUS.innerText = '';
  }
  PREVIEW_VIDEO.classList.remove('hidden');
  if (PREVIEW_VIDEO.readyState >= 2) {
    state.previewReady = true;
    renderProbabilities(state.lastPrediction);
    return;
  }
  PREVIEW_VIDEO.addEventListener(
    'loadeddata',
    function onPreviewReady() {
      state.previewReady = true;
      renderProbabilities(state.lastPrediction);
      PREVIEW_VIDEO.removeEventListener('loadeddata', onPreviewReady);
    },
    { once: true }
  );
}

export function predictLoop() {
  if (!state.predict) return;

  if (state.currentMode === 'gesture') {
    runGestureStep();
    window.requestAnimationFrame(predictLoop);
    return;
  }
  if (state.currentMode === 'face') {
    runFaceStep();
    window.requestAnimationFrame(predictLoop);
    return;
  }

  if (state.previewReady && state.trainingCompleted) {
    tf.tidy(function () {
      const videoFrameAsTensor = tf.browser.fromPixels(PREVIEW_VIDEO).div(255);
      const resizedTensorFrame = tf.image.resizeBilinear(
        videoFrameAsTensor,
        [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
        true
      );

      const imageFeatures = state.mobilenet.predict(resizedTensorFrame.expandDims());
      const prediction = state.model.predict(imageFeatures).squeeze();
      const predictionArray = Array.from(prediction.arraySync());
      const highestIndex =
        predictionArray.length > 0
          ? predictionArray.reduce(
              (bestIdx, value, idx, arr) => (value > arr[bestIdx] ? idx : bestIdx),
              0
            )
          : 0;
      renderProbabilities(predictionArray, highestIndex, state.classNames);
    });
  }

  window.requestAnimationFrame(predictLoop);
}

export function handleCollectStart(event) {
  if (state.currentMode === 'face') return;
  event.preventDefault();
  const classNumber = parseInt(event.currentTarget.getAttribute('data-1hot'), 10);
  state.gatherDataState = classNumber;
  dataGatherLoop();
}

export function handleCollectEnd(event) {
  event.preventDefault();
  state.gatherDataState = STOP_DATA_GATHER;
}

async function dataGatherLoop() {
  if (state.currentMode === 'face') return;
  if (state.videoPlaying && state.gatherDataState !== STOP_DATA_GATHER) {
    if (state.currentMode === 'gesture') {
      await collectGestureExample();
    } else {
      collectImageExample();
    }
    window.requestAnimationFrame(dataGatherLoop);
  }
}

async function collectGestureExample() {
  try {
    const recognizer = await ensureGestureRecognizer();
    if (!recognizer) return;

    const result = recognizer.recognizeForVideo(CAPTURE_VIDEO, performance.now());
    const landmarks = result?.landmarks?.[0];
    if (!landmarks) return;

    const featureVector = normalizeGestureLandmarks(landmarks);
    if (!featureVector || featureVector.length !== GESTURE_FEATURE_LENGTH) return;

    const featuresTensor = tf.tensor1d(featureVector);
    state.trainingDataInputs.push(featuresTensor);
    state.trainingDataOutputs.push(state.gatherDataState);
    handleExampleBookkeeping();
  } catch (error) {
    console.error(error);
  }
}

function collectImageExample() {
  const imageFeatures = tf.tidy(function () {
    const videoFrameAsTensor = tf.browser.fromPixels(CAPTURE_VIDEO);
    const resizedTensorFrame = tf.image.resizeBilinear(
      videoFrameAsTensor,
      [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
      true
    );
    const normalizedTensorFrame = resizedTensorFrame.div(255);
    return state.mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
  });

  state.trainingDataInputs.push(imageFeatures);
  state.trainingDataOutputs.push(state.gatherDataState);
  handleExampleBookkeeping();
}

function handleExampleBookkeeping() {
  vibrateFeedback();

  if (state.examplesCount[state.gatherDataState] === undefined) {
    state.examplesCount[state.gatherDataState] = 0;
  }
  state.examplesCount[state.gatherDataState]++;

  updateExampleCounts();
}

function vibrateFeedback() {
  if (navigator.vibrate) {
    navigator.vibrate(15);
  }
}
