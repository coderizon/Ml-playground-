import {
  MOBILE_NET_INPUT_HEIGHT,
  MOBILE_NET_INPUT_WIDTH,
  STOP_DATA_GATHER,
} from '../constants.js';
import { CAPTURE_VIDEO, PREVIEW_VIDEO, STATUS } from '../domRefs.js';
import { state } from '../state.js';
import { lockCapturePanels, updateExampleCounts } from '../ui/classes.js';
import { renderProbabilities } from '../ui/probabilities.js';
import { setMobileStep } from '../ui/steps.js';
import { runGestureStep } from './gesture.js';

export async function trainAndPredict() {
  if (state.trainingCompleted) return;
  state.predict = false;
  tf.util.shuffleCombo(state.trainingDataInputs, state.trainingDataOutputs);
  const outputsAsTensor = tf.tensor1d(state.trainingDataOutputs, 'int32');
  const oneHotOutputs = tf.oneHot(outputsAsTensor, state.classNames.length);
  const inputsAsTensor = tf.stack(state.trainingDataInputs);

  const { batchSize, epochs } = getTrainingHyperparams();

  await state.model.fit(inputsAsTensor, oneHotOutputs, {
    shuffle: true,
    batchSize,
    epochs,
    callbacks: { onEpochEnd: logProgress },
  });

  outputsAsTensor.dispose();
  oneHotOutputs.dispose();
  inputsAsTensor.dispose();
  state.predict = true;
  state.trainingCompleted = true;
  lockCapturePanels();
  showPreview();
  setMobileStep('preview');
  predictLoop();
}

function logProgress(epoch, logs) {
  console.log('Data for epoch ' + epoch, logs);
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
  event.preventDefault();
  const classNumber = parseInt(event.currentTarget.getAttribute('data-1hot'), 10);
  state.gatherDataState = classNumber;
  dataGatherLoop();
}

export function handleCollectEnd(event) {
  event.preventDefault();
  state.gatherDataState = STOP_DATA_GATHER;
}

function dataGatherLoop() {
  if (state.videoPlaying && state.gatherDataState !== STOP_DATA_GATHER) {
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
    vibrateFeedback();

    if (state.examplesCount[state.gatherDataState] === undefined) {
      state.examplesCount[state.gatherDataState] = 0;
    }
    state.examplesCount[state.gatherDataState]++;

    updateExampleCounts();
    window.requestAnimationFrame(dataGatherLoop);
  }
}

function vibrateFeedback() {
  if (navigator.vibrate) {
    navigator.vibrate(15);
  }
}
