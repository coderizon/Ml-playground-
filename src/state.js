import {
  DEFAULT_TRAINING_BATCH_SIZE,
  DEFAULT_TRAINING_EPOCHS,
  DEFAULT_TRAINING_LR,
  STOP_DATA_GATHER,
} from './constants.js';

function createInitialState() {
  return {
    currentStream: null,
    activeClassIndex: 0,
    previewReady: false,
    trainingCompleted: false,
    trainingInProgress: false,
    mobilenet: undefined,
    gatherDataState: STOP_DATA_GATHER,
    videoPlaying: false,
    trainingDataInputs: [],
    trainingDataOutputs: [],
    examplesCount: [],
    predict: false,
    model: null,
    preferredFacingMode: 'user',
    lastPrediction: [],
    currentMode: 'image',
    audioBaseRecognizer: null,
    audioTransferRecognizer: null,
    isAudioListening: false,
    audioModelName: 'mein-audio-modell',
    handLandmarker: null,
    handInitPromise: null,
    handBusy: false,
    handDrawingUtils: null,
    handVision: null,
    gestureSamples: [],
    gestureLastSampleTs: 0,
    faceLandmarker: null,
    faceInitPromise: null,
    faceBusy: false,
    faceDrawingUtils: null,
    faceVision: null,
    poseLandmarker: null,
    poseInitPromise: null,
    poseBusy: false,
    poseVision: null,
    poseDrawingUtils: null,
    poseSamples: [],
    poseLastSampleTs: 0,
    arduinoConnected: false,
    microbitConnected: false,
    calliopeConnected: false,
    trainingEpochs: DEFAULT_TRAINING_EPOCHS,
    trainingBatchSize: DEFAULT_TRAINING_BATCH_SIZE,
    trainingLearningRate: DEFAULT_TRAINING_LR,
    lastSentLabel: null,
    lastSentAt: 0,
    classNames: [],
    openWebcamButtons: [],
    webcamPanels: [],
    classNameInputs: [],
    captureSlots: [],
    dataCollectorButtons: [],
    countChips: [],
    switchCameraButtons: [],
  };
}

const internalState = createInitialState();
const listeners = new Set();

function notify() {
  listeners.forEach((listener) => listener(internalState));
}

export function getState() {
  return internalState;
}

export function setState(updates, options = {}) {
  if (!updates) return internalState;
  const patch = typeof updates === 'function' ? updates({ ...internalState }) : updates;
  if (patch && typeof patch === 'object') {
    Object.assign(internalState, patch);
    if (!options.silent) notify();
  }
  return internalState;
}

export function mutateState(mutator, options = {}) {
  if (typeof mutator === 'function') {
    mutator(internalState);
    if (!options.silent) notify();
  }
  return internalState;
}

export function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function resetState(partial = {}) {
  const fresh = createInitialState();
  mutateState((current) => {
    Object.keys(current).forEach((key) => {
      delete current[key];
    });
    Object.assign(current, fresh, partial);
  });
}

export function disposeTrainingData() {
  mutateState((current) => {
    for (let i = 0; i < current.trainingDataInputs.length; i++) {
      const input = current.trainingDataInputs[i];
      if (input && typeof input.dispose === 'function') {
        input.dispose();
      }
    }
    current.trainingDataInputs.length = 0;
    current.trainingDataOutputs.length = 0;
    current.gestureSamples.length = 0;
    current.gestureLastSampleTs = 0;
    current.poseSamples.length = 0;
    current.poseLastSampleTs = 0;
  });
}

export const state = internalState;
