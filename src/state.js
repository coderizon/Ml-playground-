import { STOP_DATA_GATHER } from './constants.js';

export const state = {
  currentStream: null,
  activeClassIndex: 0,
  previewReady: false,
  trainingCompleted: false,
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
  gestureRecognizer: null,
  gestureInitPromise: null,
  gestureBusy: false,
  drawingUtils: null,
  gestureConnections: null,
  arduinoConnected: false,
  microbitConnected: false,
  calliopeConnected: false,
  trainingEpochs: 10,
  trainingBatchSize: 5,
  trainingLearningRate: 0.001,
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

export function disposeTrainingData() {
  for (let i = 0; i < state.trainingDataInputs.length; i++) {
    state.trainingDataInputs[i].dispose();
  }
  state.trainingDataInputs.length = 0;
  state.trainingDataOutputs.length = 0;
}
