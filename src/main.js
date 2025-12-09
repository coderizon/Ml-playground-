import {
  connectArduino,
  isArduinoConnected,
  setArduinoConnectionListener,
} from './bluetooth/arduino.js';
import {
  connectMicrobit,
  isMicrobitConnected,
  setMicrobitConnectionListener,
} from './bluetooth/microbit.js';
import {
  connectCalliope,
  isCalliopeConnected,
  setCalliopeConnectionListener,
} from './bluetooth/calliope.js';
import { MODE_NAMES, STOP_DATA_GATHER } from './constants.js';
import {
  GESTURE_OVERLAY,
  PREVIEW_VIDEO,
  RESET_BUTTON,
  STATUS,
  TRAIN_BUTTON,
  addClassButton,
  connectArduinoButton,
  connectMicrobitButton,
  connectCalliopeButton,
  epochsInput,
  batchSizeInput,
  learningRateInput,
  modeLabel,
  modeMenu,
  mobileStepButtons,
} from './domRefs.js';
import { clearOverlay } from './ml/gesture.js';
import { loadMobileNetFeatureModel, rebuildModel } from './ml/model.js';
import {
  handleCollectEnd,
  handleCollectStart,
  predictLoop,
  showPreview,
  resetTrainingProgress,
  trainAndPredict,
} from './ml/training.js';
import { state, disposeTrainingData } from './state.js';
import {
  enableCam,
  hideWebcamPanel,
  openWebcamForClass,
  toggleCameraFacing,
  updateSwitchButtonsLabel,
} from './camera/webcam.js';
import { renderProbabilities } from './ui/probabilities.js';
import { initializeExistingClasses, addNewClassCard, updateExampleCounts, unlockCapturePanels } from './ui/classes.js';
import { toggleModeMenu, closeModeMenu, updateModeMenuActive } from './ui/menu.js';
import { setMobileStep } from './ui/steps.js';

const classCardHandlers = {
  onNameChange: () => renderProbabilities(state.lastPrediction),
  onOpenWebcam: (idx) => openWebcamForClass(idx),
  onCollectStart: handleCollectStart,
  onCollectEnd: handleCollectEnd,
  onSwitchCamera: () => {
    const message = toggleCameraFacing();
    if (STATUS) {
      STATUS.innerText = message;
    }
  },
  onClosePanel: (idx) => hideWebcamPanel(idx),
};

TRAIN_BUTTON.addEventListener('click', trainAndPredict);
RESET_BUTTON.addEventListener('click', resetApp);
const defaultTrainLabel = TRAIN_BUTTON ? TRAIN_BUTTON.textContent : 'Modell trainieren';

mobileStepButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    setMobileStep(btn.getAttribute('data-step-target'));
  });
});

initializeExistingClasses(classCardHandlers);
rebuildModel();
updateExampleCounts(true);
renderProbabilities([], -1, state.classNames);
setMode('image');
updateSwitchButtonsLabel();
loadMobileNetFeatureModel();
initHyperparamInputs();

if (addClassButton) {
  addClassButton.addEventListener('click', () => {
    addClassAndReset();
  });
  addClassButton.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      addClassAndReset();
    }
  });
}

if (modeLabel) {
  modeLabel.addEventListener('click', toggleModeMenu);
}

if (modeMenu) {
  modeMenu.addEventListener('click', (event) => {
    const mode = event.target.getAttribute('data-mode');
    if (mode) {
      setMode(mode);
      closeModeMenu();
    }
  });
}

document.addEventListener('click', (event) => {
  if (!modeMenu || !modeLabel) return;
  if (!modeMenu.contains(event.target) && !modeLabel.contains(event.target)) {
    closeModeMenu();
  }
});

setupBluetoothButton({
  button: connectArduinoButton,
  stateKey: 'arduinoConnected',
  setListener: setArduinoConnectionListener,
  isConnected: isArduinoConnected,
  connect: connectArduino,
  labels: {
    connected: 'Arduino verbunden',
    disconnected: 'Arduino verbinden',
    already: 'Arduino ist bereits verbunden.',
  },
});

setupBluetoothButton({
  button: connectMicrobitButton,
  stateKey: 'microbitConnected',
  setListener: setMicrobitConnectionListener,
  isConnected: isMicrobitConnected,
  connect: connectMicrobit,
  labels: {
    connected: 'Micro:bit verbunden',
    disconnected: 'Micro:bit verbinden',
    already: 'Micro:bit ist bereits verbunden.',
  },
});

setupBluetoothButton({
  button: connectCalliopeButton,
  stateKey: 'calliopeConnected',
  setListener: setCalliopeConnectionListener,
  isConnected: isCalliopeConnected,
  connect: connectCalliope,
  labels: {
    connected: 'Calliope verbunden',
    disconnected: 'Calliope verbinden',
    already: 'Calliope ist bereits verbunden.',
  },
});

function initHyperparamInputs() {
  bindHyperparamInput(epochsInput, 'trainingEpochs', {
    fallback: 10,
    parse: (value) => parseInt(value, 10),
    min: 1,
  });
  bindHyperparamInput(batchSizeInput, 'trainingBatchSize', {
    fallback: 5,
    parse: (value) => parseInt(value, 10),
    min: 1,
  });
  bindHyperparamInput(learningRateInput, 'trainingLearningRate', {
    fallback: 0.001,
    parse: (value) => parseFloat(value),
    min: 0.000001,
  });
}

function bindHyperparamInput(element, stateKey, { fallback, parse, min }) {
  if (!element) return;

  const applyValue = () => {
    const parsed = parse(element.value);
    const valid = Number.isFinite(parsed) && parsed >= min ? parsed : fallback;
    state[stateKey] = valid;
    element.value = valid;
  };

  element.value = state[stateKey] ?? fallback;
  applyValue();
  element.addEventListener('change', applyValue);
  element.addEventListener('blur', applyValue);
}

function setupBluetoothButton({
  button,
  stateKey,
  setListener,
  isConnected,
  connect,
  labels,
}) {
  if (!button) return;

  setListener((connected) => {
    state[stateKey] = connected;
    button.disabled = false;
    button.textContent = connected ? labels.connected : labels.disconnected;
    button.classList.toggle('primary', connected);
    button.classList.toggle('ghost', !connected);
  });

  button.addEventListener('click', async () => {
    if (state[stateKey] || isConnected()) {
      alert(labels.already);
      return;
    }
    button.disabled = true;
    button.textContent = 'Verbinde...';
    try {
      await connect();
    } catch (error) {
      console.error(error);
    } finally {
      if (!state[stateKey]) {
        button.disabled = false;
        button.textContent = labels.disconnected;
      }
    }
  });
}

function addClassAndReset() {
  addNewClassCard(classCardHandlers);
  state.trainingCompleted = false;
  state.predict = false;
  state.previewReady = false;
  resetTrainingProgress();
  PREVIEW_VIDEO.classList.add('hidden');
  rebuildModel();
  renderProbabilities([], -1, state.classNames);
  if (STATUS) {
    STATUS.innerText = `Neue Klasse ${state.classNames[state.classNames.length - 1]} hinzugefügt.`;
  }
  updateSwitchButtonsLabel();
}

function setMode(newMode) {
  if (newMode !== 'image' && newMode !== 'gesture' && newMode !== 'face') return;
  if (newMode === state.currentMode) {
    updateModeMenuActive();
    return;
  }
  state.currentMode = newMode;
  if (modeLabel) {
    modeLabel.textContent = MODE_NAMES[newMode] || newMode;
  }
  document.body.setAttribute('data-mode', newMode);

  state.predict = false;
  state.previewReady = false;
  state.trainingCompleted = false;
  state.gatherDataState = STOP_DATA_GATHER;
  state.lastPrediction = [];
  disposeTrainingData();
  state.examplesCount.length = 0;
  resetTrainingProgress();
  updateExampleCounts(true);
  unlockCapturePanels();
  renderProbabilities([], -1, newMode === 'face' ? [] : state.classNames);
  PREVIEW_VIDEO.classList.add('hidden');
  clearOverlay();
  toggleCaptureControls(true);
  setTrainButtonState(true);

  if (newMode === 'gesture') {
    if (GESTURE_OVERLAY) {
      GESTURE_OVERLAY.classList.remove('hidden');
      clearOverlay();
    }
    setMobileStep('preview');
    showPreview();
    enableCam();
    if (STATUS) {
      STATUS.innerText = 'Gestenmodus aktiv. Sammle Gestenbeispiele und trainiere.';
    }
    state.predict = true;
    window.requestAnimationFrame(predictLoop);
  } else if (newMode === 'face') {
    renderProbabilities([], -1, []);
    if (GESTURE_OVERLAY) {
      GESTURE_OVERLAY.classList.remove('hidden');
      clearOverlay();
    }
    setMobileStep('preview');
    showPreview();
    enableCam();
    if (STATUS) {
      STATUS.innerText = 'Face Landmarker aktiv (nur Inferenz).';
    }
    toggleCaptureControls(false);
    setTrainButtonState(false, 'Nur Inferenz');
    state.predict = true;
    window.requestAnimationFrame(predictLoop);
  } else {
    if (STATUS) {
      STATUS.innerText = 'Bildklassifikation aktiv. Sammle Daten und trainiere.';
    }
    if (GESTURE_OVERLAY) {
      GESTURE_OVERLAY.classList.add('hidden');
    }
    setMobileStep('collect');
  }

  if (newMode !== 'face') {
    rebuildModel();
  } else if (state.model) {
    state.model.dispose();
    state.model = null;
  }

  updateModeMenuActive();
}

function resetApp() {
  state.predict = false;
  state.previewReady = false;
  state.trainingCompleted = false;
  state.gatherDataState = STOP_DATA_GATHER;
  state.examplesCount.length = 0;
  resetTrainingProgress();
  disposeTrainingData();
  if (STATUS) {
    STATUS.innerText = 'No data collected';
  }
  PREVIEW_VIDEO.classList.add('hidden');
  unlockCapturePanels();
  if (state.currentMode !== 'face') {
    rebuildModel();
    setTrainButtonState(true);
    toggleCaptureControls(true);
  } else {
    setTrainButtonState(false, 'Nur Inferenz');
    toggleCaptureControls(false);
    if (GESTURE_OVERLAY) {
      GESTURE_OVERLAY.classList.remove('hidden');
    }
  }
  updateExampleCounts(true);
  state.lastPrediction = [];
  renderProbabilities([], -1, state.currentMode === 'face' ? [] : state.classNames);
  setMobileStep(state.currentMode === 'face' ? 'preview' : 'collect');

  console.log('Tensors in memory: ' + tf.memory().numTensors);
}

window.setAppMode = setMode;

function toggleCaptureControls(enabled) {
  state.openWebcamButtons.forEach((btn) => {
    if (btn) btn.disabled = !enabled;
  });
  state.dataCollectorButtons.forEach((btn) => {
    if (btn) btn.disabled = !enabled;
  });
}

function setTrainButtonState(enabled, label = defaultTrainLabel) {
  if (!TRAIN_BUTTON) return;
  TRAIN_BUTTON.disabled = !enabled;
  TRAIN_BUTTON.textContent = label;
}
