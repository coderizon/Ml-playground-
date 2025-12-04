import {
  connectArduino,
  isArduinoConnected,
  setArduinoConnectionListener,
} from './bluetooth/arduino.js';
import {
  GESTURE_LABELS,
  MODE_NAMES,
  STOP_DATA_GATHER,
} from './constants.js';
import {
  GESTURE_OVERLAY,
  PREVIEW_VIDEO,
  RESET_BUTTON,
  STATUS,
  TRAIN_BUTTON,
  addClassButton,
  connectArduinoButton,
  menuButton,
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

if (menuButton) {
  menuButton.addEventListener('click', toggleModeMenu);
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
  if (!modeMenu || !menuButton) return;
  if (!modeMenu.contains(event.target) && !menuButton.contains(event.target)) {
    closeModeMenu();
  }
});

if (connectArduinoButton) {
  setArduinoConnectionListener((connected) => {
    state.arduinoConnected = connected;
    connectArduinoButton.disabled = false;
    connectArduinoButton.textContent = connected ? 'Arduino verbunden' : 'Arduino verbinden';
    connectArduinoButton.classList.toggle('primary', connected);
    connectArduinoButton.classList.toggle('ghost', !connected);
  });

  connectArduinoButton.addEventListener('click', async () => {
    if (state.arduinoConnected || isArduinoConnected()) {
      alert('Arduino ist bereits verbunden.');
      return;
    }
    connectArduinoButton.disabled = true;
    connectArduinoButton.textContent = 'Verbinde...';
    try {
      await connectArduino();
    } catch (error) {
      console.error(error);
    } finally {
      if (!state.arduinoConnected) {
        connectArduinoButton.disabled = false;
        connectArduinoButton.textContent = 'Arduino verbinden';
      }
    }
  });
}

function addClassAndReset() {
  addNewClassCard(classCardHandlers);
  state.trainingCompleted = false;
  state.predict = false;
  state.previewReady = false;
  PREVIEW_VIDEO.classList.add('hidden');
  rebuildModel();
  renderProbabilities([], -1, state.classNames);
  if (STATUS) {
    STATUS.innerText = `Neue Klasse ${state.classNames[state.classNames.length - 1]} hinzugefügt.`;
  }
  updateSwitchButtonsLabel();
}

function setMode(newMode) {
  if (newMode !== 'image' && newMode !== 'gesture') return;
  if (newMode === state.currentMode) {
    updateModeMenuActive();
    return;
  }
  state.currentMode = newMode;
  if (modeLabel) {
    modeLabel.textContent = MODE_NAMES[newMode] || newMode;
  }
  document.body.setAttribute('data-mode', newMode);

  if (newMode === 'gesture') {
    state.predict = true;
    state.previewReady = false;
    state.trainingCompleted = false;
    if (STATUS) {
      STATUS.innerText = 'Gesture Recognition wird geladen...';
    }
    setMobileStep('preview');
    showPreview();
    enableCam();
    renderProbabilities([], -1, GESTURE_LABELS);
    if (GESTURE_OVERLAY) {
      GESTURE_OVERLAY.classList.remove('hidden');
      clearOverlay();
    }
    window.requestAnimationFrame(predictLoop);
  } else {
    state.predict = false;
    state.previewReady = false;
    state.trainingCompleted = false;
    if (STATUS) {
      STATUS.innerText = 'Bildklassifikation aktiv. Sammle Daten und trainiere.';
    }
    PREVIEW_VIDEO.classList.add('hidden');
    state.lastPrediction = [];
    renderProbabilities([], -1, state.classNames);
    clearOverlay();
    if (GESTURE_OVERLAY) {
      GESTURE_OVERLAY.classList.add('hidden');
    }
    setMobileStep('collect');
  }

  updateModeMenuActive();
}

function resetApp() {
  state.predict = false;
  state.previewReady = false;
  state.trainingCompleted = false;
  state.gatherDataState = STOP_DATA_GATHER;
  state.examplesCount.length = 0;
  disposeTrainingData();
  if (STATUS) {
    STATUS.innerText = 'No data collected';
  }
  PREVIEW_VIDEO.classList.add('hidden');
  unlockCapturePanels();
  rebuildModel();
  updateExampleCounts(true);
  state.lastPrediction = [];
  renderProbabilities([], -1, state.classNames);
  setMobileStep('collect');

  console.log('Tensors in memory: ' + tf.memory().numTensors);
}
