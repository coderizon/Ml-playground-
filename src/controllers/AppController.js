import {
  TRAIN_BUTTON,
  RESET_BUTTON,
  modeLabel,
  modeMenu,
  addClassButton,
  mobileStepButtons,
  epochsInput,
  batchSizeInput,
  learningRateInput,
} from '../domRefs.js';
import { getState, setState } from '../state.js';
import { renderProbabilities } from '../ui/probabilities.js';
import { initializeExistingClasses, updateExampleCounts } from '../ui/classes.js';
import { trainAndPredict, resetTrainingProgress, handleCollectStart, handleCollectEnd } from '../ml/training.js';
import { setMobileStep } from '../ui/steps.js';
import { toggleModeMenu, closeModeMenu } from '../ui/menu.js';
import { updateSwitchButtonsLabel } from '../camera/webcam.js';
import { initModeController, setMode, resetApp, openCaptureForClass, closeCapturePanel, handleSwitchCamera, addClassAndReset } from './ModeController.js';
import {
  connectArduino,
  isArduinoConnected,
  setArduinoConnectionListener,
} from '../bluetooth/arduino.js';
import {
  connectMicrobit,
  isMicrobitConnected,
  setMicrobitConnectionListener,
} from '../bluetooth/microbit.js';
import {
  connectCalliope,
  isCalliopeConnected,
  setCalliopeConnectionListener,
} from '../bluetooth/calliope.js';
import { DEFAULT_TRAINING_BATCH_SIZE, DEFAULT_TRAINING_EPOCHS, DEFAULT_TRAINING_LR } from '../constants.js';
import {
  bleArduinoCard,
  bleCalliopeCard,
  bleConnectButton,
  bleMicrobitCard,
  bleModal,
  bleModalBackdrop,
  bleModalClose,
} from '../domRefs.js';

const state = getState();

export function initAppController() {
  const classCardHandlers = buildClassCardHandlers();
  initModeController(classCardHandlers);

  bindCoreEvents();
  initializeExistingClasses(classCardHandlers);
  resetTrainingProgress();
  updateExampleCounts(true);
  renderProbabilities([], -1, state.classNames);
  setMode('image').catch(console.error);
  updateSwitchButtonsLabel();
  initHyperparamInputs();
  initBleModal();
}

function bindCoreEvents() {
  if (TRAIN_BUTTON) {
    TRAIN_BUTTON.addEventListener('click', trainAndPredict);
  }
  if (RESET_BUTTON) {
    RESET_BUTTON.addEventListener('click', resetApp);
  }

  mobileStepButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      setMobileStep(btn.getAttribute('data-step-target'));
    });
  });

  if (addClassButton) {
    addClassButton.addEventListener('click', addClassAndReset);
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
      const target = event.target.closest('[data-mode]');
      const mode = target?.getAttribute('data-mode');
      if (mode) {
        setMode(mode).catch(console.error);
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
}

function buildClassCardHandlers() {
  return {
    onNameChange: () => renderProbabilities(state.lastPrediction, -1, state.classNames),
    onOpenWebcam: (idx) => openCaptureForClass(idx),
    onCollectStart: handleCollectStart,
    onCollectEnd: handleCollectEnd,
    onSwitchCamera: () => handleSwitchCamera(),
    onClosePanel: (idx) => closeCapturePanel(idx),
  };
}

function initHyperparamInputs() {
  bindHyperparamInput(epochsInput, 'trainingEpochs', {
    fallback: DEFAULT_TRAINING_EPOCHS,
    parse: (value) => parseInt(value, 10),
    min: 1,
  });
  bindHyperparamInput(batchSizeInput, 'trainingBatchSize', {
    fallback: DEFAULT_TRAINING_BATCH_SIZE,
    parse: (value) => parseInt(value, 10),
    min: 1,
  });
  bindHyperparamInput(learningRateInput, 'trainingLearningRate', {
    fallback: DEFAULT_TRAINING_LR,
    parse: (value) => parseFloat(value),
    min: 0.000001,
  });
}

function bindHyperparamInput(element, stateKey, { fallback, parse, min }) {
  if (!element) return;

  const applyValue = () => {
    const parsed = parse(element.value);
    const valid = Number.isFinite(parsed) && parsed >= min ? parsed : fallback;
    setState({ [stateKey]: valid });
    element.value = valid;
  };

  element.value = state[stateKey] ?? fallback;
  applyValue();
  element.addEventListener('change', applyValue);
  element.addEventListener('blur', applyValue);
}

function initBleModal() {
  if (!bleConnectButton || !bleModal) return;

  const closeModal = () => {
    bleModal.classList.add('hidden');
    bleModalBackdrop?.classList.add('hidden');
    document.body.classList.remove('modal-open');
  };

  const openModal = () => {
    bleModal.classList.remove('hidden');
    bleModalBackdrop?.classList.remove('hidden');
    document.body.classList.add('modal-open');
  };

  bleConnectButton.addEventListener('click', openModal);
  bleModalBackdrop?.addEventListener('click', closeModal);
  bleModalClose?.addEventListener('click', closeModal);
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !bleModal.classList.contains('hidden')) {
      closeModal();
    }
  });

  const devices = [
    {
      button: bleArduinoCard,
      stateKey: 'arduinoConnected',
      setListener: setArduinoConnectionListener,
      isConnected: isArduinoConnected,
      connect: connectArduino,
      label: 'Arduino Uno R4',
    },
    {
      button: bleCalliopeCard,
      stateKey: 'calliopeConnected',
      setListener: setCalliopeConnectionListener,
      isConnected: isCalliopeConnected,
      connect: connectCalliope,
      label: 'Calliope Mini',
    },
    {
      button: bleMicrobitCard,
      stateKey: 'microbitConnected',
      setListener: setMicrobitConnectionListener,
      isConnected: isMicrobitConnected,
      connect: connectMicrobit,
      label: 'Micro:bit',
    },
  ];

  devices.forEach((device) => setupBleDeviceCard(device, closeModal));
}

function setupBleDeviceCard(
  { button, stateKey, setListener, isConnected, connect, label },
  closeModal
) {
  if (!button) return;
  const statusEl = button.querySelector('.ble-device-status');

  const setStatus = (text) => {
    if (statusEl) statusEl.textContent = text;
  };

  const updateState = (connected) => {
    setState({ [stateKey]: connected });
    button.classList.toggle('is-connected', connected);
    button.disabled = connected;
    setStatus(connected ? `${label} verbunden` : 'Bereit');
    if (connected) closeModal();
  };

  setListener((connected) => {
    updateState(connected);
  });

  button.addEventListener('click', async () => {
    if (getState()[stateKey] || isConnected()) {
      updateState(true);
      return;
    }
    setStatus('Verbinde...');
    button.disabled = true;
    try {
      await connect();
    } catch (error) {
      console.error(error);
      setStatus('Verbindung fehlgeschlagen');
      button.disabled = false;
    }
  });
}
