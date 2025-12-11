import {
  AUDIO_CAPTURE_LABEL,
  AUDIO_COLLECT_LABEL,
  DEFAULT_CAPTURE_LABEL,
  DEFAULT_COLLECT_LABEL,
  MODE_NAMES,
  POSE_CAPTURE_LABEL,
  STOP_DATA_GATHER,
  SUPPORTED_MODES,
} from '../constants.js';
import {
  CAPTURE_VIDEO,
  GESTURE_OVERLAY,
  PREVIEW_VIDEO,
  STATUS,
  TRAIN_BUTTON,
  modeLabel,
} from '../domRefs.js';
import { clearOverlay } from '../ml/overlay.js';
import { ensureHandLandmarker, runGestureLoop, stopGestureLoop, resetGestureSamples } from '../ml/gesture.js';
import { ensureFaceLandmarker, runFaceLoop, stopFaceLoop } from '../ml/face.js';
import { loadMobileNetFeatureModel, rebuildModel } from '../ml/model.js';
import { disposeTrainingData, getState, mutateState, setState } from '../state.js';
import {
  resetClassCards,
  addNewClassCard,
  unlockCapturePanels,
  updateExampleCounts,
  updateClassCardCopy,
} from '../ui/classes.js';
import { renderProbabilities } from '../ui/probabilities.js';
import { updateModeMenuActive } from '../ui/menu.js';
import { setMobileStep } from '../ui/steps.js';
import {
  enableCam,
  hideWebcamPanel,
  moveCaptureToSlot,
  toggleCameraFacing,
  updateSwitchButtonsLabel,
} from '../camera/webcam.js';
import { resetTrainingProgress } from '../ml/training.js';
import { ensurePoseLandmarker, resetPoseSamples, runPoseLoop, stopPoseLoop } from '../ml/pose.js';
import { clearAudioExamples, initAudio, stopAudioListening } from '../ml/audio.js';

const state = getState();
const defaultTrainLabel = TRAIN_BUTTON ? TRAIN_BUTTON.textContent : 'Modell trainieren';

let classCardHandlers = null;

export function initModeController(handlers) {
  classCardHandlers = handlers;
  window.setAppMode = (mode) => setMode(mode).catch(console.error);
}

export async function setMode(newMode) {
  if (!SUPPORTED_MODES.includes(newMode)) return;
  if (newMode === state.currentMode) {
    updateModeMenuActive();
    return;
  }

  await teardownCurrentMode();
  resetSharedState();
  setTrainButtonEnabled(true, defaultTrainLabel);

  setState({
    currentMode: newMode,
  });

  if (modeLabel) {
    modeLabel.textContent = MODE_NAMES[newMode] || newMode;
  }
  document.body.setAttribute('data-mode', newMode);

  PREVIEW_VIDEO.classList.add('hidden');
  clearOverlay();
  unlockCapturePanels();
  resetClassCards(classCardHandlers);
  updateClassCopyForMode(newMode);
  renderProbabilities([], -1, newMode === 'face' ? [] : state.classNames);

  if (newMode === 'gesture') {
    await activateGestureMode();
  } else if (newMode === 'face') {
    await activateFaceMode();
  } else if (newMode === 'audio') {
    await activateAudioMode();
  } else if (newMode === 'pose') {
    await activatePoseMode();
  } else {
    await activateImageMode();
  }

  updateModeMenuActive();
  updateSwitchButtonsLabel();
}

export function resetApp() {
  stopGestureLoop();
  stopFaceLoop();
  stopPoseLoop();
  stopAudioListening().catch(console.error);
  setState({
    predict: false,
    previewReady: false,
    trainingCompleted: false,
    trainingInProgress: false,
    gatherDataState: STOP_DATA_GATHER,
    lastPrediction: [],
  });
  disposeTrainingData();
  resetGestureSamples();
  resetPoseSamples();
  clearAudioExamples();
  mutateState((draft) => {
    draft.examplesCount.length = 0;
    draft.isAudioListening = false;
  });
  resetTrainingProgress();
  updateExampleCounts(true);
  unlockCapturePanels();
  renderProbabilities([], -1, state.currentMode === 'face' ? [] : state.classNames);
  updateClassCopyForMode(state.currentMode);

  if (state.currentMode === 'image') {
    rebuildModel();
    setTrainButtonEnabled(true);
    if (GESTURE_OVERLAY) {
      GESTURE_OVERLAY.classList.add('hidden');
    }
  } else if (state.currentMode === 'gesture') {
    if (state.model) {
      state.model.dispose();
      setState({ model: null });
    }
    setTrainButtonEnabled(true);
    if (GESTURE_OVERLAY) {
      GESTURE_OVERLAY.classList.remove('hidden');
      clearOverlay();
    }
  } else if (state.currentMode === 'pose') {
    if (state.model) {
      state.model.dispose();
      setState({ model: null });
    }
    if (GESTURE_OVERLAY) {
      GESTURE_OVERLAY.classList.remove('hidden');
      clearOverlay();
    }
    setTrainButtonEnabled(true);
  } else if (state.currentMode === 'face') {
    if (GESTURE_OVERLAY) {
      GESTURE_OVERLAY.classList.remove('hidden');
      clearOverlay();
    }
    PREVIEW_VIDEO.classList.remove('hidden');
    setState({ predict: true });
    preparePreviewForFace();
    runFaceLoop();
    if (STATUS) {
      STATUS.innerText = 'Gesichtserkennung aktiv.';
    }
  } else if (state.currentMode === 'audio') {
    setTrainButtonEnabled(true);
    PREVIEW_VIDEO.classList.add('hidden');
    if (STATUS) {
      STATUS.innerText = 'Audio-Modus zurückgesetzt. Sammle neue Aufnahmen.';
    }
  }
  setState({ lastPrediction: [] });
  if (state.currentMode === 'face') {
    renderProbabilities([], -1, []);
    setMobileStep('preview');
  } else {
    renderProbabilities([], -1, state.classNames);
    setMobileStep('collect');
  }
}

export function addClassAndReset() {
  resetTrainingProgress();
  setState({
    trainingCompleted: false,
    predict: false,
    previewReady: false,
  });
  addNewClassCard(classCardHandlers);
  PREVIEW_VIDEO.classList.add('hidden');
  if (state.currentMode === 'image') {
    rebuildModel();
  } else if (state.model) {
    state.model.dispose();
    setState({ model: null });
  }
  renderProbabilities([], -1, state.classNames);
  updateClassCopyForMode(state.currentMode);
  if (STATUS) {
    STATUS.innerText = `Neue Klasse ${state.classNames[state.classNames.length - 1]} hinzugefügt.`;
  }
  updateSwitchButtonsLabel();
}

export function openCaptureForClass(idx) {
  mutateState((draft) => {
    draft.activeClassIndex = idx;
  });
  state.webcamPanels.forEach((panel) => {
    panel.classList.toggle(
      'visible',
      parseInt(panel.getAttribute('data-class-panel'), 10) === idx
    );
  });

  if (state.currentMode !== 'audio') {
    moveCaptureToSlot(idx);
    enableCam();
    if (STATUS) {
      STATUS.innerText = `Webcam geöffnet für ${state.classNames[idx]}. Halte zum Aufnehmen.`;
    }
  } else {
    if (CAPTURE_VIDEO) {
      CAPTURE_VIDEO.classList.add('hidden');
    }
    if (STATUS) {
      STATUS.innerText = `Bereit für ${state.classNames[idx]}. Klicke zum Audio-Aufnehmen.`;
    }
  }
}

export function closeCapturePanel(idx) {
  setState({ gatherDataState: STOP_DATA_GATHER });
  hideWebcamPanel(idx);
}

export function handleSwitchCamera() {
  if (state.currentMode === 'audio') {
    if (STATUS) {
      STATUS.innerText = 'Im Audio-Modus ist keine Kamera aktiv.';
    }
    return;
  }
  const message = toggleCameraFacing();
  if (STATUS) {
    STATUS.innerText = message;
  }
}

function resetSharedState() {
  setState({
    predict: false,
    previewReady: false,
    trainingCompleted: false,
    trainingInProgress: false,
    gatherDataState: STOP_DATA_GATHER,
    lastPrediction: [],
  });
  disposeTrainingData();
  resetGestureSamples();
  resetPoseSamples();
  mutateState((draft) => {
    draft.examplesCount.length = 0;
  });
  resetTrainingProgress();
  if (state.model) {
    state.model.dispose();
    setState({ model: null });
  }
  updateExampleCounts(true);
}

async function teardownCurrentMode() {
  stopGestureLoop();
  stopFaceLoop();
  stopPoseLoop();
  try {
    await stopAudioListening();
  } catch (error) {
    console.error(error);
  }
}

async function activateGestureMode() {
  if (GESTURE_OVERLAY) {
    GESTURE_OVERLAY.classList.remove('hidden');
    clearOverlay();
  }
  setMobileStep('collect');
  await enableCam();
  const landmarker = await ensureHandLandmarker();
  if (state.currentMode === 'gesture' && landmarker) {
    runGestureLoop();
  }
  updateClassCardCopy({
    openButtonLabel: DEFAULT_CAPTURE_LABEL,
    panelLabel: DEFAULT_CAPTURE_LABEL,
    collectorLabel: DEFAULT_COLLECT_LABEL,
  });
  if (STATUS) {
    STATUS.innerText = 'Gestenerkennung aktiv. Sammle Daten und trainiere.';
  }
}

async function activatePoseMode() {
  if (GESTURE_OVERLAY) {
    GESTURE_OVERLAY.classList.remove('hidden');
    clearOverlay();
  }
  setMobileStep('collect');
  await enableCam();
  const landmarker = await ensurePoseLandmarker();
  if (state.currentMode === 'pose' && landmarker) {
    runPoseLoop();
  }
  updateClassCardCopy({
    openButtonLabel: DEFAULT_CAPTURE_LABEL,
    panelLabel: 'Kamera',
    collectorLabel: DEFAULT_COLLECT_LABEL,
  });
  if (STATUS) {
    STATUS.innerText = 'Posen-Erkennung aktiv. Sammle Daten und trainiere.';
  }
}

async function activateFaceMode() {
  if (GESTURE_OVERLAY) {
    GESTURE_OVERLAY.classList.remove('hidden');
    clearOverlay();
  }
  setMobileStep('preview');
  await enableCam();
  preparePreviewForFace();
  const landmarker = await ensureFaceLandmarker();
  if (state.currentMode === 'face' && landmarker) {
    setState({ predict: true });
    renderProbabilities([], -1, []);
    runFaceLoop();
    if (STATUS) {
      STATUS.innerText = 'Gesichtserkennung aktiv. Vorschau läuft.';
    }
  }
  updateClassCopyForMode('face');
}

async function activateAudioMode() {
  if (GESTURE_OVERLAY) {
    GESTURE_OVERLAY.classList.add('hidden');
    clearOverlay();
  }
  setMobileStep('collect');
  PREVIEW_VIDEO.classList.add('hidden');
  await initAudio();
  updateClassCopyForMode('audio');
  if (STATUS) {
    STATUS.innerText = 'Audioerkennung aktiv. Sammle Samples und trainiere.';
  }
}

async function activateImageMode() {
  if (GESTURE_OVERLAY) {
    GESTURE_OVERLAY.classList.add('hidden');
  }
  setMobileStep('collect');
  rebuildModel();
  updateClassCopyForMode('image');
  if (!state.mobilenet) {
    await loadMobileNetFeatureModel();
  }
  if (STATUS) {
    STATUS.innerText = 'Bildklassifikation aktiv. Sammle Daten und trainiere.';
  }
}

function updateClassCopyForMode(mode) {
  if (mode === 'audio') {
    updateClassCardCopy({
      openButtonLabel: AUDIO_CAPTURE_LABEL,
      panelLabel: AUDIO_CAPTURE_LABEL,
      collectorLabel: AUDIO_COLLECT_LABEL,
    });
    return;
  }
  const panelLabel = mode === 'pose' ? POSE_CAPTURE_LABEL : DEFAULT_CAPTURE_LABEL;
  updateClassCardCopy({
    openButtonLabel: DEFAULT_CAPTURE_LABEL,
    panelLabel,
    collectorLabel: DEFAULT_COLLECT_LABEL,
  });
}

function setTrainButtonEnabled(enabled, label = defaultTrainLabel) {
  if (!TRAIN_BUTTON) return;
  TRAIN_BUTTON.disabled = !enabled;
  TRAIN_BUTTON.textContent = label;
}

function preparePreviewForFace() {
  if (!PREVIEW_VIDEO) return;
  PREVIEW_VIDEO.classList.remove('hidden');
  const hasFrame = () =>
    PREVIEW_VIDEO.readyState >= 2 &&
    PREVIEW_VIDEO.videoWidth > 0 &&
    PREVIEW_VIDEO.videoHeight > 0;

  const markReady = () => {
    if (!hasFrame()) return;
    setState({ previewReady: true });
    PREVIEW_VIDEO.removeEventListener('loadeddata', markReady);
    PREVIEW_VIDEO.removeEventListener('canplay', markReady);
    PREVIEW_VIDEO.removeEventListener('playing', markReady);
  };

  if (hasFrame()) {
    markReady();
  } else {
    PREVIEW_VIDEO.addEventListener('loadeddata', markReady);
    PREVIEW_VIDEO.addEventListener('canplay', markReady);
    PREVIEW_VIDEO.addEventListener('playing', markReady);
  }
}
