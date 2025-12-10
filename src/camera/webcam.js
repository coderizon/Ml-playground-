import { STOP_DATA_GATHER } from '../constants.js';
import { getState, mutateState, setState } from '../state.js';
import { CAPTURE_VIDEO, PREVIEW_VIDEO, STATUS } from '../domRefs.js';

export const captureCanvas = document.createElement('canvas');
captureCanvas.className = 'overlay-canvas';

export function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

export function stopCurrentStream() {
  const current = getState().currentStream;
  if (!current) return;
  current.getTracks().forEach((track) => track.stop());
  mutateState((draft) => {
    draft.currentStream = null;
    draft.videoPlaying = false;
  });
}

function attachStreamToVideos(stream) {
  CAPTURE_VIDEO.srcObject = stream;
  PREVIEW_VIDEO.srcObject = stream;
  CAPTURE_VIDEO.addEventListener('loadeddata', function onLoad() {
    mutateState((draft) => {
      draft.videoPlaying = true;
    });
    CAPTURE_VIDEO.classList.remove('hidden');
    CAPTURE_VIDEO.removeEventListener('loadeddata', onLoad);
  });
}

export function updateSwitchButtonsLabel() {
  const { preferredFacingMode, switchCameraButtons } = getState();
  const targetLabel = preferredFacingMode === 'user' ? 'Außenkamera' : 'Selfie-Kamera';
  switchCameraButtons.forEach((btn) => {
    if (btn) btn.textContent = targetLabel;
  });
}

export async function enableCam(allowFallback = true) {
  if (!hasGetUserMedia()) {
    console.warn('getUserMedia() is not supported by your browser');
    return;
  }

  const { currentStream, preferredFacingMode } = getState();
  const facingConstraint =
    preferredFacingMode === 'environment' ? { exact: 'environment' } : 'user';
  const constraints = {
    video: {
      width: 640,
      height: 480,
      aspectRatio: { ideal: 4 / 3 },
      facingMode: facingConstraint,
    },
  };

  if (currentStream) {
    attachStreamToVideos(currentStream);
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    mutateState((draft) => {
      draft.currentStream = stream;
    });
    attachStreamToVideos(stream);
  } catch (err) {
    console.error(err);
    if (allowFallback && preferredFacingMode === 'environment') {
      mutateState((draft) => {
        draft.preferredFacingMode = 'user';
      });
      updateSwitchButtonsLabel();
      if (STATUS) {
        STATUS.innerText = 'Außenkamera nicht verfügbar, Selfie-Kamera aktiviert.';
      }
      return enableCam(false);
    }
    if (STATUS) {
      STATUS.innerText = 'Webcam konnte nicht gestartet werden.';
    }
  }
}

export function moveCaptureToSlot(idx) {
  const slot = getState().captureSlots.find(
    (s) => parseInt(s.getAttribute('data-class-slot'), 10) === idx
  );
  if (!slot) return;

  if (CAPTURE_VIDEO.parentElement !== slot) {
    slot.innerHTML = '';
    slot.appendChild(CAPTURE_VIDEO);
  }

  if (captureCanvas.parentElement !== slot) {
    slot.appendChild(captureCanvas);
  }
}

export function openWebcamForClass(idx) {
  const appState = getState();
  mutateState((draft) => {
    draft.activeClassIndex = idx;
  });
  appState.webcamPanels.forEach((panel) => {
    panel.classList.toggle(
      'visible',
      parseInt(panel.getAttribute('data-class-panel'), 10) === idx
    );
  });
  moveCaptureToSlot(idx);
  enableCam();
  if (STATUS) {
    STATUS.innerText = `Webcam geöffnet für ${appState.classNames[idx]}. Halte zum Aufnehmen.`;
  }
}

export function hideWebcamPanel(idx) {
  if (getState().gatherDataState !== STOP_DATA_GATHER) {
    setState({ gatherDataState: STOP_DATA_GATHER });
  }
  const panel = getState().webcamPanels.find(
    (p) => parseInt(p.getAttribute('data-class-panel'), 10) === idx
  );
  if (panel) {
    panel.classList.remove('visible');
  }
}

export function toggleCameraFacing() {
  mutateState((draft) => {
    draft.preferredFacingMode = draft.preferredFacingMode === 'user' ? 'environment' : 'user';
  });
  stopCurrentStream();
  moveCaptureToSlot(getState().activeClassIndex);
  enableCam();
  updateSwitchButtonsLabel();
  return getState().preferredFacingMode === 'environment'
    ? 'Außenkamera aktiviert.'
    : 'Selfie-Kamera aktiviert.';
}
