import { GESTURE_LABELS, HAND_CONNECTIONS } from '../constants.js';
import { GESTURE_OVERLAY, PREVIEW_VIDEO, STATUS } from '../domRefs.js';
import { state } from '../state.js';
import { renderProbabilities } from '../ui/probabilities.js';

export async function ensureGestureRecognizer() {
  if (state.gestureRecognizer) return state.gestureRecognizer;
  if (state.gestureInitPromise) return state.gestureInitPromise;

  state.gestureInitPromise = (async () => {
    try {
      const vision = await import(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0'
      );
      const fileset = await vision.FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm'
      );
      state.gestureRecognizer = await vision.GestureRecognizer.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task',
        },
        runningMode: 'VIDEO',
        numHands: 1,
      });
      if (GESTURE_OVERLAY) {
        const ctx = GESTURE_OVERLAY.getContext('2d');
        state.drawingUtils = new vision.DrawingUtils(ctx);
      }
      state.gestureConnections = vision.GestureRecognizer.HAND_CONNECTIONS || HAND_CONNECTIONS;
      if (STATUS) {
        STATUS.innerText = 'Gesture Recognition bereit.';
      }
      return state.gestureRecognizer;
    } catch (err) {
      console.error(err);
      if (STATUS) {
        STATUS.innerText = 'Gesture Recognition konnte nicht geladen werden.';
      }
      state.gestureRecognizer = null;
      return null;
    }
  })();

  return state.gestureInitPromise;
}

export async function runGestureStep() {
  if (state.gestureBusy) return;
  if (!state.previewReady) return;
  if (!GESTURE_OVERLAY) return;
  state.gestureBusy = true;
  try {
    const recognizer = await ensureGestureRecognizer();
    if (!recognizer) return;
    const nowInMs = performance.now();
    const result = recognizer.recognizeForVideo(PREVIEW_VIDEO, nowInMs);
    if (!result || !result.gestures || !result.gestures.length) {
      renderProbabilities([], -1, GESTURE_LABELS);
      clearOverlay();
      return;
    }
    const categories = result.gestures[0] || [];
    const probs = GESTURE_LABELS.map((name) => {
      const match = categories.find((c) => c.categoryName === name);
      return match ? match.score : 0;
    });
    const topIndex =
      probs.length > 0
        ? probs.reduce((best, val, idx, arr) => (val > arr[best] ? idx : best), 0)
        : -1;
    renderProbabilities(probs, topIndex, GESTURE_LABELS);

    if (result.landmarks && result.landmarks.length) {
      drawHandOverlay(result.landmarks[0]);
    } else {
      clearOverlay();
    }
  } catch (err) {
    console.error(err);
  } finally {
    state.gestureBusy = false;
  }
}

export function clearOverlay() {
  if (!GESTURE_OVERLAY) return;
  const ctx = GESTURE_OVERLAY.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, GESTURE_OVERLAY.width, GESTURE_OVERLAY.height);
}

export function resizeOverlay() {
  if (!GESTURE_OVERLAY || !PREVIEW_VIDEO) return;
  const w = PREVIEW_VIDEO.videoWidth;
  const h = PREVIEW_VIDEO.videoHeight;
  if (!w || !h) return;
  if (GESTURE_OVERLAY.width !== w || GESTURE_OVERLAY.height !== h) {
    GESTURE_OVERLAY.width = w;
    GESTURE_OVERLAY.height = h;
  }
}

export function drawHandOverlay(landmarks = []) {
  if (!GESTURE_OVERLAY || !PREVIEW_VIDEO) return;
  resizeOverlay();
  const ctx = GESTURE_OVERLAY.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, GESTURE_OVERLAY.width, GESTURE_OVERLAY.height);

  const connections = state.gestureConnections || HAND_CONNECTIONS;

  if (state.drawingUtils && connections) {
    state.drawingUtils.drawConnectors(landmarks, connections, { color: '#28b88a', lineWidth: 5 });
    state.drawingUtils.drawLandmarks(landmarks, { color: '#ff3366', lineWidth: 3, radius: 6 });
    return;
  }

  const w = GESTURE_OVERLAY.width;
  const h = GESTURE_OVERLAY.height;

  ctx.strokeStyle = '#28b88a';
  ctx.lineWidth = 5;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';

  connections.forEach(([a, b]) => {
    if (!landmarks[a] || !landmarks[b]) return;
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
    ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
    ctx.stroke();
  });

  ctx.fillStyle = '#ff3366';
  landmarks.forEach((point) => {
    if (!point) return;
    ctx.beginPath();
    ctx.arc(point.x * w, point.y * h, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 3;
    ctx.stroke();
  });
}
