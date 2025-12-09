import { GESTURE_FEATURE_LENGTH, HAND_CONNECTIONS } from '../constants.js';
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

export function normalizeGestureLandmarks(landmarks = []) {
  if (!Array.isArray(landmarks) || landmarks.length === 0) return null;
  const wrist = landmarks[0];
  if (!wrist) return null;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  landmarks.forEach((pt) => {
    if (!pt) return;
    minX = Math.min(minX, pt.x);
    minY = Math.min(minY, pt.y);
    maxX = Math.max(maxX, pt.x);
    maxY = Math.max(maxY, pt.y);
  });

  const scale = Math.max(0.0001, Math.max(maxX - minX, maxY - minY)); // avoid div by zero
  const features = [];

  for (let i = 0; i < GESTURE_FEATURE_LENGTH / 3; i++) {
    const point = landmarks[i] || wrist;
    const x = (point?.x ?? wrist.x ?? 0) - wrist.x;
    const y = (point?.y ?? wrist.y ?? 0) - wrist.y;
    const z = (point?.z ?? wrist.z ?? 0) - (wrist.z ?? 0);
    features.push(x / scale, y / scale, z / scale);
  }

  return features;
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
    const landmarks = result?.landmarks?.[0];
    if (!landmarks) {
      renderProbabilities([], -1, state.classNames);
      clearOverlay();
      return;
    }

    drawHandOverlay(landmarks);

    if (!state.trainingCompleted || !state.model) {
      renderProbabilities([], -1, state.classNames);
      return;
    }

    const featureVector = normalizeGestureLandmarks(landmarks);
    if (!featureVector) {
      renderProbabilities([], -1, state.classNames);
      return;
    }

    tf.tidy(() => {
      const input = tf.tensor2d([featureVector]);
      const prediction = state.model.predict(input).squeeze();
      const predictionArray = Array.from(prediction.dataSync());
      const bestIndex =
        predictionArray.length > 0
          ? predictionArray.reduce(
              (bestIdx, value, idx, arr) => (value > arr[bestIdx] ? idx : bestIdx),
              0
            )
          : -1;
      renderProbabilities(predictionArray, bestIndex, state.classNames);
    });
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
