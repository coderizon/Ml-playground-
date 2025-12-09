import { GESTURE_FEATURE_LENGTH, HAND_CONNECTIONS } from '../constants.js';
import { GESTURE_OVERLAY, PREVIEW_VIDEO, STATUS } from '../domRefs.js';
import { state } from '../state.js';
import { renderProbabilities } from '../ui/probabilities.js';

const oneEuroFilters = [];

export async function ensureGestureRecognizer() {
  if (state.gestureRecognizer) return state.gestureRecognizer;
  if (state.gestureInitPromise) return state.gestureInitPromise;

  state.gestureInitPromise = (async () => {
    try {
      const handPoseDetection = await loadHandPoseDetectionScript();
      state.gestureRecognizer = await handPoseDetection.createDetector(
        handPoseDetection.SupportedModels.MediaPipeHands,
        {
          runtime: 'tfjs',
          modelType: 'lite',
          maxHands: 1,
        }
      );
      state.gestureConnections = HAND_CONNECTIONS;
      if (STATUS) {
        STATUS.innerText = 'Gesture Recognition bereit (TFJS Hand Pose Detection).';
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

async function loadHandPoseDetectionScript() {
  if (typeof window !== 'undefined' && window.handPoseDetection?.createDetector) {
    return window.handPoseDetection;
  }

  const candidates = [
    'https://cdn.jsdelivr.net/npm/@tensorflow-models/hand-pose-detection@0.0.7/dist/hand-pose-detection.min.js',
    'https://cdn.jsdelivr.net/npm/@tensorflow-models/hand-pose-detection@0.0.7/dist/hand-pose-detection.js',
    'https://unpkg.com/@tensorflow-models/hand-pose-detection@0.0.7/dist/hand-pose-detection.min.js',
  ];

  for (const src of candidates) {
    /* eslint-disable no-await-in-loop */
    const loaded = await loadScript(src);
    if (loaded) {
      return window.handPoseDetection;
    }
  }

  throw new Error('Failed to load Hand Pose Detection script.');
}

function loadScript(src) {
  return new Promise((resolve) => {
    const existing = document.querySelector(`script[data-handpose-src="${src}"]`);
    if (existing && window.handPoseDetection?.createDetector) {
      resolve(true);
      return;
    }
    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.dataset.handposeSrc = src;
    script.onload = () => resolve(!!window.handPoseDetection?.createDetector);
    script.onerror = () => resolve(false);
    document.head.appendChild(script);
  });
}

export async function detectGestureLandmarks(videoEl, { flipHorizontal = true } = {}) {
  const recognizer = await ensureGestureRecognizer();
  if (!recognizer || !videoEl) return null;
  const predictions = await recognizer.estimateHands(videoEl, { flipHorizontal });
  const hand = predictions?.[0];
  const rawLandmarks = hand?.keypoints3D ?? hand?.keypoints;
  if (!rawLandmarks || rawLandmarks.length === 0) return null;
  const vw = videoEl.videoWidth || 1;
  const vh = videoEl.videoHeight || 1;
  const norm = Math.max(vw, vh);
  const normalized = rawLandmarks.map((pt) => ({
    x: pt.x / vw,
    y: pt.y / vh,
    z: (pt.z ?? 0) / norm,
  }));
  return applyOneEuroFilter(normalized);
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
  if (
    !PREVIEW_VIDEO ||
    PREVIEW_VIDEO.readyState < 2 ||
    !PREVIEW_VIDEO.videoWidth ||
    !PREVIEW_VIDEO.videoHeight
  ) {
    return;
  }
  if (state.gestureBusy) return;
  if (!state.previewReady) return;
  if (!GESTURE_OVERLAY) return;
  state.gestureBusy = true;
  try {
    const landmarks = await detectGestureLandmarks(PREVIEW_VIDEO);
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

function applyOneEuroFilter(landmarks) {
  const nowSeconds = performance.now() / 1000;
  ensureFilters(landmarks.length);
  return landmarks.map((pt, idx) => {
    const filters = oneEuroFilters[idx];
    return {
      x: filters.x.filter(pt.x, nowSeconds),
      y: filters.y.filter(pt.y, nowSeconds),
      z: filters.z.filter(pt.z ?? 0, nowSeconds),
    };
  });
}

function ensureFilters(count) {
  const freq = 60;
  const minCutoff = 0.004;
  const beta = 0.7;
  const dCutoff = 1.0;
  for (let i = oneEuroFilters.length; i < count; i++) {
    oneEuroFilters[i] = {
      x: new OneEuroFilter(freq, minCutoff, beta, dCutoff),
      y: new OneEuroFilter(freq, minCutoff, beta, dCutoff),
      z: new OneEuroFilter(freq, minCutoff, beta, dCutoff),
    };
  }
}

class OneEuroFilter {
  constructor(freq, minCutoff, beta, dCutoff) {
    this.freq = freq;
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.lastTime = null;
    this.xPrev = null;
    this.dxPrev = 0;
  }

  alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(x, t) {
    if (this.lastTime === null) {
      this.lastTime = t;
      this.xPrev = x;
      return x;
    }
    const dt = Math.max(1e-3, t - this.lastTime);
    this.lastTime = t;

    const dx = (x - this.xPrev) / dt;
    const alphaD = this.alpha(this.dCutoff, dt);
    this.dxPrev = this.dxPrev + alphaD * (dx - this.dxPrev);

    const cutoff = this.minCutoff + this.beta * Math.abs(this.dxPrev);
    const alpha = this.alpha(cutoff, dt);
    this.xPrev = this.xPrev + alpha * (x - this.xPrev);

    return this.xPrev;
  }
}
