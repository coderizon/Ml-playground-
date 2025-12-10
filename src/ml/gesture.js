import { CAPTURE_VIDEO, GESTURE_OVERLAY, PREVIEW_VIDEO, STATUS } from '../domRefs.js';
import { captureCanvas } from '../camera/webcam.js';
import { GESTURE_FEATURE_SIZE, GESTURE_SAMPLE_INTERVAL_MS, STOP_DATA_GATHER } from '../constants.js';
import { getState, mutateState } from '../state.js';
import { clearOverlay, resizeOverlay } from './overlay.js';
import { updateExampleCounts } from '../ui/classes.js';

const HAND_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const MP_VERSION = '0.10.8';
let gestureLoopHandle = null;
let captureDrawingUtils = null;
const state = getState();

export function resetGestureSamples() {
  mutateState((draft) => {
    draft.gestureSamples.length = 0;
  });
}

export async function ensureHandLandmarker() {
  if (state.handLandmarker) return state.handLandmarker;
  if (state.handInitPromise) return state.handInitPromise;

  mutateState((draft) => {
    draft.handInitPromise = (async () => {
    try {
      const vision = await import(
        `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`
      );
      const fileset = await vision.FilesetResolver.forVisionTasks(
        `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`
      );
      const landmarker = await vision.HandLandmarker.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath: HAND_MODEL_URL,
          delegate: 'CPU',
        },
        numHands: 1,
        runningMode: 'VIDEO',
      });

      mutateState((draft) => {
        draft.handVision = vision;
        draft.handLandmarker = landmarker;
      });

      if (GESTURE_OVERLAY) {
        const ctx = GESTURE_OVERLAY.getContext('2d');
        mutateState((draft) => {
          draft.handDrawingUtils = ctx ? new vision.DrawingUtils(ctx) : null;
        });
      }
      if (captureCanvas) {
        const captureCtx = captureCanvas.getContext('2d');
        captureDrawingUtils = captureCtx ? new vision.DrawingUtils(captureCtx) : null;
      }

      if (STATUS) {
        STATUS.innerText = 'Hand Landmarker bereit.';
      }

      return landmarker;
    } catch (error) {
      console.error(error);
      mutateState((draft) => {
        draft.handLandmarker = null;
        draft.handInitPromise = null;
      });
      if (STATUS) {
        STATUS.innerText = 'Hand Landmarker konnte nicht geladen werden.';
      }
      return null;
    }
    })();
  });

  return state.handInitPromise;
}

function drawSkeletonOnCanvas(landmarks) {
  if (!captureCanvas) return;
  const ctx = captureCanvas.getContext('2d');
  if (!ctx) return;

  const width = CAPTURE_VIDEO?.videoWidth || captureCanvas.width;
  const height = CAPTURE_VIDEO?.videoHeight || captureCanvas.height;

  if (width && height) {
    captureCanvas.width = width;
    captureCanvas.height = height;
  }

  ctx.clearRect(0, 0, captureCanvas.width || width || 0, captureCanvas.height || height || 0);

  if (!landmarks || !landmarks.length) return;

  const HandLandmarker = state.handVision?.HandLandmarker;
  if (!HandLandmarker || !state.handVision) return;

  if (!captureDrawingUtils) {
    captureDrawingUtils = new state.handVision.DrawingUtils(ctx);
  }

  captureDrawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
    color: '#55c3ff',
    lineWidth: 3,
  });
  captureDrawingUtils.drawLandmarks(landmarks, {
    color: '#0066ff',
    lineWidth: 2,
  });
}

export function runGestureLoop() {
  if (gestureLoopHandle || state.currentMode !== 'gesture') return;

  const loop = async () => {
    if (state.currentMode !== 'gesture') {
      stopGestureLoop();
      return;
    }

    gestureLoopHandle = window.requestAnimationFrame(loop);

    const detection = await detectHandLandmarks(CAPTURE_VIDEO);
    drawSkeletonOnCanvas(detection?.landmarks ?? null);

  if (!detection || detection.vector.length !== GESTURE_FEATURE_SIZE) return;
  if (state.gatherDataState === STOP_DATA_GATHER) return;

  const now = performance.now();
  if (now - state.gestureLastSampleTs < GESTURE_SAMPLE_INTERVAL_MS) return;

  mutateState((draft) => {
    draft.gestureSamples.push({
      landmarks: detection.vector,
      labelId: draft.gatherDataState,
    });
    draft.gestureLastSampleTs = now;
    if (draft.examplesCount[draft.gatherDataState] === undefined) {
      draft.examplesCount[draft.gatherDataState] = 0;
    }
    draft.examplesCount[draft.gatherDataState]++;
  });
  updateExampleCounts();
};

  gestureLoopHandle = window.requestAnimationFrame(loop);
}

export function stopGestureLoop() {
  if (gestureLoopHandle) {
    window.cancelAnimationFrame(gestureLoopHandle);
    gestureLoopHandle = null;
  }
  mutateState((draft) => {
    draft.gestureLastSampleTs = 0;
  });
  drawSkeletonOnCanvas(null);
}

export async function predictGesture() {
  if (!state.model) return null;
  const detection = await detectHandLandmarks(PREVIEW_VIDEO);
  if (!detection || detection.vector.length !== GESTURE_FEATURE_SIZE) {
    clearGestureOverlay();
    return null;
  }

  drawHandOverlay(detection.landmarks);

  const probabilities = tf.tidy(() => {
    const input = tf.tensor2d([detection.vector], [1, GESTURE_FEATURE_SIZE]);
    const prediction = state.model.predict(input).squeeze();
    return prediction.arraySync();
  });

  const bestIndex =
    probabilities.length > 0
      ? probabilities.reduce(
          (bestIdx, value, idx, arr) => (value > arr[bestIdx] ? idx : bestIdx),
          0
        )
      : -1;

  return { probabilities, bestIndex };
}

export async function trainGestureModel({ batchSize, epochs, learningRate, onEpochEnd }) {
  if (!state.gestureSamples.length) {
    throw new Error('Keine Gesten-Beispiele gesammelt.');
  }

  const outputUnits = Math.max(state.classNames.length, 1);

  mutateState((draft) => {
    if (draft.model) {
      draft.model.dispose();
    }
    draft.model = buildGestureClassifier(outputUnits, learningRate);
  });

  const xs = tf.tensor2d(
    state.gestureSamples.map((sample) => sample.landmarks),
    [state.gestureSamples.length, GESTURE_FEATURE_SIZE]
  );
  const labelTensor = tf.tensor1d(
    state.gestureSamples.map((sample) => sample.labelId),
    'int32'
  );
  const ys = tf.oneHot(labelTensor, outputUnits);

  try {
    await state.model.fit(xs, ys, {
      shuffle: true,
      batchSize,
      epochs,
      callbacks: {
        onEpochEnd,
      },
    });
  } finally {
    xs.dispose();
    ys.dispose();
    labelTensor.dispose();
  }

  return state.model;
}

function buildGestureClassifier(outputUnits, learningRate) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [GESTURE_FEATURE_SIZE], units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: outputUnits, activation: 'softmax' }));

  const lr = sanitizeLearningRate(learningRate);
  model.compile({
    optimizer: tf.train.adam(lr),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

async function detectHandLandmarks(videoEl) {
  if (!videoEl || videoEl.readyState < 2 || !videoEl.videoWidth || !videoEl.videoHeight) {
    return null;
  }
  if (state.handBusy) return null;

  const landmarker = await ensureHandLandmarker();
  if (!landmarker) return null;

  mutateState((draft) => {
    draft.handBusy = true;
  });
  try {
    const nowInMs = performance.now();
    const result = landmarker.detectForVideo(videoEl, nowInMs);
    const landmarks = result?.landmarks?.[0];
    if (!landmarks || !landmarks.length) return null;

    return { landmarks, vector: flattenLandmarks(landmarks) };
  } catch (error) {
    console.error(error);
    return null;
  } finally {
    mutateState((draft) => {
      draft.handBusy = false;
    });
  }
}

function flattenLandmarks(landmarks = []) {
  const flat = [];
  for (let i = 0; i < landmarks.length; i++) {
    const point = landmarks[i];
    flat.push(point.x ?? 0);
    flat.push(point.y ?? 0);
    flat.push(point.z ?? 0);
  }
  return flat;
}

function drawHandOverlay(landmarks = []) {
  if (!GESTURE_OVERLAY) return;
  resizeOverlay();
  const ctx = GESTURE_OVERLAY.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, GESTURE_OVERLAY.width, GESTURE_OVERLAY.height);

  const HandLandmarker = state.handVision?.HandLandmarker;
  if (!HandLandmarker || !state.handDrawingUtils) return;

  state.handDrawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
    color: '#55c3ff',
    lineWidth: 3,
  });
  state.handDrawingUtils.drawLandmarks(landmarks, {
    color: '#0066ff',
    lineWidth: 2,
  });
}

function clearGestureOverlay() {
  if (!GESTURE_OVERLAY) return;
  const ctx = GESTURE_OVERLAY.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, GESTURE_OVERLAY.width, GESTURE_OVERLAY.height);
}

function sanitizeLearningRate(value) {
  const lr = Number(value);
  if (!Number.isFinite(lr) || lr <= 0) return 0.001;
  return lr;
}
