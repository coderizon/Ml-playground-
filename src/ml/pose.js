import {
  POSE_FEATURE_SIZE,
  POSE_SAMPLE_INTERVAL_MS,
  STOP_DATA_GATHER,
} from '../constants.js';
import { CAPTURE_VIDEO, GESTURE_OVERLAY, PREVIEW_VIDEO, STATUS } from '../domRefs.js';
import { captureCanvas } from '../camera/webcam.js';
import { clearOverlay, resizeOverlay } from './overlay.js';
import { getState, mutateState } from '../state.js';
import { updateExampleCounts } from '../ui/classes.js';

const POSE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task';
const MP_VERSION = '0.10.8';
let poseCaptureLoop = null;
const state = getState();

export function resetPoseSamples() {
  mutateState((draft) => {
    draft.poseSamples.length = 0;
    draft.poseLastSampleTs = 0;
  });
}

export async function ensurePoseLandmarker() {
  if (state.poseLandmarker) return state.poseLandmarker;
  if (state.poseInitPromise) return state.poseInitPromise;

  mutateState((draft) => {
    draft.poseInitPromise = (async () => {
      try {
        const vision = await import(
          `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`
        );
        const fileset = await vision.FilesetResolver.forVisionTasks(
          `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`
        );
        const landmarker = await vision.PoseLandmarker.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath: POSE_MODEL_URL,
            delegate: 'CPU',
          },
          runningMode: 'VIDEO',
          numPoses: 1,
        });

        mutateState((innerDraft) => {
          innerDraft.poseLandmarker = landmarker;
          innerDraft.poseVision = vision;
        });

        if (GESTURE_OVERLAY) {
          const ctx = GESTURE_OVERLAY.getContext('2d');
          mutateState((innerDraft) => {
            innerDraft.poseDrawingUtils = ctx ? new vision.DrawingUtils(ctx) : null;
          });
        }

        if (STATUS) {
          STATUS.innerText = 'Pose Landmarker bereit.';
        }

        return landmarker;
      } catch (error) {
        console.error(error);
        mutateState((innerDraft) => {
          innerDraft.poseLandmarker = null;
          innerDraft.poseInitPromise = null;
        });
        if (STATUS) {
          STATUS.innerText = 'Pose Landmarker konnte nicht geladen werden.';
        }
        return null;
      }
    })();
  });

  return state.poseInitPromise;
}

export function runPoseLoop() {
  if (poseCaptureLoop || state.currentMode !== 'pose') return;

  const loop = async () => {
    if (state.currentMode !== 'pose') {
      stopPoseLoop();
      return;
    }
    poseCaptureLoop = window.requestAnimationFrame(loop);

    const detection = await detectPoseLandmarks(CAPTURE_VIDEO);
    drawPoseOnCanvas(detection?.landmarks ?? null, captureCanvas);

    if (!detection || detection.vector.length !== POSE_FEATURE_SIZE) return;
    if (state.gatherDataState === STOP_DATA_GATHER) return;

    const now = performance.now();
    if (now - state.poseLastSampleTs < POSE_SAMPLE_INTERVAL_MS) return;

    mutateState((draft) => {
      draft.poseSamples.push({
        landmarks: detection.vector,
        labelId: draft.gatherDataState,
      });
      draft.poseLastSampleTs = now;
      if (draft.examplesCount[draft.gatherDataState] === undefined) {
        draft.examplesCount[draft.gatherDataState] = 0;
      }
      draft.examplesCount[draft.gatherDataState]++;
    });
    updateExampleCounts();
  };

  poseCaptureLoop = window.requestAnimationFrame(loop);
}

export function stopPoseLoop() {
  if (poseCaptureLoop) {
    window.cancelAnimationFrame(poseCaptureLoop);
    poseCaptureLoop = null;
  }
  mutateState((draft) => {
    draft.poseLastSampleTs = 0;
  });
  drawPoseOnCanvas(null, captureCanvas);
}

export async function predictPose() {
  if (!state.model) return null;
  const detection = await detectPoseLandmarks(PREVIEW_VIDEO);
  if (!detection || detection.vector.length !== POSE_FEATURE_SIZE) {
    clearOverlay();
    return null;
  }

  drawPoseOverlay(detection.landmarks);

  const probabilities = tf.tidy(() => {
    const input = tf.tensor2d([detection.vector], [1, POSE_FEATURE_SIZE]);
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

export async function trainPoseModel({ batchSize, epochs, learningRate, onEpochEnd }) {
  if (!state.poseSamples.length) {
    throw new Error('Keine Pose-Beispiele gesammelt.');
  }

  const outputUnits = Math.max(state.classNames.length, 1);

  mutateState((draft) => {
    if (draft.model) {
      draft.model.dispose();
    }
    draft.model = buildPoseClassifier(outputUnits, learningRate);
  });

  const xs = tf.tensor2d(
    state.poseSamples.map((sample) => sample.landmarks),
    [state.poseSamples.length, POSE_FEATURE_SIZE]
  );
  const labelTensor = tf.tensor1d(state.poseSamples.map((sample) => sample.labelId), 'int32');
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

function buildPoseClassifier(outputUnits, learningRate) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [POSE_FEATURE_SIZE], units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: outputUnits, activation: 'softmax' }));

  const lr = sanitizeLearningRate(learningRate);
  model.compile({
    optimizer: tf.train.adam(lr),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

async function detectPoseLandmarks(videoEl) {
  if (!videoEl || videoEl.readyState < 2 || !videoEl.videoWidth || !videoEl.videoHeight) {
    return null;
  }
  if (state.poseBusy) return null;

  const landmarker = await ensurePoseLandmarker();
  if (!landmarker) return null;

  mutateState((draft) => {
    draft.poseBusy = true;
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
      draft.poseBusy = false;
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

function drawPoseOverlay(landmarks = []) {
  if (!GESTURE_OVERLAY) return;
  resizeOverlay();
  const ctx = GESTURE_OVERLAY.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, GESTURE_OVERLAY.width, GESTURE_OVERLAY.height);

  const PoseLandmarker = state.poseVision?.PoseLandmarker;
  if (!PoseLandmarker || !state.poseDrawingUtils) return;

  state.poseDrawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
    color: '#10b981',
    lineWidth: 3,
  });
  state.poseDrawingUtils.drawLandmarks(landmarks, {
    color: '#22d3ee',
    lineWidth: 2,
  });
}

function drawPoseOnCanvas(landmarks = [], canvas) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const width = CAPTURE_VIDEO?.videoWidth || canvas.width;
  const height = CAPTURE_VIDEO?.videoHeight || canvas.height;
  if (width && height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.clearRect(0, 0, canvas.width || width || 0, canvas.height || height || 0);
  if (!landmarks || !landmarks.length) return;

  const PoseLandmarker = state.poseVision?.PoseLandmarker;
  if (!PoseLandmarker || !state.poseVision) return;
  const drawingUtils = new state.poseVision.DrawingUtils(ctx);
  drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
    color: '#10b981',
    lineWidth: 3,
  });
  drawingUtils.drawLandmarks(landmarks, { color: '#22d3ee', lineWidth: 2 });
}

function sanitizeLearningRate(value) {
  const lr = Number(value);
  if (!Number.isFinite(lr) || lr <= 0) return 0.001;
  return lr;
}
