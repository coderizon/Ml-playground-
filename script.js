const STATUS = document.getElementById('status');
const CAPTURE_VIDEO = document.getElementById('captureCam');
const PREVIEW_VIDEO = document.getElementById('previewCam');
const RESET_BUTTON = document.getElementById('reset');
const TRAIN_BUTTON = document.getElementById('train');
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;
const CLASS_NAMES = ['Class 1', 'Class 2'];

const openWebcamButtons = document.querySelectorAll('.open-webcam');
const webcamPanels = document.querySelectorAll('.webcam-panel');
const closePanelButtons = document.querySelectorAll('.icon-close');
const classNameInputs = document.querySelectorAll('.class-name-input');
const captureSlots = document.querySelectorAll('.capture-slot');
const countChips = document.querySelectorAll('[data-count-for]');
const mobileStepButtons = document.querySelectorAll('[data-step-target]');
const bodyEl = document.body;
const supportsPointer = 'onpointerdown' in window;

TRAIN_BUTTON.addEventListener('click', trainAndPredict);
RESET_BUTTON.addEventListener('click', reset);

const dataCollectorButtons = document.querySelectorAll('button.dataCollector');
dataCollectorButtons.forEach((btn) => {
  if (supportsPointer) {
    btn.addEventListener('pointerdown', handleCollectStart, { passive: false });
    btn.addEventListener('pointerup', handleCollectEnd);
    btn.addEventListener('pointerleave', handleCollectEnd);
  } else {
    btn.addEventListener('mousedown', handleCollectStart);
    btn.addEventListener('mouseup', handleCollectEnd);
    btn.addEventListener('touchstart', handleCollectStart, { passive: false });
    btn.addEventListener('touchend', handleCollectEnd);
  }
});

openWebcamButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    const idx = parseInt(btn.getAttribute('data-class-index'));
    openWebcamForClass(idx);
  });
});

closePanelButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    const idx = parseInt(btn.getAttribute('data-close-panel'));
    hideWebcamPanel(idx);
  });
});

classNameInputs.forEach((input, idx) => {
  CLASS_NAMES[idx] = input.value;
  input.addEventListener('input', () => {
    CLASS_NAMES[idx] = input.value || `Class ${idx + 1}`;
    const collector = dataCollectorButtons[idx];
    if (collector) collector.setAttribute('data-name', CLASS_NAMES[idx]);
    STATUS.innerText = `Klasse ${idx + 1} benannt als ${CLASS_NAMES[idx]}.`;
  });
});

mobileStepButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    setMobileStep(btn.getAttribute('data-step-target'));
  });
});

let currentStream;
let activeClassIndex = 0;
let previewReady = false;
let trainingCompleted = false;

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

async function enableCam() {
  if (!hasGetUserMedia()) {
    console.warn('getUserMedia() is not supported by your browser');
    return;
  }

  const constraints = { video: { width: 640, height: 480 } };

  if (currentStream) {
    attachStreamToVideos(currentStream);
    return;
  }

  try {
    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    attachStreamToVideos(currentStream);
  } catch (err) {
    console.error(err);
  }
}

function attachStreamToVideos(stream) {
  CAPTURE_VIDEO.srcObject = stream;
  PREVIEW_VIDEO.srcObject = stream;
  CAPTURE_VIDEO.addEventListener('loadeddata', function onLoad() {
    videoPlaying = true;
    CAPTURE_VIDEO.classList.remove('hidden');
    CAPTURE_VIDEO.removeEventListener('loadeddata', onLoad);
  });
}

function openWebcamForClass(idx) {
  activeClassIndex = idx;
  webcamPanels.forEach((panel) => {
    panel.classList.toggle('visible', parseInt(panel.getAttribute('data-class-panel')) === idx);
  });
  moveCaptureToSlot(idx);
  enableCam();
  STATUS.innerText = `Webcam geöffnet für ${CLASS_NAMES[idx]}. Halte zum Aufnehmen.`;
}

function moveCaptureToSlot(idx) {
  const slot = Array.from(captureSlots).find(
    (s) => parseInt(s.getAttribute('data-class-slot')) === idx
  );
  if (slot && CAPTURE_VIDEO.parentElement !== slot) {
    slot.innerHTML = '';
    slot.appendChild(CAPTURE_VIDEO);
  }
}

function hideWebcamPanel(idx) {
  if (gatherDataState !== STOP_DATA_GATHER) {
    gatherDataState = STOP_DATA_GATHER;
  }
  const panel = Array.from(webcamPanels).find(
    (p) => parseInt(p.getAttribute('data-class-panel')) === idx
  );
  if (panel) {
    panel.classList.remove('visible');
  }
}

async function trainAndPredict() {
  if (trainingCompleted) return;
  predict = false;
  tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
  const outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
  const oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
  const inputsAsTensor = tf.stack(trainingDataInputs);

  await model.fit(inputsAsTensor, oneHotOutputs, {
    shuffle: true,
    batchSize: 5,
    epochs: 10,
    callbacks: { onEpochEnd: logProgress },
  });

  outputsAsTensor.dispose();
  oneHotOutputs.dispose();
  inputsAsTensor.dispose();
  predict = true;
  trainingCompleted = true;
  lockCapturePanels();
  showPreview();
  setMobileStep('preview');
  predictLoop();
}

function logProgress(epoch, logs) {
  console.log('Data for epoch ' + epoch, logs);
}

function showPreview() {
  PREVIEW_VIDEO.classList.remove('hidden');
  if (PREVIEW_VIDEO.readyState >= 2) {
    previewReady = true;
    return;
  }
  PREVIEW_VIDEO.addEventListener(
    'loadeddata',
    function onPreviewReady() {
      previewReady = true;
      PREVIEW_VIDEO.removeEventListener('loadeddata', onPreviewReady);
    },
    { once: true }
  );
}

function predictLoop() {
  if (!predict) return;

  if (previewReady) {
    tf.tidy(function () {
      const videoFrameAsTensor = tf.browser.fromPixels(PREVIEW_VIDEO).div(255);
      const resizedTensorFrame = tf.image.resizeBilinear(
        videoFrameAsTensor,
        [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
        true
      );

      const imageFeatures = mobilenet.predict(resizedTensorFrame.expandDims());
      const prediction = model.predict(imageFeatures).squeeze();
      const highestIndex = prediction.argMax().arraySync();
      const predictionArray = prediction.arraySync();

      STATUS.innerText =
        'Prediction: ' +
        CLASS_NAMES[highestIndex] +
        ' with ' +
        Math.floor(predictionArray[highestIndex] * 100) +
        '% confidence';
    });
  }

  window.requestAnimationFrame(predictLoop);
}

function gatherDataForClass() {
  if (trainingCompleted) return;
  const classNumber = parseInt(this.getAttribute('data-1hot'));
  gatherDataState = gatherDataState === STOP_DATA_GATHER ? classNumber : STOP_DATA_GATHER;
  dataGatherLoop();
}

function dataGatherLoop() {
  if (videoPlaying && gatherDataState !== STOP_DATA_GATHER) {
    const imageFeatures = tf.tidy(function () {
      const videoFrameAsTensor = tf.browser.fromPixels(CAPTURE_VIDEO);
      const resizedTensorFrame = tf.image.resizeBilinear(
        videoFrameAsTensor,
        [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
        true
      );
      const normalizedTensorFrame = resizedTensorFrame.div(255);
      return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
    });

    trainingDataInputs.push(imageFeatures);
    trainingDataOutputs.push(gatherDataState);
    vibrateFeedback();

    if (examplesCount[gatherDataState] === undefined) {
      examplesCount[gatherDataState] = 0;
    }
    examplesCount[gatherDataState]++;

    updateExampleCounts();
    window.requestAnimationFrame(dataGatherLoop);
  }
}

function handleCollectStart(event) {
  event.preventDefault();
  const classNumber = parseInt(event.currentTarget.getAttribute('data-1hot'));
  gatherDataState = classNumber;
  dataGatherLoop();
}

function handleCollectEnd(event) {
  event.preventDefault();
  gatherDataState = STOP_DATA_GATHER;
}

function reset() {
  predict = false;
  previewReady = false;
  trainingCompleted = false;
  examplesCount.length = 0;
  for (let i = 0; i < trainingDataInputs.length; i++) {
    trainingDataInputs[i].dispose();
  }
  trainingDataInputs.length = 0;
  trainingDataOutputs.length = 0;
  STATUS.innerText = 'No data collected';
  PREVIEW_VIDEO.classList.add('hidden');
  unlockCapturePanels();
  updateExampleCounts(true);
  setMobileStep('collect');

  console.log('Tensors in memory: ' + tf.memory().numTensors);
}

let mobilenet = undefined;
let gatherDataState = STOP_DATA_GATHER;
let videoPlaying = false;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false;

async function loadMobileNetFeatureModel() {
  const URL =
    'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';

  mobilenet = await tf.loadGraphModel(URL, { fromTFHub: true });
  STATUS.innerText = 'MobileNet v3 loaded successfully!';

  tf.tidy(function () {
    const answer = mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
    console.log(answer.shape);
  });
}

loadMobileNetFeatureModel();

let model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [1024], units: 128, activation: 'relu' }));
model.add(tf.layers.dense({ units: CLASS_NAMES.length, activation: 'softmax' }));

model.summary();

model.compile({
  optimizer: 'adam',
  loss: CLASS_NAMES.length === 2 ? 'binaryCrossentropy' : 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

function lockCapturePanels() {
  webcamPanels.forEach((panel) => panel.classList.remove('visible'));
  CAPTURE_VIDEO.classList.add('hidden');
  openWebcamButtons.forEach((btn) => (btn.disabled = true));
  dataCollectorButtons.forEach((btn) => (btn.disabled = true));
}

function unlockCapturePanels() {
  openWebcamButtons.forEach((btn) => (btn.disabled = false));
  dataCollectorButtons.forEach((btn) => (btn.disabled = false));
  CAPTURE_VIDEO.classList.remove('hidden');
}

function updateExampleCounts(reset = false) {
  countChips.forEach((chip) => {
    const idx = parseInt(chip.getAttribute('data-count-for'));
    const count = reset ? 0 : examplesCount[idx] || 0;
    chip.textContent = `${count} Bildbeispiele`;
  });
}

function setMobileStep(step) {
  bodyEl.setAttribute('data-step', step);
  mobileStepButtons.forEach((btn) => {
    btn.classList.toggle('active', btn.getAttribute('data-step-target') === step);
  });
}

function vibrateFeedback() {
  if (navigator.vibrate) {
    navigator.vibrate(15);
  }
}
