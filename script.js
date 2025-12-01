const STATUS = document.getElementById('status');
const CAPTURE_VIDEO = document.getElementById('captureCam');
const PREVIEW_VIDEO = document.getElementById('previewCam');
const RESET_BUTTON = document.getElementById('reset');
const TRAIN_BUTTON = document.getElementById('train');
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;

const mobileStepButtons = document.querySelectorAll('[data-step-target]');
const bodyEl = document.body;
const supportsPointer = 'onpointerdown' in window;
const addClassButton = document.getElementById('add-class');
const classesColumn = document.querySelector('.classes-column');

const CLASS_NAMES = [];
const openWebcamButtons = [];
const webcamPanels = [];
const classNameInputs = [];
const captureSlots = [];
const dataCollectorButtons = [];
const countChips = [];
const switchCameraButtons = [];

let currentStream;
let activeClassIndex = 0;
let previewReady = false;
let trainingCompleted = false;
let mobilenet = undefined;
let gatherDataState = STOP_DATA_GATHER;
let videoPlaying = false;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false;
let model;
let preferredFacingMode = 'user';

TRAIN_BUTTON.addEventListener('click', trainAndPredict);
RESET_BUTTON.addEventListener('click', reset);

mobileStepButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    setMobileStep(btn.getAttribute('data-step-target'));
  });
});

initializeExistingClasses();
if (addClassButton) {
  addClassButton.addEventListener('click', addNewClassCard);
  addClassButton.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      addNewClassCard();
    }
  });
}

function initializeExistingClasses() {
  const existingCards = document.querySelectorAll('.class-card');
  existingCards.forEach((card, idx) => {
    setupClassCard(card, idx);
  });
  rebuildModel();
  updateExampleCounts(true);
}

function setupClassCard(card, idx) {
  const nameInput = card.querySelector('.class-name-input');
  const openBtn = card.querySelector('.open-webcam');
  const panel = card.querySelector('.webcam-panel');
  const closeBtn = panel.querySelector('.icon-close');
  const switchBtn = panel.querySelector('.switch-camera');
  const slot = card.querySelector('.capture-slot');
  const collectorBtn = card.querySelector('.dataCollector');
  const countChip = card.querySelector('.count-chip');

  if (!nameInput || !openBtn || !panel || !slot || !collectorBtn || !countChip) return;

  nameInput.setAttribute('data-class-index', idx);
  openBtn.setAttribute('data-class-index', idx);
  panel.setAttribute('data-class-panel', idx);
  slot.setAttribute('data-class-slot', idx);
  collectorBtn.setAttribute('data-1hot', idx);
  countChip.setAttribute('data-count-for', idx);

  const classLabel = nameInput.value || `Class ${idx + 1}`;
  CLASS_NAMES[idx] = classLabel;
  collectorBtn.setAttribute('data-name', classLabel);

  classNameInputs[idx] = nameInput;
  openWebcamButtons[idx] = openBtn;
  webcamPanels[idx] = panel;
  captureSlots[idx] = slot;
  dataCollectorButtons[idx] = collectorBtn;
  countChips[idx] = countChip;
  if (switchBtn) {
    switchCameraButtons[idx] = switchBtn;
  }

  attachNameInputListener(nameInput, idx, collectorBtn);
  attachOpenButtonListener(openBtn, idx);
  attachCollectorButtonListeners(collectorBtn);
  attachSwitchCameraListener(switchBtn);

  if (closeBtn) {
    closeBtn.setAttribute('data-close-panel', idx);
    closeBtn.addEventListener('click', () => hideWebcamPanel(idx));
  }
}

function buildClassCardElement(idx) {
  const card = document.createElement('div');
  card.className = 'card class-card';
  card.innerHTML = `
    <div class="card-header">
      <div class="title-group editable">
        <input class="class-name-input" data-class-index="${idx}" value="Class ${idx + 1}" aria-label="Klassenname eingeben">
      </div>
      <span class="dots">⋮</span>
    </div>
    <p class="section-label">Bildbeispiele hinzufügen:</p>
    <div class="action-row">
      <button class="open-webcam ghost" data-class-index="${idx}">Webcam</button>
    </div>
    <div class="webcam-panel" data-class-panel="${idx}">
      <div class="panel-top">
        <span>Webcam</span>
        <div class="panel-actions">
          <button class="ghost switch-camera" data-switch-camera aria-label="Kamera wechseln">Außenkamera</button>
          <button class="icon-close" data-close-panel="${idx}" aria-label="Panel schließen">×</button>
        </div>
      </div>
      <div class="count-row">
        <span class="count-chip" data-count-for="${idx}">0 Bildbeispiele</span>
      </div>
      <div class="capture-slot" data-class-slot="${idx}"></div>
      <button class="dataCollector primary block" data-1hot="${idx}" data-name="Class ${idx + 1}">Zum Aufnehmen halten</button>
    </div>
  `;
  return card;
}

function attachCollectorButtonListeners(btn) {
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
}

function attachOpenButtonListener(btn, idx) {
  btn.addEventListener('click', () => openWebcamForClass(idx));
}

function attachNameInputListener(input, idx, collectorBtn) {
  input.addEventListener('input', () => {
    CLASS_NAMES[idx] = input.value || `Class ${idx + 1}`;
    collectorBtn.setAttribute('data-name', CLASS_NAMES[idx]);
    STATUS.innerText = `Klasse ${idx + 1} benannt als ${CLASS_NAMES[idx]}.`;
  });
}

function attachSwitchCameraListener(btn) {
  if (!btn) return;
  btn.addEventListener('click', () => {
    preferredFacingMode = preferredFacingMode === 'user' ? 'environment' : 'user';
    stopCurrentStream();
    moveCaptureToSlot(activeClassIndex);
    enableCam();
    updateSwitchButtonsLabel();
    STATUS.innerText =
      preferredFacingMode === 'environment'
        ? 'Außenkamera aktiviert.'
        : 'Selfie-Kamera aktiviert.';
  });
  updateSwitchButtonsLabel();
}

function addNewClassCard() {
  const newIndex = CLASS_NAMES.length;
  const newCard = buildClassCardElement(newIndex);
  if (classesColumn && addClassButton) {
    classesColumn.insertBefore(newCard, addClassButton);
  } else if (classesColumn) {
    classesColumn.appendChild(newCard);
  }
  setupClassCard(newCard, newIndex);
  examplesCount[newIndex] = 0;
  updateExampleCounts();
  trainingCompleted = false;
  predict = false;
  previewReady = false;
  PREVIEW_VIDEO.classList.add('hidden');
  rebuildModel();
  STATUS.innerText = `Neue Klasse ${CLASS_NAMES[newIndex]} hinzugefügt.`;
}

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function stopCurrentStream() {
  if (!currentStream) return;
  currentStream.getTracks().forEach((track) => track.stop());
  currentStream = undefined;
  videoPlaying = false;
}

function updateSwitchButtonsLabel() {
  const targetLabel = preferredFacingMode === 'user' ? 'Außenkamera' : 'Selfie-Kamera';
  switchCameraButtons.forEach((btn) => {
    if (btn) btn.textContent = targetLabel;
  });
}

async function enableCam(allowFallback = true) {
  if (!hasGetUserMedia()) {
    console.warn('getUserMedia() is not supported by your browser');
    return;
  }

  const facingConstraint = preferredFacingMode === 'environment' ? { exact: 'environment' } : 'user';
  const constraints = { video: { width: 640, height: 480, facingMode: facingConstraint } };

  if (currentStream) {
    attachStreamToVideos(currentStream);
    return;
  }

  try {
    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    attachStreamToVideos(currentStream);
  } catch (err) {
    console.error(err);
    if (allowFallback && preferredFacingMode === 'environment') {
      preferredFacingMode = 'user';
      updateSwitchButtonsLabel();
      STATUS.innerText = 'Außenkamera nicht verfügbar, Selfie-Kamera aktiviert.';
      return enableCam(false);
    }
    STATUS.innerText = 'Webcam konnte nicht gestartet werden.';
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
  const slot = captureSlots.find((s) => parseInt(s.getAttribute('data-class-slot')) === idx);
  if (slot && CAPTURE_VIDEO.parentElement !== slot) {
    slot.innerHTML = '';
    slot.appendChild(CAPTURE_VIDEO);
  }
}

function hideWebcamPanel(idx) {
  if (gatherDataState !== STOP_DATA_GATHER) {
    gatherDataState = STOP_DATA_GATHER;
  }
  const panel = webcamPanels.find((p) => parseInt(p.getAttribute('data-class-panel')) === idx);
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
  gatherDataState = STOP_DATA_GATHER;
  examplesCount.length = 0;
  for (let i = 0; i < trainingDataInputs.length; i++) {
    trainingDataInputs[i].dispose();
  }
  trainingDataInputs.length = 0;
  trainingDataOutputs.length = 0;
  STATUS.innerText = 'No data collected';
  PREVIEW_VIDEO.classList.add('hidden');
  unlockCapturePanels();
  rebuildModel();
  updateExampleCounts(true);
  setMobileStep('collect');

  console.log('Tensors in memory: ' + tf.memory().numTensors);
}

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

function rebuildModel() {
  const outputUnits = Math.max(CLASS_NAMES.length, 1);

  if (model) {
    model.dispose();
  }

  model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [1024], units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: outputUnits, activation: 'softmax' }));

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  model.summary();
}

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
