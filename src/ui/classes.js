import {
  CLASS_DEFAULT_PREFIX,
  DEFAULT_CAPTURE_LABEL,
  DEFAULT_COLLECT_LABEL,
} from '../constants.js';
import { CAPTURE_VIDEO, STATUS, addClassButton, classesColumn } from '../domRefs.js';
import { getState, mutateState } from '../state.js';

const state = getState();

export function initializeExistingClasses(handlers = {}) {
  const existingCards = document.querySelectorAll('.class-card');
  existingCards.forEach((card, idx) => {
    setupClassCard(card, idx, handlers);
  });
}

export function addNewClassCard(handlers = {}) {
  const newIndex = state.classNames.length;
  const newCard = buildClassCardElement(newIndex);
  if (classesColumn && addClassButton) {
    classesColumn.insertBefore(newCard, addClassButton);
  } else if (classesColumn) {
    classesColumn.appendChild(newCard);
  }
  setupClassCard(newCard, newIndex, handlers);
  mutateState((draft) => {
    draft.examplesCount[newIndex] = 0;
  });
  updateExampleCounts();
  return newIndex;
}

export function setupClassCard(card, idx, handlers = {}) {
  const {
    onNameChange = () => {},
    onOpenWebcam = () => {},
    onCollectStart = () => {},
    onCollectEnd = () => {},
    onSwitchCamera = () => {},
    onClosePanel = () => {},
  } = handlers;

  const nameInput = card.querySelector('.class-name-input');
  const openBtn = card.querySelector('.open-webcam');
  const panel = card.querySelector('.webcam-panel');
  const panelLabel = panel ? panel.querySelector('.panel-top span') : null;
  const closeBtn = panel ? panel.querySelector('.icon-close') : null;
  const switchBtn = panel ? panel.querySelector('.switch-camera') : null;
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

  const classLabel = nameInput.value || getDefaultClassLabel(idx);
  collectorBtn.textContent = DEFAULT_COLLECT_LABEL;
  mutateState((draft) => {
    draft.classNames[idx] = classLabel;
    collectorBtn.setAttribute('data-name', classLabel);
    draft.classNameInputs[idx] = nameInput;
    draft.openWebcamButtons[idx] = openBtn;
    draft.webcamPanels[idx] = panel;
    draft.captureSlots[idx] = slot;
    draft.dataCollectorButtons[idx] = collectorBtn;
    draft.countChips[idx] = countChip;
    if (switchBtn) {
      draft.switchCameraButtons[idx] = switchBtn;
    }
  });

  if (panelLabel) {
    panelLabel.textContent = DEFAULT_CAPTURE_LABEL;
  }

  const clearNameOnce = () => {
    if (nameInput.dataset.cleared === 'true') return;
    nameInput.dataset.cleared = 'true';
    nameInput.value = '';
    mutateState((draft) => {
      draft.classNames[idx] = '';
    });
    collectorBtn.setAttribute('data-name', '');
    onNameChange(idx, state.classNames[idx]);
  };

  nameInput.addEventListener('focus', clearNameOnce);
  nameInput.addEventListener('pointerdown', clearNameOnce);

  nameInput.addEventListener('input', () => {
    const nextName = nameInput.value || getDefaultClassLabel(idx);
    mutateState((draft) => {
      draft.classNames[idx] = nextName;
    });
    collectorBtn.setAttribute('data-name', nextName);
    if (STATUS) {
      STATUS.innerText = `Klasse ${idx + 1} benannt als ${state.classNames[idx]}.`;
    }
    onNameChange(idx, state.classNames[idx]);
  });

  openBtn.addEventListener('click', () => onOpenWebcam(idx));
  attachCollectorButtonListeners(collectorBtn, onCollectStart, onCollectEnd);

  if (switchBtn) {
    switchBtn.addEventListener('click', () => onSwitchCamera(idx));
  }

  if (closeBtn) {
    closeBtn.setAttribute('data-close-panel', idx);
    closeBtn.addEventListener('click', () => onClosePanel(idx));
  }
}

function attachCollectorButtonListeners(btn, onCollectStart, onCollectEnd) {
  const supportsPointer = 'onpointerdown' in window;
  if (supportsPointer) {
    btn.addEventListener('pointerdown', onCollectStart, { passive: false });
    btn.addEventListener('pointerup', onCollectEnd);
    btn.addEventListener('pointerleave', onCollectEnd);
  } else {
    btn.addEventListener('mousedown', onCollectStart);
    btn.addEventListener('mouseup', onCollectEnd);
    btn.addEventListener('touchstart', onCollectStart, { passive: false });
    btn.addEventListener('touchend', onCollectEnd);
  }
}

function buildClassCardElement(idx) {
  const defaultLabel = getDefaultClassLabel(idx);
  const panelLabel = DEFAULT_CAPTURE_LABEL;
  const card = document.createElement('div');
  card.className = 'card class-card';
  card.innerHTML = `
    <div class="card-header">
      <div class="title-group editable">
        <input class="class-name-input" data-class-index="${idx}" value="${defaultLabel}" aria-label="Klassenname eingeben">
      </div>
      <span class="dots">⋮</span>
    </div>
    <p class="section-label">Beispiele hinzufügen:</p>
    <div class="action-row">
      <button class="open-webcam ghost" data-class-index="${idx}">${panelLabel}</button>
    </div>
    <div class="webcam-panel" data-class-panel="${idx}">
      <div class="panel-top">
        <span>${panelLabel}</span>
        <div class="panel-actions">
          <button class="ghost switch-camera" data-switch-camera aria-label="Kamera wechseln">Außenkamera</button>
          <button class="icon-close" data-close-panel="${idx}" aria-label="Panel schließen">×</button>
        </div>
      </div>
      <div class="count-row">
        <span class="count-chip" data-count-for="${idx}">0 Beispiele</span>
      </div>
      <div class="capture-slot" data-class-slot="${idx}"></div>
      <button class="dataCollector primary block" data-1hot="${idx}" data-name="${defaultLabel}">${DEFAULT_COLLECT_LABEL}</button>
    </div>
  `;
  return card;
}

function getDefaultClassLabel(idx) {
  return `${CLASS_DEFAULT_PREFIX} ${idx + 1}`;
}

export function updateExampleCounts(reset = false) {
  state.countChips.forEach((chip) => {
    const idx = parseInt(chip.getAttribute('data-count-for'), 10);
    const count = reset ? 0 : state.examplesCount[idx] || 0;
    chip.textContent = `${count} Beispiele`;
  });
}

export function resetClassCards(handlers = {}) {
  if (classesColumn) {
    classesColumn.querySelectorAll('.class-card').forEach((card) => card.remove());
  }

  mutateState((draft) => {
    draft.classNames.length = 0;
    draft.examplesCount.length = 0;
    draft.classNameInputs.length = 0;
    draft.openWebcamButtons.length = 0;
    draft.webcamPanels.length = 0;
    draft.captureSlots.length = 0;
    draft.dataCollectorButtons.length = 0;
    draft.countChips.length = 0;
    draft.switchCameraButtons.length = 0;
  });

  addNewClassCard(handlers);
  updateExampleCounts(true);
}

export function lockCapturePanels() {
  state.webcamPanels.forEach((panel) => panel.classList.remove('visible'));
  if (CAPTURE_VIDEO) {
    CAPTURE_VIDEO.classList.add('hidden');
  }
  state.openWebcamButtons.forEach((btn) => (btn.disabled = true));
  state.dataCollectorButtons.forEach((btn) => (btn.disabled = true));
}

export function unlockCapturePanels() {
  state.openWebcamButtons.forEach((btn) => (btn.disabled = false));
  state.dataCollectorButtons.forEach((btn) => (btn.disabled = false));
  if (CAPTURE_VIDEO) {
    CAPTURE_VIDEO.classList.remove('hidden');
  }
}

export function updateClassCardCopy({
  openButtonLabel = DEFAULT_CAPTURE_LABEL,
  panelLabel = DEFAULT_CAPTURE_LABEL,
  collectorLabel = DEFAULT_COLLECT_LABEL,
} = {}) {
  state.openWebcamButtons.forEach((btn) => {
    if (btn) btn.textContent = openButtonLabel;
  });
  state.webcamPanels.forEach((panel) => {
    const labelEl = panel?.querySelector('.panel-top span');
    if (labelEl) labelEl.textContent = panelLabel;
  });
  state.dataCollectorButtons.forEach((btn) => {
    if (btn) btn.textContent = collectorLabel;
  });
}
