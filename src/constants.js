export const MOBILE_NET_INPUT_WIDTH = 224;
export const MOBILE_NET_INPUT_HEIGHT = 224;
export const STOP_DATA_GATHER = -1;
export const ARDUINO_SEND_THRESHOLD = 0.6;
export const ARDUINO_SEND_COOLDOWN_MS = 500;
export const DEFAULT_TRAINING_EPOCHS = 10;
export const DEFAULT_TRAINING_BATCH_SIZE = 5;
export const DEFAULT_TRAINING_LR = 0.001;
export const SUPPORTED_MODES = ['image', 'gesture', 'face', 'audio', 'pose'];

export const BAR_COLORS = [
  ['#f07818', '#ffd8ba'],
  ['#d14ebd', '#ffd6f4'],
  ['#5067ff', '#d4ddff'],
  ['#28b88a', '#c8f1e3'],
  ['#f2b134', '#ffe7bd'],
  ['#8e54e9', '#e3d6ff'],
];

export const MODE_NAMES = {
  image: 'Bildklassifikation',
  gesture: 'Gestenerkennung',
  face: 'Gesichtserkennung',
  audio: 'Audioerkennung',
  pose: 'Posen-Erkennung',
};

export const GESTURE_FEATURE_SIZE = 63; // 21 points * 3 coords
export const GESTURE_SAMPLE_INTERVAL_MS = 120;
export const POSE_FEATURE_SIZE = 99; // 33 points * 3 coords
export const POSE_SAMPLE_INTERVAL_MS = 120;

export const CLASS_DEFAULT_PREFIX = 'Class';
export const DEFAULT_COLLECT_LABEL = 'Zum Aufnehmen halten';
export const DEFAULT_CAPTURE_LABEL = 'Webcam';
export const POSE_CAPTURE_LABEL = 'Kamera';
export const AUDIO_COLLECT_LABEL = 'Aufnehmen';
export const AUDIO_CAPTURE_LABEL = 'Audio';
