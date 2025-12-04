export const MOBILE_NET_INPUT_WIDTH = 224;
export const MOBILE_NET_INPUT_HEIGHT = 224;
export const STOP_DATA_GATHER = -1;
export const ARDUINO_SEND_THRESHOLD = 0.6;
export const ARDUINO_SEND_COOLDOWN_MS = 500;

export const BAR_COLORS = [
  ['#f07818', '#ffd8ba'],
  ['#d14ebd', '#ffd6f4'],
  ['#5067ff', '#d4ddff'],
  ['#28b88a', '#c8f1e3'],
  ['#f2b134', '#ffe7bd'],
  ['#8e54e9', '#e3d6ff'],
];

export const GESTURE_LABELS = [
  'None',
  'Closed_Fist',
  'Open_Palm',
  'Pointing_Up',
  'Thumb_Down',
  'Thumb_Up',
  'Victory',
  'ILoveYou',
];

export const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [5, 17],
];

export const MODE_NAMES = {
  image: 'Bildklassifikation',
  gesture: 'Gesture Recognition',
};
