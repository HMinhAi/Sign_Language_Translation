import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque

# Cấu hình
MODEL_PATH = "models/phase1/final_model.h5"
LABELS_PATH = "models/phase1/labels.txt"    
MAX_FRAMES = 180
FEATURE_DIM = 128
CONF_THRESHOLD = 0.5

# Hàm load labels
def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f.readlines()]

# Hàm trích keypoints 3D 2 tay + mask
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

def extract_dual_hand_keypoints_3d(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    left_hand = np.zeros(63)
    right_hand = np.zeros(63)
    left_mask, right_mask = 0.0, 0.0

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, handedness in enumerate(results.multi_handedness):
            label = handedness.classification[0].label  # 'Left' hoặc 'Right'
            landmarks = results.multi_hand_landmarks[idx]
            coords = []
            for lm in landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            if label == 'Left':
                left_hand = np.array(coords)
                left_mask = 1.0
            else:
                right_hand = np.array(coords)
                right_mask = 1.0
    return np.concatenate([left_hand, right_hand, [left_mask, right_mask]])

# Hàm chuẩn hoá + pad/cut sequence
def prepare_sequence(buffer, max_frames=MAX_FRAMES):
    seq = np.array(buffer, dtype=np.float32)
    # pad/cut về đúng độ dài
    if len(seq) < max_frames:
        pad = np.zeros((max_frames - len(seq), FEATURE_DIM))
        seq = np.concatenate([seq, pad], axis=0)
    elif len(seq) > max_frames:
        seq = seq[-max_frames:]
    # chuẩn hoá theo video (giống train)
    mean = np.mean(seq[:, :-2], axis=(0, 1), keepdims=True)
    std = np.std(seq[:, :-2], axis=(0, 1), keepdims=True) + 1e-6
    seq[:, :-2] = (seq[:, :-2] - mean) / std
    return seq[np.newaxis, ...]   # shape (1, T, D)

# Main
def main():
    model = load_model(MODEL_PATH)
    labels = load_labels(LABELS_PATH)
    cap = cv2.VideoCapture(0)

    buffer = deque(maxlen=MAX_FRAMES)
    font = cv2.FONT_HERSHEY_SIMPLEX

    print("Bắt đầu camera... Nhấn Q để thoát.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = extract_dual_hand_keypoints_3d(frame)
        buffer.append(keypoints)

        if len(buffer) >= 20:  # đợi đủ vài frame
            seq = prepare_sequence(buffer)
            pred = model.predict(seq, verbose=0)[0]
            idx = np.argmax(pred)
            conf = pred[idx]

            # hiển thị xác suất cao nhất và top-3
            top3 = np.argsort(pred)[-3:][::-1]
            y0 = 40
            for i, j in enumerate(top3):
                text = f"{labels[j]}: {pred[j]*100:.1f}%"
                color = (0,255,0) if i==0 else (255,255,255)
                cv2.putText(frame, text, (10, y0 + 30*i),
                            font, 0.8, color, 2, cv2.LINE_AA)

        cv2.imshow("Sign Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
