import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Cấu hình MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# Hàm trích keypoints 3D ổn định cho 2 tay
def extract_dual_hand_keypoints_3d(frame):
    """
    Trích keypoints 3D ổn định thứ tự trái/phải.
    Mỗi frame -> vector (63 + 63 + 2) = 128 chiều
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    left_hand = np.zeros(63)   # 21 keypoints × (x, y, z)
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

    # Ghép 2 tay + mask => vector 128 chiều
    return np.concatenate([left_hand, right_hand, [left_mask, right_mask]])

# Hàm xử lý toàn bộ dataset
def process_dataset(input_root="Data", output_root="Data_keypoints"):
    """
    Duyệt toàn bộ Data/{train,val,test}/{label}/*.mp4
    -> Trích keypoints 3D + mask -> lưu .npy cùng cấu trúc nhãn.
    """
    os.makedirs(output_root, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(input_root, split)
        if not os.path.exists(split_dir):
            print(f"Bỏ qua '{split}' (không tồn tại).")
            continue

        print(f"\nĐang xử lý tập '{split}' ...")
        output_split_dir = os.path.join(output_root, split)
        os.makedirs(output_split_dir, exist_ok=True)

        labels = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        for label in tqdm(labels, desc=f"Đang xử lý nhãn ({split})"):
            input_label_dir = os.path.join(split_dir, label)
            output_label_dir = os.path.join(output_split_dir, label)
            os.makedirs(output_label_dir, exist_ok=True)

            for video_name in os.listdir(input_label_dir):
                if not video_name.endswith(".mp4"):
                    continue

                video_path = os.path.join(input_label_dir, video_name)
                save_path = os.path.join(output_label_dir, video_name.replace(".mp4", ".npy"))

                cap = cv2.VideoCapture(video_path)
                seq = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    keypoints = extract_dual_hand_keypoints_3d(frame)
                    seq.append(keypoints)
                cap.release()

                if len(seq) == 0:
                    print(f"Video rỗng hoặc không có tay: {video_name}")
                    continue

                seq = np.array(seq, dtype=np.float32)
                np.save(save_path, seq)

        print(f"Hoàn tất tập '{split}' → lưu tại: {output_split_dir}")

# CHẠY TRỰC TIẾP
if __name__ == "__main__":
    input_root = input("Nhập thư mục dữ liệu gốc (vd: Data): ").strip()
    output_root = input("Thư mục lưu keypoints (vd: Data_keypoints): ").strip()

    process_dataset(input_root, output_root)
    print("\nHoàn tất trích xuất keypoints!")
