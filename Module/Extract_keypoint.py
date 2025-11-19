import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Cấu hình MediaPipe Holistic
mp_holistic = mp.solutions.holistic

# Hàm trích xuất keypoint compact 
def extract_holistic_keypoints_compact(results):
    keypoints = []
    # 1. Left Hand (63 features)
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
    else:
        keypoints.extend([0.0] * 63)
    # 2. Right Hand (63 features)
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
    else:
        keypoints.extend([0.0] * 63)
    # 3. Upper Body Pose (chỉ lấy vai, khuỷu tay)
    upper_body_indices = [11, 12, 13, 14]
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
        for idx in upper_body_indices:
            if idx < len(pose_landmarks):
                landmark = pose_landmarks[idx]
                keypoints.extend([landmark.x, landmark.y, landmark.z])
            else:
                keypoints.extend([0.0] * 12)
    else:
        keypoints.extend([0.0] * (len(upper_body_indices)*4))
    # 4. Face Key Points - chỉ lấy điểm trung tâm mặt (landmark 0)
    if results.face_landmarks:
        face_landmarks = results.face_landmarks.landmark
        if len(face_landmarks) > 0:
            landmark = face_landmarks[0]
            keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            keypoints.extend([0.0] * 3)
    else:
        keypoints.extend([0.0] * 3)
    
    print(len(keypoints))
    return np.array(keypoints, dtype=np.float32)


# Hàm xử lý toàn bộ dataset với holistic keypoint compact
def process_dataset(input_root="Data", output_root="test"):
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
            videos = [v for v in os.listdir(input_label_dir) if v.endswith(".mp4")]
            for video_name in tqdm(videos, desc=f"{label}", leave=False):
                video_path = os.path.join(input_label_dir, video_name)
                save_path = os.path.join(output_label_dir, video_name.replace(".mp4", ".npy"))
                cap = cv2.VideoCapture(video_path)
                seq = []
                with mp_holistic.Holistic(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    model_complexity=1
                ) as holistic:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = holistic.process(image_rgb)
                        keypoints = extract_holistic_keypoints_compact(results)
                        seq.append(keypoints)
                cap.release()
                if len(seq) == 0:
                    print(f"Video rỗng hoặc không có keypoint: {video_name}")
                    continue
                seq = np.array(seq, dtype=np.float32)
                np.save(save_path, seq)
        print(f"Hoàn tất tập '{split}' → lưu tại: {output_split_dir}")

if __name__ == "__main__":
    input_root = input("Nhập thư mục dữ liệu gốc (vd: Data): ").strip()
    process_dataset(input_root)
    print("\nHoàn tất trích xuất keypoints!")
