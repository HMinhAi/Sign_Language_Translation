"""
Holistic Detector - Kết hợp Hand, Pose và Face landmarks từ MediaPipe
Sử dụng cho Sign Language Recognition với đầy đủ thông tin cơ thể
"""

import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def extract_holistic_keypoints(results, include_face=True, include_pose_upper=True):
    """
    Trích xuất keypoints từ MediaPipe Holistic results
    
    Args:
        results: MediaPipe Holistic results
        include_face: Có bao gồm face landmarks không
        include_pose_upper: Có bao gồm upper body pose không
        
    Returns:
        keypoints: np.ndarray với cấu trúc:
            - Left hand: 21 landmarks × (x, y, z, visibility) = 84 features
            - Right hand: 21 landmarks × (x, y, z, visibility) = 84 features
            - Pose (upper): 11 landmarks × (x, y, z, visibility) = 44 features
            - Face (optional): 468 landmarks × (x, y, z) = 1404 features (nếu include_face=True)
        
        Total: 212 features (không có face) hoặc 1616 features (có face)
    """
    keypoints = []
    
    # 1. Left Hand Landmarks (21 points × 4 = 84 features)
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    else:
        keypoints.extend([0.0] * 84)  # Padding nếu không detect được
    
    # 2. Right Hand Landmarks (21 points × 4 = 84 features)
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    else:
        keypoints.extend([0.0] * 84)
    
    # 3. Upper Body Pose Landmarks (vai, khuỷu tay, cổ tay, mũi, mắt, tai)
    # Chỉ lấy upper body để giảm noise từ lower body
    if include_pose_upper and results.pose_landmarks:
        # Indices cho upper body:
        # 0: nose, 1-2: eyes, 3-4: ears
        # 11-12: shoulders, 13-14: elbows, 15-16: wrists
        upper_body_indices = [11, 12, 13, 14]
        
        pose_landmarks = results.pose_landmarks.landmark
        for idx in upper_body_indices:
            if idx < len(pose_landmarks):
                landmark = pose_landmarks[idx]
                keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            else:
                keypoints.extend([0.0] * 4)
    elif include_pose_upper:
        keypoints.extend([0.0] * 44)  # 11 landmarks × 4
    
    # 4. Face Landmarks (optional - 468 points × 3 = 1404 features)
    # Lưu ý: Face landmarks rất nhiều, có thể làm model chậm
    # Nên cân nhắc chỉ lấy một số điểm quan trọng (mắt, miệng, lông mày)
    if include_face:
        if results.face_landmarks:
            # Có thể chọn lấy tất cả 468 điểm hoặc chỉ một số điểm quan trọng
            # Ví dụ này lấy tất cả (có thể tốn bộ nhớ)
            for landmark in results.face_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            keypoints.extend([0.0] * 1404)  # 468 × 3
    
    return np.array(keypoints, dtype=np.float32)


def extract_holistic_keypoints_compact(results):
    """
    Phiên bản compact - chỉ lấy các điểm quan trọng nhất
    
    Returns:
        keypoints: np.ndarray với cấu trúc:
            - Left hand: 21 × 3 = 63 features
            - Right hand: 21 × 3 = 63 features
            - Pose upper: 4 × 3 = 12 features
            - Face key points: 1 × 3 = 3 features (điểm giữa của mặt)
        
        Total: 141 features
    """
    keypoints = []
    
    # 1. Left Hand (84 features)
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
    else:
        keypoints.extend([0.0] *63)
    
    # 2. Right Hand (84 features)
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
    else:
        keypoints.extend([0.0] * 63)
    
    # 3. Upper Body Pose (chỉ lấy vai, khuỷu tay)
    # Indices: 11-left shoulder, 12-right shoulder, 13-left elbow, 14-right elbow
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
        keypoints.extend([0.0] * (len(upper_body_indices)*3))
    
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
    # Để giữ shape nhất quán với các sample khác, có thể pad thêm cho đủ số chiều nếu cần
    return np.array(keypoints, dtype=np.float32)


# Demo code - chạy với camera hoặc video
if __name__ == "__main__":
    # Có thể đổi thành đường dẫn video hoặc 0 cho camera
    video_path = 0  # Camera
    # video_path = r"Data\train\I\0.mp4"
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video source: {video_path}")
        exit()
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}")
    print(f"Total frames: {frame_count if frame_count > 0 else 'Live stream'}")
    
    frame_number = 0
    keypoints_sequence = []  # Lưu sequence keypoints
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
        model_complexity=1  # 0: lite, 1: full, 2: heavy
    ) as holistic:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                if frame_number == 0:
                    print("Failed to read frame")
                break
            
            frame_number += 1
            
            # Process image
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            # Extract keypoints (sử dụng phiên bản compact)
            keypoints = extract_holistic_keypoints_compact(results)
            print(keypoints)
            keypoints_sequence.append(keypoints)
            
            # Draw only the keypoints selected in extract_holistic_keypoints_compact
            image.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            h, w, _ = image.shape
            kp = keypoints
            # print(f"[DEBUG] keypoints shape: {kp.shape}")
            # Left hand: 21x4 (x, y, z, vis)
            if len(kp) >= 21*4:
                for i in range(21):
                    idx = i*4
                    if idx+1 < len(kp):
                        x, y = kp[idx], kp[idx+1]
                        if x > 0 and y > 0:
                            cx, cy = int(x * w), int(y * h)
                            cv2.circle(image, (cx, cy), 4, (0, 255, 255), -1)
                            cv2.putText(image, str(i), (cx+4, cy-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1, cv2.LINE_AA)
            # Right hand: 21x4 (x, y, z, vis)
            offset = 21*4
            if len(kp) >= offset + 21*4:
                for i in range(21):
                    idx = offset + i*4
                    if idx+1 < len(kp):
                        x, y = kp[idx], kp[idx+1]
                        if x > 0 and y > 0:
                            cx, cy = int(x * w), int(y * h)
                            cv2.circle(image, (cx, cy), 4, (255, 255, 0), -1)
                            cv2.putText(image, str(i), (cx+4, cy-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1, cv2.LINE_AA)
            # Pose upper: chỉ vẽ vai, khuỷu tay
            offset = 21*4*2
            upper_body_names = [
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow'
            ]
            if len(kp) >= offset + 4*4:
                for i in range(4):
                    idx = offset + i*4
                    if idx+1 < len(kp):
                        x, y = kp[idx], kp[idx+1]
                        if x > 0 and y > 0:
                            cx, cy = int(x * w), int(y * h)
                            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
                            label = upper_body_names[i] if i < len(upper_body_names) else str(i)
                            cv2.putText(image, label, (cx+4, cy-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv2.LINE_AA)
            # Face: 1x3 (x, y, z)
            offset = 21*4*2 + 4*4
            if len(kp) >= offset + 3:
                x, y = kp[offset], kp[offset+1]
                if x > 0 and y > 0:
                    cx, cy = int(x * w), int(y * h)
                    cv2.circle(image, (cx, cy), 6, (0, 0, 255), -1)
                    cv2.putText(image, '0', (cx+6, cy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            
            # Display info
            cv2.putText(image, f'Frame: {frame_number}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f'Keypoints: {len(keypoints)} dims', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show main image
            cv2.imshow('MediaPipe Holistic', image)
            
            # Exit on ESC or window close
            key = cv2.waitKey(5) & 0xFF
            if key == 27:
                print("\nStopped by user")
                break
            elif cv2.getWindowProperty('MediaPipe Holistic', cv2.WND_PROP_VISIBLE) < 1:
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print(f"\nTotal frames processed: {frame_number}")
    if keypoints_sequence:
        keypoints_array = np.array(keypoints_sequence)
        print(f"Keypoints sequence shape: {keypoints_array.shape}")
        print(f"Shape: (num_frames={keypoints_array.shape[0]}, num_features={keypoints_array.shape[1]})")
        
        # Có thể lưu ra file .npy
        # np.save('output_keypoints.npy', keypoints_array)
        # print("Saved to output_keypoints.npy")
