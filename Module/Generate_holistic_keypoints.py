"""
Script để xử lý lại dataset từ video thành holistic keypoints
Kết hợp Hand + Pose + Face landmarks
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
from tqdm.auto import tqdm
import json

# Import từ Holistic_detector
import sys
sys.path.append(os.path.dirname(__file__))
from Holistic_detector import extract_holistic_keypoints_compact


def process_video_to_keypoints(video_path, holistic_model, use_compact=True):
    """
    Xử lý một video thành sequence keypoints
    
    Args:
        video_path: Đường dẫn video
        holistic_model: MediaPipe Holistic model
        use_compact: Dùng phiên bản compact (272 features) hay full
        
    Returns:
        keypoints_array: np.ndarray shape (num_frames, num_features)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return None
    
    keypoints_sequence = []
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        # Process
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic_model.process(image_rgb)
        
        # Extract keypoints
        if use_compact:
            keypoints = extract_holistic_keypoints_compact(results)
        else:
            from Holistic_detector import extract_holistic_keypoints
            keypoints = extract_holistic_keypoints(results, include_face=False, include_pose_upper=True)
        
        keypoints_sequence.append(keypoints)
    
    cap.release()
    
    if not keypoints_sequence:
        return None
    
    return np.array(keypoints_sequence, dtype=np.float32)


def process_dataset(input_root, output_root, use_compact=True, model_complexity=1):
    """
    Xử lý toàn bộ dataset từ videos thành keypoints
    
    Args:
        input_root: Thư mục chứa videos (ví dụ: Data/test)
        output_root: Thư mục output (ví dụ: Data_Keypoints_Holistic)
        use_compact: Dùng phiên bản compact không
        model_complexity: 0 (lite), 1 (full), 2 (heavy)
    
    Cấu trúc:
        input_root/
            train/
                ClassName1/
                    0.mp4
                    1.mp4
                ClassName2/
                    ...
            val/
                ...
    """
    input_path = Path(input_root)
    output_path = Path(output_root)
    
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        return
    
    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    
    stats = {
        'total_videos': 0,
        'processed': 0,
        'failed': 0,
        'total_frames': 0
    }
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=model_complexity
    ) as holistic:
        
        # Duyệt qua train/val splits
        for split in ['train', 'val']:
            split_input = input_path / split
            split_output = output_path / split
            
            if not split_input.exists():
                print(f"Skipping {split} (not found)")
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing {split} split")
            print(f"{'='*60}")
            
            # Duyệt qua các class
            class_dirs = [d for d in split_input.iterdir() if d.is_dir()]
            
            for class_dir in tqdm(class_dirs, desc=f"{split} classes"):
                class_name = class_dir.name
                output_class_dir = split_output / class_name
                output_class_dir.mkdir(parents=True, exist_ok=True)
                
                # Duyệt qua các video trong class
                video_files = list(class_dir.glob('*.mp4')) + list(class_dir.glob('*.avi'))
                
                for video_file in tqdm(video_files, desc=f"  {class_name}", leave=False):
                    stats['total_videos'] += 1
                    
                    # Tên output file (giữ nguyên tên, đổi extension)
                    output_file = output_class_dir / f"{video_file.stem}.npy"
                    
                    # Bỏ qua nếu đã xử lý
                    if output_file.exists():
                        stats['processed'] += 1
                        continue
                    
                    # Process video
                    keypoints = process_video_to_keypoints(str(video_file), holistic, use_compact)
                    
                    if keypoints is not None and len(keypoints) > 0:
                        # Lưu ra file .npy
                        np.save(output_file, keypoints)
                        stats['processed'] += 1
                        stats['total_frames'] += len(keypoints)
                    else:
                        stats['failed'] += 1
                        print(f"\n  ⚠️ Failed to process: {video_file}")
    
    # Save statistics
    stats_file = output_path / 'processing_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total videos: {stats['total_videos']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total frames extracted: {stats['total_frames']}")
    print(f"Average frames per video: {stats['total_frames']/stats['processed']:.1f}")
    print(f"\nOutput saved to: {output_path}")
    print(f"Stats saved to: {stats_file}")


def compare_keypoint_dimensions():
    """
    So sánh số features giữa các phương pháp
    """
    print("="*60)
    print("KEYPOINT DIMENSIONS COMPARISON")
    print("="*60)
    
    methods = {
        "Hands only (current)": {
            "Left hand": 21 * 3,  # x, y, z
            "Right hand": 21 * 3,
            "Total": 126
        },
        "Hands with visibility": {
            "Left hand": 21 * 4,  # x, y, z, visibility
            "Right hand": 21 * 4,
            "Total": 168
        },
        "Holistic Compact": {
            "Left hand": 21 * 4,
            "Right hand": 21 * 4,
            "Upper pose": 11 * 4,
            "Face key points": 20 * 3,
            "Total": 272
        },
        "Holistic Full (no face)": {
            "Left hand": 21 * 4,
            "Right hand": 21 * 4,
            "Upper pose": 11 * 4,
            "Total": 212
        },
        "Holistic Full (with face)": {
            "Left hand": 21 * 4,
            "Right hand": 21 * 4,
            "Upper pose": 11 * 4,
            "Face (all)": 468 * 3,
            "Total": 1616
        }
    }
    
    for method_name, components in methods.items():
        print(f"\n{method_name}:")
        total = components.pop('Total')
        for comp_name, comp_size in components.items():
            print(f"  - {comp_name}: {comp_size} features")
        print(f"  → Total: {total} features")


if __name__ == "__main__":
    # So sánh dimensions
    compare_keypoint_dimensions()
    
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    print("\n# Xử lý dataset từ videos:")
    print("python Generate_holistic_keypoints.py")
    print("\n# Hoặc trong code:")
    print("""
    from Generate_holistic_keypoints import process_dataset
    
    process_dataset(
        input_root='Data/test',
        output_root='Data_Keypoints_Holistic',
        use_compact=True,  # Dùng 272 features
        model_complexity=1  # 0: fast, 1: balanced, 2: accurate
    )
    """)
    
    # Uncomment dòng dưới để chạy thực tế
    # process_dataset(
    #     input_root='Data/test',
    #     output_root='Data_Keypoints_Holistic',
    #     use_compact=True,
    #     model_complexity=1
    # )
