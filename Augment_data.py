import os
import cv2
import random
import numpy as np
import shutil

print("\n--- Bắt đầu quá trình Video Augmentation ---")

# Các hàm augment Videos

def change_speed(video_frames, speed_factor):
    frame_count = len(video_frames)
    new_length = max(1, int(frame_count / speed_factor))
    indices = np.linspace(0, frame_count - 1, new_length).astype(int)
    return [video_frames[i] for i in indices]

def random_crop(video_frames, crop_percent=0.1):
    n = len(video_frames)
    remove = int(n * crop_percent)
    start = random.randint(0, remove)
    end = n - (remove - start)
    return video_frames[start:end]

def trim_start(video_frames, trim_ratio=0.1):
    trim = int(len(video_frames) * trim_ratio)
    return video_frames[trim:] if trim < len(video_frames) else video_frames

def trim_end(video_frames, trim_ratio=0.1):
    trim = int(len(video_frames) * trim_ratio)
    return video_frames[:-trim] if trim > 0 else video_frames

def random_middle(video_frames, keep_ratio=0.85):
    n = len(video_frames)
    keep = int(n * keep_ratio)
    start = random.randint(0, n - keep)
    return video_frames[start:start + keep]

# Danh sách 6 kiểu Augment
augmentations = [
    ("Fast", lambda f: change_speed(f, random.uniform(1.1, 1.2))),
    ("Slow", lambda f: change_speed(f, random.uniform(0.8, 0.9)))
    # ("RandomCrop", lambda f: random_crop(f, random.uniform(0.05, 0.1))),
    # ("RandomMiddle", lambda f: random_middle(f, random.uniform(0.8, 0.9))),
    # ("TrimStart", lambda f: trim_start(f, random.uniform(0.1, 0.15))),
    # ("TrimEnd", lambda f: trim_end(f, random.uniform(0.1, 0.15)))
]

# Hàm chính

def augment_videos_in_folder(root_dir, delete_aug_folder=True):
    """Tạo augment, copy sang thư mục nhãn, và đổi tên toàn bộ."""
    print(f"\nĐang augment trong thư mục: {root_dir}")

    for label_folder in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label_folder)
        if not os.path.isdir(label_path):
            continue

        print(f"\nNhãn: {label_folder}")

        aug_folder = os.path.join(label_path, "augmented")
        os.makedirs(aug_folder, exist_ok=True)

        video_files = [f for f in os.listdir(label_path)
                       if f.endswith(".mp4") and not f.startswith(".")]
        video_files.sort()

        for file_name in video_files:
            file_path = os.path.join(label_path, file_name)

            cap = cv2.VideoCapture(file_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            if len(frames) < 5:
                print(f"Bỏ qua {file_name} (video quá ngắn)")
                continue

            # Chọn ngẫu nhiên 1 kiểu augment
            aug_name, aug_func = random.choice(augmentations)
            aug_frames = aug_func(frames)

            # Lưu bản augment
            base_name = os.path.splitext(file_name)[0]
            new_filename = f"{base_name}_{aug_name}.mp4"
            out_path = os.path.join(aug_folder, new_filename)

            h, w, _ = aug_frames[0].shape
            fps = 25
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            for frame in aug_frames:
                out.write(frame)
            out.release()

            print(f"{file_name} → {aug_name}")

        # Sau khi augment xong: copy sang folder gốc và đổi tên
        merge_and_rename(label_path, aug_folder)

        # Tuỳ chọn: xoá folder tạm
        if delete_aug_folder:
            shutil.rmtree(aug_folder)
            print(f"Đã xoá thư mục tạm '{aug_folder}'.")

def merge_and_rename(label_path, aug_folder):
    """Copy tất cả video augment sang thư mục nhãn và đổi tên toàn bộ."""
    aug_videos = [f for f in os.listdir(aug_folder) if f.endswith(".mp4")]

    for f in aug_videos:
        src = os.path.join(aug_folder, f)
        dst = os.path.join(label_path, f)
        shutil.copy(src, dst)

    # Đổi tên toàn bộ file (gốc + augment)
    rename_videos(label_path)
    print(f"Đã gộp {len(aug_videos)} file augment sang '{label_path}'.")

def rename_videos(folder_path):
    """Đổi tên toàn bộ video trong folder theo thứ tự 0.mp4, 1.mp4, ..."""
    video_files = [f for f in os.listdir(folder_path)
                   if f.endswith(".mp4") and not f.startswith(".")]
    video_files.sort()
    for i, filename in enumerate(video_files):
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, f"{i}.mp4")
        os.rename(old_path, new_path)
    print(f"Đã đổi tên {len(video_files)} file trong '{os.path.basename(folder_path)}'.")

# Run

if __name__ == "__main__":
    target_folder = input("Nhập đường dẫn thư mục cần augment (vd: data/train): ").strip()
    augment_videos_in_folder(target_folder, delete_aug_folder=True)
    print("\nHoàn tất: augment + copy + rename thành công!")
