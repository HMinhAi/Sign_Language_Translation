import os
import numpy as np
import random
from typing import List, Dict

def load_keypoints(file_path):
    """Tải keypoint numpy từ file (ví dụ .npy)"""
    return np.load(file_path)

def generate_ctc_pairs(data_dir: str, max_signs: int = 3):
    """
    Ghép ngẫu nhiên 2–3 video isolated thành một chuỗi.
    Return: (concat_seq, label_seq)
    """
    sign_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    label_names = [os.path.splitext(f)[0] for f in sign_files]

    samples = []
    for _ in range(len(sign_files)):
        # Chọn ngẫu nhiên 2–3 ký hiệu khác nhau
        selected = random.sample(sign_files, k=random.randint(2, max_signs))
        seqs = [load_keypoints(os.path.join(data_dir, s)) for s in selected]

        # Ghép chuỗi keypoint theo thời gian
        concat_seq = np.concatenate(seqs, axis=0)

        # Tạo nhãn số (theo index)
        label_seq = [label_names.index(os.path.splitext(s)[0]) for s in selected]

        samples.append({"keypoints": concat_seq, "labels": label_seq})

    return samples


if __name__ == "__main__":
    data = generate_ctc_pairs("data/isolated_keypoints/")
    print(f"Sinh {len(data)} chuỗi giả lập, mỗi chuỗi có {len(data[0]['labels'])} ký hiệu")
    print("Ví dụ:", data[0]['labels'], data[0]['keypoints'].shape)
