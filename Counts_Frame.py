import numpy as np
import os
import cv2

def get_frame_stats(folder):
    lengths = []
    for cls in os.listdir(folder):
        cls_path = os.path.join(folder, cls)
        if not os.path.isdir(cls_path): continue
        for f in os.listdir(cls_path):
            if f.endswith(".mp4"):
                cap = cv2.VideoCapture(os.path.join(cls_path, f))
                lengths.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                cap.release()
    print(f"Trung bình: {np.mean(lengths):.1f} frames | Trung vị: {np.median(lengths):.1f} | Max: {np.max(lengths)}")
    return lengths

lengths = get_frame_stats("Data/train")
max_frames = min( int(np.percentile(lengths, 95)), 160 )
print(max_frames)