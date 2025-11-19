import cv2, os, json, time, torch
import numpy as np
import mediapipe as mp
import torch.nn.functional as F
from collections import deque
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=138, d_model=256, nhead=4, num_layers=3):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=512, dropout=0.15, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.encoder(x)
        return x


class TransformerCTC(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        return self.fc_out(x)

# ===== CONFIG =====
MODEL_PATH = "continuous_best.pt"
CLASS_MAP = "Data_keypoints_holistic/class_names.json"
NUM_KEYPOINTS = 141
INPUT_DIM = 138
MAX_FRAMES = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.5

# ===== Load model =====
with open(CLASS_MAP, "r") as f:
    classes = json.load(f)["classes"]
num_classes = len(classes) + 1  # +1 blank
blank_idx = num_classes - 1

encoder = TransformerEncoder(input_dim=INPUT_DIM, d_model=256, nhead=4, num_layers=3)
model = TransformerCTC(encoder, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE).eval()

# ===== MediaPipe setup =====
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ===== Helper functions =====
def extract_keypoints(results):
    keypoints = []
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark: keypoints += [lm.x, lm.y, lm.z]
    else: keypoints += [0.]*63
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark: keypoints += [lm.x, lm.y, lm.z]
    else: keypoints += [0.]*63
    pose_idx = [11,12,13,14]
    if results.pose_landmarks:
        for i in pose_idx: 
            lm = results.pose_landmarks.landmark[i]
            keypoints += [lm.x, lm.y, lm.z]
    else: keypoints += [0.]*12
    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[0]
        keypoints += [nose.x, nose.y, nose.z]
    else: keypoints += [0.]*3
    return np.array(keypoints, np.float32)

def normalize_keypoints(kp):
    face_center = kp[:,138:]
    main = kp[:,:138].reshape(kp.shape[0],-1,3)
    main -= face_center[:,None,:]
    return main.reshape(kp.shape[0],-1)

def greedy_decode(log_probs):
    preds = log_probs.argmax(-1)
    outs=[]
    for seq in preds:
        prev=-1; tmp=[]
        for p in seq.cpu().numpy():
            if p!=prev and p!=blank_idx: tmp.append(int(p))
            prev=p
        outs.append(tmp)
    return outs

# ===== Main loop =====
cap = cv2.VideoCapture(0)
buf = deque(maxlen=MAX_FRAMES)
fps_time=time.time(); frame_count=0; fps=0
prediction=""; conf=0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame=cv2.flip(frame,1)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res=holistic.process(rgb)
    kp=extract_keypoints(res)
    buf.append(kp)

    if len(buf)>=60:   # đủ frame để dự đoán
        seq=np.array(buf,dtype=np.float32)
        seq=normalize_keypoints(seq)
        x=torch.tensor(seq[None,:,:INPUT_DIM],dtype=torch.float32,device=DEVICE)
        with torch.no_grad():
            logits=model(x)
            log_probs=F.log_softmax(logits,dim=-1)
            decoded=greedy_decode(log_probs)
            print(decoded)
        if decoded and decoded[0]:
            ids=[i for i in decoded[0] if i<len(classes)]
            if ids:
                names=[classes[i] for i in ids]
                prediction=" ".join(names)
    
    # hiển thị FPS
    frame_count+=1
    if frame_count>=30:
        now=time.time(); fps=frame_count/(now-fps_time); fps_time=now; frame_count=0
    cv2.putText(frame,f"{fps:.1f} FPS",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(frame,f"{prediction}",(10,70),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.imshow("Sign Language Recognition (CTC)",frame)

    if cv2.waitKey(1)&0xFF==ord('q'): break

cap.release(); cv2.destroyAllWindows(); holistic.close()
