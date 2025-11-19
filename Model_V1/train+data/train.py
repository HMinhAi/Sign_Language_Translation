import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

#  PARAMETERS 
CLASS_NAMES = ['Accept', 'Buy', 'Call', 'Candy', 'Catch', 'Deaf', 'Everyone','Food', 'Give', 'Green', 
               'Help', 'Hungry', 'I','Learn', 'Light-blue','Like', 'Milk', 'Music', 'Name', 'Red',
               'Ship', 'Son', 'Thanks','Want', 'Water', 'Where', 'Women', 'Yellow', 'Yogurt','You']

DATA_DIR = r"lsa64_keypoints"
MAX_FRAMES = 60
BATCH_SIZE = 16
EPOCHS = 40
LR = 5e-5
INPUT_DIM = 2*21*3  # 126
EMBED_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 4
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  DATASET 
class SignDataset(Dataset):
    def __init__(self, file_list, labels, max_frames=MAX_FRAMES):
        self.files = file_list
        self.labels = labels
        self.max_frames = max_frames

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        kp = np.load(self.files[idx])  # (num_frames,2,21,3)
        num_frames = kp.shape[0]
        kp = kp.reshape(num_frames, -1)  # (num_frames,126)

        # pad hoặc cut
        if num_frames < self.max_frames:
            pad = np.zeros((self.max_frames - num_frames, kp.shape[1]), dtype=np.float32)
            kp = np.vstack([kp, pad])
        elif num_frames > self.max_frames:
            kp = kp[:self.max_frames]

        label = self.labels[idx]
        return torch.tensor(kp, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

#  LOAD FILES 
file_list = []
labels = []

for idx, c in enumerate(CLASS_NAMES):
    class_folder = os.path.join(DATA_DIR, c)
    if not os.path.exists(class_folder):
        continue
    files = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if f.endswith(".npy")]
    file_list.extend(files)
    labels.extend([idx]*len(files))

file_list = np.array(file_list)
labels = np.array(labels)

train_files, temp_files, train_labels, temp_labels = train_test_split(
    file_list, labels, test_size=0.3, stratify=labels, random_state=42
)
val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

train_dataset = SignDataset(train_files, train_labels)
val_dataset = SignDataset(val_files, val_labels)
test_dataset = SignDataset(test_files, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- MODEL ----------------
class SignTransformer(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                 num_layers=NUM_LAYERS, num_classes=NUM_CLASSES, class_names=CLASS_NAMES, dropout=0.3):
        super(SignTransformer, self).__init__()
        self.class_names = class_names
        self.input_fc = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.input_fc(x)             # (batch, seq_len, embed_dim)
        x = self.transformer(x)          # (batch, seq_len, embed_dim)
        x = x.mean(dim=1)                # mean pooling
        out = self.cls_head(x)           # logits

        return out

model = SignTransformer().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
print(DEVICE)
print('='*60)

#  TRAIN 
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_correct = 0
    for X, y in trainkk_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out_logits = model.input_fc(X)   # forward phần embedding
        out_logits = model.transformer(out_logits)
        out_logits = out_logits.mean(dim=1)
        logits = model.cls_head(out_logits)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
    train_acc = total_correct / len(train_dataset)
    train_loss = total_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

#  TEST 
model.eval()
test_correct = 0
with torch.no_grad():
    for X_test, y_test in test_loader:
        X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
        out_logits = model.input_fc(X_test)
        out_logits = model.transformer(out_logits)
        out_logits = out_logits.mean(dim=1)
        logits = model.cls_head(out_logits)
        preds = logits.argmax(dim=1)
        test_correct += (preds == y_test).sum().item()

print(f"Test Accuracy: {test_correct/len(test_dataset):.4f}")

#  SAVE TorchScript 
scripted_model = torch.jit.script(model)
scripted_model.save("hand_gesture.pt")




