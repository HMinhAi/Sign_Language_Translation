import numpy as np
import matplotlib.pyplot as plt

history = np.load("models/phase1/history.npy", allow_pickle=True).item()

plt.figure(figsize=(8,4))
plt.plot(history["accuracy"], label="Train Acc")
plt.plot(history["val_accuracy"], label="Val Acc")
plt.title("Training Progress - Phase 1")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()
