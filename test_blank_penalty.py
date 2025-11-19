"""
Simple test to check if blank penalty helps.
"""
import numpy as np
import tensorflow as tf

# Simulate model output where blank is dominant
print("=" * 60)
print("TESTING BLANK PENALTY")
print("=" * 60)

# Create fake logits where blank (index 30) has high scores
seq_len = 60
num_classes = 31

# Scenario: Blank dominates
logits = np.random.randn(seq_len, num_classes).astype(np.float32)
logits[:, 30] += 5.0  # Make blank token much stronger

print(f"\n1. Original logits:")
print(f"   Shape: {logits.shape}")
print(f"   Blank (30) mean: {logits[:, 30].mean():.3f}")
print(f"   Others mean: {logits[:, :30].mean():.3f}")

# Decode without penalty
probs = tf.nn.softmax(logits, axis=-1).numpy()
predictions = np.argmax(probs, axis=-1)
blank_count = np.sum(predictions == 30)

print(f"\n2. Without penalty:")
print(f"   Predictions: {predictions[:20]}...")
print(f"   Blank count: {blank_count}/{seq_len} ({blank_count/seq_len*100:.1f}%)")

# Decode with blank penalty
probs_adjusted = probs.copy()
probs_adjusted[:, 30] *= 0.1  # Penalize blank
probs_adjusted = probs_adjusted / probs_adjusted.sum(axis=-1, keepdims=True)

predictions_adjusted = np.argmax(probs_adjusted, axis=-1)
blank_count_adj = np.sum(predictions_adjusted == 30)

print(f"\n3. With blank penalty (0.1x):")
print(f"   Predictions: {predictions_adjusted[:20]}...")
print(f"   Blank count: {blank_count_adj}/{seq_len} ({blank_count_adj/seq_len*100:.1f}%)")

# Remove blanks and duplicates
def decode_ctc(preds):
    output = []
    prev = -1
    for pred in preds:
        if pred != 30 and pred != prev:
            output.append(int(pred))
        prev = pred
    return output

decoded_original = decode_ctc(predictions)
decoded_adjusted = decode_ctc(predictions_adjusted)

print(f"\n4. CTC Decoded:")
print(f"   Original: {decoded_original}")
print(f"   With penalty: {decoded_adjusted}")

print(f"\n" + "=" * 60)
if decoded_original:
    print("✅ Original decode works!")
else:
    print("❌ Original decode empty (all blanks)")
    
if decoded_adjusted:
    print("✅ Penalty helps - got non-blank predictions!")
else:
    print("❌ Penalty didn't help")
print("=" * 60)
