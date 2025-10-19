import tensorflow as tf

def build_ctc_sign_model(
        num_keypoints=42,
        feature_dim=128,
        num_heads=8,
        vocab_size=30,   # số nhãn ký hiệu
        dropout_rate=0.3,
        max_frames=128
):
    inputs = tf.keras.Input(shape=(max_frames, num_keypoints), name="keypoints_input")

    # Positional Encoding
    pos_encoding = tf.keras.layers.Embedding(input_dim=max_frames, output_dim=feature_dim)(
        tf.range(start=0, limit=max_frames))
    x = tf.keras.layers.Dense(feature_dim)(inputs)
    x = x + pos_encoding

    # Transformer Encoder
    for _ in range(2):
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=feature_dim // num_heads)(x, x)
        attn_output = tf.keras.layers.Dropout(dropout_rate)(attn_output)
        x = tf.keras.layers.LayerNormalization()(x + attn_output)

        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(feature_dim * 4, activation="relu"),
            tf.keras.layers.Dense(feature_dim),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        x = tf.keras.layers.LayerNormalization()(x + ffn(x))

    # Output logits (không softmax)
    logits = tf.keras.layers.Dense(vocab_size + 1, name="logits")(x)  # +1 cho ký hiệu blank của CTC
    model = tf.keras.Model(inputs, logits, name="CTCSignModel")

    return model


class CTCLossLayer(tf.keras.layers.Layer):
    """Custom CTC loss cho tf.keras Model"""
    def call(self, inputs):
        y_pred, labels, input_len, label_len = inputs
        loss = tf.keras.backend.ctc_batch_cost(labels, y_pred, input_len, label_len)
        self.add_loss(tf.reduce_mean(loss))
        return y_pred


if __name__ == "__main__":
    model = build_ctc_sign_model()
    model.summary()
