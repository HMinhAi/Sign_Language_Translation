import tensorflow as tf

def build_isolated_sign_model(
        num_keypoints=128,  # 21 keypoints x 2 tay
        feature_dim=256,
        num_heads=8,
        num_classes=30,
        dropout_rate=0.3,
        max_frames=180
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

    # Pooling và phân loại
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="IsolatedSignModel")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    model = build_isolated_sign_model()
    model.summary()
