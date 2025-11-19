import tensorflow as tf


def build_isolated_sign_model(
        num_keypoints=42,  # 21 keypoints x 2 tay
        feature_dim=128,
        num_heads=8,
        num_transformer_blocks=4,
        num_classes=30,
        dropout_rate=0.3,
        max_frames=128,
        use_ctc_output=True,
):
    """Xây dựng mô hình nhận diện ký hiệu tay với kiến trúc Transformer.

    Khi ``use_ctc_output=True`` mô hình trả về logits theo từng khung hình với kích
    thước ``(batch, max_frames, num_classes + 1)`` để huấn luyện bằng CTC loss
    (thêm 1 lớp cho ký hiệu *blank*). Nếu đặt ``use_ctc_output=False`` mô hình sẽ
    thực hiện phân loại trực tiếp bằng cách trung bình có trọng số theo độ dài
    chuỗi.
    """
    inputs = tf.keras.Input(shape=(max_frames, num_keypoints), name="keypoints_input")
    frame_lengths = tf.keras.Input(shape=(), dtype="int32", name="frame_lengths")

    # Mặt nạ khung hình để tránh ảnh hưởng của padding
    frame_mask = tf.keras.layers.Lambda(
        lambda seq_len: tf.sequence_mask(seq_len, maxlen=max_frames),
        name="frame_mask"
    )(frame_lengths)
    frame_mask_float = tf.keras.layers.Lambda(
        lambda tensor: tf.cast(tf.expand_dims(tensor, axis=-1), tf.float32),
        name="frame_mask_float"
    )(frame_mask)

    # Tiền xử lý và chiếu lên không gian đặc trưng
    x = tf.keras.layers.Multiply(name="apply_mask_input")([inputs, frame_mask_float])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="input_norm")(x)
    x = tf.keras.layers.Dense(feature_dim, name="feature_projection")(x)

    # Positional Encoding học được (broadcast theo batch)
    position_indices = tf.range(start=0, limit=max_frames, delta=1)
    positional_embedding = tf.keras.layers.Embedding(
        input_dim=max_frames,
        output_dim=feature_dim,
        name="positional_embedding"
    )(position_indices)
    positional_embedding = tf.keras.layers.Lambda(
        lambda pe: tf.expand_dims(pe, axis=0),
        name="expand_positional_embedding"
    )(positional_embedding)
    x = tf.keras.layers.Add(name="add_positional_embedding")([x, positional_embedding])
    x = tf.keras.layers.Dropout(dropout_rate, name="embedding_dropout")(x)

    # Chuẩn bị attention mask (batch, target_seq, source_seq)
    attention_mask = tf.keras.layers.Lambda(
        lambda tensor: tf.cast(
            tf.repeat(tf.expand_dims(tensor, axis=1), repeats=max_frames, axis=1),
            tf.float32
        ),
        name="attention_mask"
    )(frame_mask)

    ffn_dim = feature_dim * 4
    for block_idx in range(num_transformer_blocks):
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=feature_dim // num_heads,
            dropout=dropout_rate,
            name=f"mha_block_{block_idx}"
        )(x, x, attention_mask=attention_mask)
        attn_output = tf.keras.layers.Dropout(dropout_rate, name=f"attn_dropout_{block_idx}")(attn_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"attn_layer_norm_{block_idx}")(x + attn_output)
        x = tf.keras.layers.Multiply(name=f"mask_after_attn_{block_idx}")([x, frame_mask_float])

        ffn_output = tf.keras.layers.Dense(
            ffn_dim,
            activation=tf.nn.gelu,
            name=f"ffn_expand_{block_idx}"
        )(x)
        ffn_output = tf.keras.layers.Dropout(dropout_rate, name=f"ffn_dropout_{block_idx}_1")(ffn_output)
        ffn_output = tf.keras.layers.Dense(feature_dim, name=f"ffn_project_{block_idx}")(ffn_output)
        ffn_output = tf.keras.layers.Dropout(dropout_rate, name=f"ffn_dropout_{block_idx}_2")(ffn_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"ffn_layer_norm_{block_idx}")(x + ffn_output)
        x = tf.keras.layers.Multiply(name=f"mask_after_ffn_{block_idx}")([x, frame_mask_float])

    x = tf.keras.layers.Dropout(dropout_rate, name="pre_output_dropout")(x)
    x = tf.keras.layers.Multiply(name="mask_before_output")([x, frame_mask_float])

    if use_ctc_output:
        logits = tf.keras.layers.Dense(num_classes + 1, name="ctc_logits")(x)
        model = tf.keras.Model(
            inputs=[inputs, frame_lengths],
            outputs=logits,
            name="IsolatedSignModelCTCReady"
        )
    else:
        # Trung bình theo độ dài thật của chuỗi thay vì GlobalAveragePooling
        summed = tf.keras.layers.Lambda(
            lambda tensor: tf.reduce_sum(tensor, axis=1),
            name="masked_sum"
        )(x)
        valid_lengths = tf.keras.layers.Lambda(
            lambda seq_len: tf.maximum(tf.cast(tf.expand_dims(seq_len, axis=-1), tf.float32), 1.0),
            name="safe_lengths"
        )(frame_lengths)
        pooled = tf.keras.layers.Lambda(
            lambda args: args[0] / args[1],
            name="masked_average"
        )([summed, valid_lengths])
        pooled = tf.keras.layers.Dense(256, activation="relu", name="classifier_dense")(pooled)
        pooled = tf.keras.layers.Dropout(dropout_rate, name="classifier_dropout")(pooled)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probs")(pooled)
        model = tf.keras.Model(
            inputs=[inputs, frame_lengths],
            outputs=outputs,
            name="IsolatedSignModel"
        )
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


if __name__ == "__main__":
    model = build_isolated_sign_model()
    model.summary()