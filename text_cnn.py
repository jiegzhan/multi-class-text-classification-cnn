import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_text_cnn_model(
    max_document_length,
    vocab_size,
    embedding_dim,
    filter_sizes,
    num_filters,
    num_classes,
    dropout_rate=0.5,
    l2_reg_lambda=0.0,
    learning_rate=0.001,
):
    regularizer = keras.regularizers.l2(l2_reg_lambda) if l2_reg_lambda > 0 else None

    inputs = layers.Input(shape=(max_document_length,), dtype="int32", name="input_x")

    embedding = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        embeddings_initializer=keras.initializers.RandomUniform(-1.0, 1.0),
        name="embedding",
    )(inputs)

    pooled_outputs = []
    for fs in filter_sizes:
        conv = layers.Conv1D(
            filters=num_filters,
            kernel_size=fs,
            activation="relu",
            padding="valid",
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
            bias_initializer=keras.initializers.Constant(0.1),
            kernel_regularizer=regularizer,
            name=f"conv_{fs}",
        )(embedding)
        pool = layers.GlobalMaxPooling1D(name=f"maxpool_{fs}")(conv)
        pooled_outputs.append(pool)

    if len(pooled_outputs) > 1:
        concat = layers.Concatenate(name="concat")(pooled_outputs)
    else:
        concat = pooled_outputs[0]

    dropped = layers.Dropout(rate=dropout_rate, name="dropout")(concat)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer="glorot_uniform",
        bias_initializer=keras.initializers.Constant(0.1),
        kernel_regularizer=regularizer,
        name="predictions",
    )(dropped)

    model = keras.Model(inputs=inputs, outputs=outputs, name="TextCNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
