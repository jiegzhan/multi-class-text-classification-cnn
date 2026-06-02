import os
import sys
import json
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

import data_helper
from text_cnn import build_text_cnn_model

logging.getLogger().setLevel(logging.INFO)

SEED = 42


def train_cnn():
    if len(sys.argv) < 3:
        print("Usage: python train.py <data_file> <params_file>")
        sys.exit(1)

    train_file = sys.argv[1]
    parameter_file = sys.argv[2]

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    x_raw, y_raw, df, labels = data_helper.load_data_and_labels(train_file)

    with open(parameter_file) as f:
        params = json.load(f)

    # Split first so the vectorizer (vocab + sequence length) is fit on train data only.
    x_, x_test_raw, y_, y_test = train_test_split(x_raw, y_raw, test_size=0.1, random_state=SEED)
    x_train_raw, x_dev_raw, y_train, y_dev = train_test_split(x_, y_, test_size=0.1, random_state=SEED)

    train_lengths = np.array([len(x.split(' ')) for x in x_train_raw])
    max_document_length = int(np.percentile(train_lengths, 95))
    logging.info('Train sentence length: max={}, p95={} (using p95)'.format(int(train_lengths.max()), max_document_length))

    # clean_str already lowercases, so disable the layer's standardizer.
    vectorize_layer = keras.layers.TextVectorization(
        max_tokens=None,
        output_mode="int",
        output_sequence_length=max_document_length,
        standardize=None,
        split="whitespace",
    )
    vectorize_layer.adapt(np.array(x_train_raw))

    vocab_size = vectorize_layer.vocabulary_size()
    logging.info('Vocabulary size: {}'.format(vocab_size))

    x_train = vectorize_layer(np.array(x_train_raw)).numpy()
    x_dev = vectorize_layer(np.array(x_dev_raw)).numpy()
    x_test = vectorize_layer(np.array(x_test_raw)).numpy()
    y_train = np.array(y_train)
    y_dev = np.array(y_dev)
    y_test = np.array(y_test)

    logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))

    filter_sizes = list(map(int, params['filter_sizes'].split(',')))

    model = build_text_cnn_model(
        max_document_length=max_document_length,
        vocab_size=vocab_size,
        embedding_dim=params['embedding_dim'],
        filter_sizes=filter_sizes,
        num_filters=params['num_filters'],
        num_classes=y_train.shape[1],
        dropout_rate=params['dropout_rate'],
        l2_reg_lambda=params['l2_reg_lambda'],
        learning_rate=params.get('learning_rate', 0.001),
    )
    model.summary()

    batch_size = params['batch_size']

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=len(x_train), seed=SEED)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (
        tf.data.Dataset.from_tensor_slices((x_dev, y_dev))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join('.', 'trained_model_' + timestamp))

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(out_dir, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=params['num_epochs'],
        callbacks=callbacks,
    )

    config = {
        'max_document_length': max_document_length,
        'vocab_size': vocab_size,
        'labels': labels,
        'vocabulary': vectorize_layer.get_vocabulary(include_special_tokens=False),
    }
    with open(os.path.join(out_dir, 'train_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    test_dataset = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_loss, test_accuracy = model.evaluate(test_dataset)
    logging.info('Accuracy on test set: {:.4f}'.format(test_accuracy))
    logging.info('The training is complete')


if __name__ == '__main__':
    train_cnn()
