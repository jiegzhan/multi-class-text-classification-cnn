import os
import sys
import json
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras

import data_helper

logging.getLogger().setLevel(logging.INFO)


def predict_unseen_data():
    if len(sys.argv) < 3:
        print("Usage: python predict.py <model_directory> <test_data.json>")
        sys.exit(1)

    model_dir = sys.argv[1]
    test_file = sys.argv[2]

    model = keras.models.load_model(os.path.join(model_dir, 'best_model.keras'))
    vectorizer_model = keras.models.load_model(os.path.join(model_dir, 'vectorizer.keras'))
    vectorize_layer = vectorizer_model.layers[0]

    with open(os.path.join(model_dir, 'train_config.json')) as f:
        config = json.load(f)
    labels = config['labels']

    with open(test_file) as f:
        test_examples = json.load(f)

    x_raw = [example['consumer_complaint_narrative'] for example in test_examples]
    x_cleaned = [data_helper.clean_str(x) for x in x_raw]
    logging.info('The number of test examples: {}'.format(len(x_cleaned)))

    x_test = vectorize_layer(np.array(x_cleaned)).numpy()

    predictions = model.predict(x_test)
    predicted_indices = np.argmax(predictions, axis=1)
    predicted_labels = [labels[idx] for idx in predicted_indices]

    if test_examples and 'product' in test_examples[0]:
        y_true = [example['product'] for example in test_examples]
        correct = sum(p == t for p, t in zip(predicted_labels, y_true))
        accuracy = correct / len(y_true)
        logging.info('Accuracy: {:.4f}'.format(accuracy))

    for idx, example in enumerate(test_examples):
        example['predicted_product'] = predicted_labels[idx]
        example['confidence'] = float(np.max(predictions[idx]))

    output_file = './data/predictions_output.json'
    with open(output_file, 'w') as f:
        json.dump(test_examples, f, indent=4)

    logging.info('Predictions saved to {}'.format(output_file))
    logging.info('Prediction complete')


if __name__ == '__main__':
    predict_unseen_data()
