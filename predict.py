import os
import sys
import json
import logging
import data_helpers
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

def predict_unseen_data():
	"""Step 1: load trained model and parameters"""
	params = json.loads(open('./parameters.json').read())
	checkpoint_dir = sys.argv[1]
	checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir + 'checkpoints')

	"""Step 2: load data for prediction"""
	x_raw = ["a masterpiece four years in the making", "everything is off."]
	y_test = [1, 0]

	vocab_path = os.path.join(checkpoint_dir, "vocab")
	vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
	x_test = np.array(list(vocab_processor.transform(x_raw)))
	print(x_test)

	"""Step 3: compute the predictions"""
	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)

		with sess.as_default():
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)

			input_x = graph.get_operation_by_name("input_x").outputs[0]
			dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
			predictions = graph.get_operation_by_name("output/predictions").outputs[0]

			batches = data_helpers.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)
			all_predictions = []
			for x_test_batch in batches:
				batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
				all_predictions = np.concatenate([all_predictions, batch_predictions])

	if y_test is not None:
		correct_predictions = float(sum(all_predictions == y_test))
		print("Total number of test examples: {}".format(len(y_test)))
		print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))

if __name__ == '__main__':
	predict_unseen_data()
