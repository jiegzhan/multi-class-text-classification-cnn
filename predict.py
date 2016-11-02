import os
import sys
import time
import json
import datetime
import data_helpers
import numpy as np
import tensorflow as tf
from text_cnn import TextCNN
from tensorflow.contrib import learn

def predict_unseen_data():
	"""Step 1: load trained model and parameters"""
	params = json.loads(open('./parameters.json').read())
	checkpoint_dir = sys.argv[1]
	print(checkpoint_dir)
	checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir + 'checkpoints')
	print(checkpoint_file)

	"""Step 2: load data for prediction"""
	x_raw = ["a masterpiece four years in the making", "everything is off."]
	y_test = [1, 0]

	# Map data into vocabulary
	vocab_path = os.path.join(checkpoint_dir, "vocab")
	vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
	x_test = np.array(list(vocab_processor.transform(x_raw)))
	print(x_test)

	print("\nEvaluating...\n")
	# checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			# Load the saved meta graph and restore variables
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)

			# Get the placeholders from the graph by name
			input_x = graph.get_operation_by_name("input_x").outputs[0]
			# input_y = graph.get_operation_by_name("input_y").outputs[0]
			dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

			# Tensors we want to evaluate
			predictions = graph.get_operation_by_name("output/predictions").outputs[0]

			# Generate batches for one epoch
			batches = data_helpers.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)

			# Collect the predictions here
			all_predictions = []

			for x_test_batch in batches:
				batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
				all_predictions = np.concatenate([all_predictions, batch_predictions])

	# Print accuracy if y_test is defined
	if y_test is not None:
		correct_predictions = float(sum(all_predictions == y_test))
		print("Total number of test examples: {}".format(len(y_test)))
		print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

if __name__ == '__main__':
	predict_unseen_data()
