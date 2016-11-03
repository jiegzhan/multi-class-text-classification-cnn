import os
import sys
import json
import time
import logging
import datetime
import data_helpers
import numpy as np
import pandas as pd
import tensorflow as tf
from text_cnn import TextCNN
from tensorflow.contrib import learn

logging.getLogger().setLevel(logging.INFO)

def train_cnn():
	params = json.loads(open('./parameters.json').read())

	x_text, y, df, labels = data_helpers.load_data_and_labels('./data/consumer_complaints.csv.zip')

	max_document_length = max([len(x.split(' ')) for x in x_text])
	print('max_documnet_length: {}'.format(max_document_length))

	vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
	x = np.array(list(vocab_processor.fit_transform(x_text)))

	test_size = int(0.1 * len(x))
	x_, x_test = x[:-test_size], x[-test_size:]
	y_, y_test = y[:-test_size], y[-test_size:]

	df_train, df_test = df[:-test_size], df[-test_size:]
	print(df_train.shape)
	print(df_test.shape)
	with open('./data_train.json', 'w') as outfile:
		json.dump(df_train.to_dict(orient='records'), outfile, indent=4)
	with open('./data_test.json', 'w') as outfile:
		json.dump(df_test.to_dict(orient='records'), outfile, indent=4)
	with open('./labels.json', 'w') as outfile:
		json.dump(labels, outfile, indent=4)

	shuffle_indices = np.random.permutation(np.arange(len(y_)))
	x_shuffled = x_[shuffle_indices]
	y_shuffled = y_[shuffle_indices]

	dev_size = int(0.1 * len(x_))
	x_train, x_dev = x_shuffled[:-dev_size], x_shuffled[-dev_size:]
	y_train, y_dev = y_shuffled[:-dev_size], y_shuffled[-dev_size:]

	print('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
	print('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

	with tf.Graph().as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn = TextCNN(
				sequence_length=x_train.shape[1],
				num_classes=y_train.shape[1],
				vocab_size=len(vocab_processor.vocabulary_),
				embedding_size=params['embedding_dim'],
				filter_sizes=list(map(int, params['filter_sizes'].split(","))),
				num_filters=params['num_filters'],
				l2_reg_lambda=params['l2_reg_lambda'])

			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(1e-3)
			grads_and_vars = optimizer.compute_gradients(cnn.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

			timestamp = str(int(time.time()))
			out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model_" + timestamp))

			checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
			checkpoint_prefix = os.path.join(checkpoint_dir, "model")
			if not os.path.exists(checkpoint_dir):
				os.makedirs(checkpoint_dir)
			saver = tf.train.Saver(tf.all_variables())

			vocab_processor.save(os.path.join(out_dir, "vocab"))

			sess.run(tf.initialize_all_variables())

			def train_step(x_batch, y_batch):
				feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: params['dropout_keep_prob']}
				_, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)

			def dev_step(x_batch, y_batch):
				feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0}
				step, loss, accuracy, num_correct = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.num_correct], feed_dict)
				return loss, accuracy, num_correct

			# Training starts here
			batches = data_helpers.batch_iter(list(zip(x_train, y_train)), params['batch_size'], params['num_epochs'])
			best_accuracy, best_at_step = 0, 0

			for batch in batches:
				x_batch, y_batch = zip(*batch)
				train_step(x_batch, y_batch)
				current_step = tf.train.global_step(sess, global_step)

				if current_step % params['evaluate_every'] == 0:
					print("\nEvaluation:")
					dev_batches = data_helpers.batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)
					total_dev_correct = 0
					for dev_batch in dev_batches:
						x_dev_batch, y_dev_batch = zip(*dev_batch)
						loss, accuracy, num_dev_correct = dev_step(x_dev_batch, y_dev_batch)
						total_dev_correct += num_dev_correct

					accuracy = float(total_dev_correct) / dev_size
					print('total_dev_correct: {}'.format(total_dev_correct))
					print('accuracy on dev: {}'.format(accuracy))

					if accuracy >= best_accuracy:
						best_accuracy, best_at_step = accuracy, current_step
						path = saver.save(sess, checkpoint_prefix, global_step=current_step)
						logging.critical('Save the best model checkpoint to {} at evaluate step {}'.format(path, best_at_step))
						logging.critical('Best accuracy on dev set: {}, at step {}'.format(best_accuracy, best_at_step))

if __name__ == '__main__':
	train_cnn()
