import re
import csv
import sys
import json
import numpy as np
import pandas as pd
import itertools
from collections import Counter

def clean_str(string):
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)

	string = re.sub(r'\S*(x{2,}|X{2,})\S*',"xxx", string)
	string = re.sub(r'[^\x00-\x7F]+', "", string)

	return string.strip().lower()

def load_data_and_labels(filename):
	df = pd.read_csv(filename, dtype={'consumer_complaint_narrative': object})
	selected = ['product', 'consumer_complaint_narrative']
	non_selected = list(set(df.columns) - set(selected))
	print(df.shape)

	df = df.drop(non_selected, axis=1)
	df = df.dropna(axis=0, how='any', subset=selected)
	df = df.reindex(np.random.permutation(df.index))

	labels = sorted(list(set(df[selected[0]].tolist())))
	print(labels)
	one_hot = np.zeros((len(labels), len(labels)), int)
	np.fill_diagonal(one_hot, 1)

	label_dict = dict(zip(labels, one_hot))
	for key in sorted(label_dict.keys()):
		print('{} ---> {}'.format(key, label_dict[key]))

	y = df[selected[0]].apply(lambda x: label_dict[x]).tolist()
	x_text = df[selected[1]].apply(lambda x: clean_str(x)).tolist()

	y = np.asarray(y)

	return [x_text, y, df, labels]

	# Load data from files
	positive_examples = list(open("./data/rt-polaritydata/rt-polarity.pos", "r").readlines())
	positive_examples = [s.strip() for s in positive_examples]
	negative_examples = list(open("./data/rt-polaritydata/rt-polarity.neg", "r").readlines())
	negative_examples = [s.strip() for s in negative_examples]
	# Split by words
	x_text = positive_examples + negative_examples
	x_text = [clean_str(sent) for sent in x_text]
	# Generate labels
	positive_labels = [[0, 1] for _ in positive_examples]
	negative_labels = [[1, 0] for _ in negative_examples]
	y = np.concatenate([positive_labels, negative_labels], 0)
	return [x_text, y]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(len(data)/batch_size) + 1
	for epoch in range(num_epochs):
		# Shuffle the data at each epoch
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]

if __name__ == '__main__':
	input_file = './data/consumer_complaints.csv'
	load_data_and_labels(input_file)
