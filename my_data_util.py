
import numpy as np
import tensorflow as tf
import pickle, os

def read_pickle_data(fname):
	with open(fname, 'rb') as f:
		data = pickle.load(f)
	return data

def save_pickle_data(data, fname):
	with open(fname, 'wb') as f:
		pickle.dump(data, f)

def _corpus_to_sparse_tensor(corpus, stoi, max_sent=20, max_word=25):
	sentiment_labels = []
	idxs = []
	values = []
	shape = [len(corpus), max_sent, max_word]
	for did, doc in enumerate(corpus):
		sentiment_labels.append(1 - doc[2])
		for sid, sent in enumerate(doc[0]['token'][:max_sent]):
			for wid, word in enumerate(sent[:max_word]):
				if word in stoi:
					idxs.append([did, sid, wid])
					values.append(stoi[word])
				else:
					idxs.append([did, sid, wid])
					values.append(0)
					
	idxs = np.array(idxs, dtype='int32')
	values = np.array(values, dtype='int32')
	shape = np.array(shape, dtype='int32')

	sentiment_labels = np.array(sentiment_labels, dtype='int32')
	
	return (idxs, values, shape), sentiment_labels


def data_to_tfdataset(train_data, train_data_sentiment_label, val_data, val_data_sentiment_label, unlabeled_data, unlabeled_data_domain_label, val_unlabeled_data, val_unlabeled_data_domain_label, batch_size=100):
	ndomains = len(train_data)
	domain_batch_size = batch_size // ndomains

	unlabeled_data_size = [data.shape[0] for data in unlabeled_data]
	min_unlabeled_data_size = min(unlabeled_data_size)
	domain_take = min_unlabeled_data_size // domain_batch_size
	
	train_dataset = []
	validation_dataset = []
	unlabeled_dataset = []
	validation_unlabeled_dataset = []
	for t, tl, v, vl, u, ul, vu, vul in zip(train_data, train_data_sentiment_label, val_data, val_data_sentiment_label, unlabeled_data, unlabeled_data_domain_label, val_unlabeled_data, val_unlabeled_data_domain_label):
		train = tf.data.Dataset.from_tensor_slices((t, tl)).shuffle(10000).repeat().batch(domain_batch_size)
		valid = tf.data.Dataset.from_tensor_slices((v, vl)).shuffle(10000).batch(domain_batch_size)
		unlabeled = tf.data.Dataset.from_tensor_slices((u, ul)).shuffle(40000).batch(domain_batch_size).take(domain_take)
		valid_unlabeled = tf.data.Dataset.from_tensor_slices((vu, vul)).shuffle(10000).batch(domain_batch_size)

		train_dataset.append(train)
		validation_dataset.append(valid)
		unlabeled_dataset.append(unlabeled)
		validation_unlabeled_dataset.append(valid_unlabeled)
		
	_ = []
	_.extend(train_dataset)
	_.extend(unlabeled_dataset)
	all_training_dataset = tf.data.Dataset.zip(tuple(_))
	_ = []
	_.extend(validation_dataset)
	_.extend(validation_unlabeled_dataset)
	all_validation_dataset = tf.data.Dataset.zip(tuple(_))
	
	return all_training_dataset, all_validation_dataset


def read_all_data():
	all_data_path = './mymodel/data/all_data.pickle'

	print('reading all data')
	train_data, train_data_sentiment_label, val_data, val_data_sentiment_label, unlabeled_data, unlabeled_data_domain_label, val_unlabeled_data, val_unlabeled_data_domain_label, all_test_data, all_test_data_sentiment_label, embeddings, itos, stoi = read_pickle_data(
		all_data_path)

	return train_data, train_data_sentiment_label, val_data, val_data_sentiment_label, unlabeled_data, unlabeled_data_domain_label, val_unlabeled_data, val_unlabeled_data_domain_label, all_test_data, all_test_data_sentiment_label, embeddings, itos, stoi
