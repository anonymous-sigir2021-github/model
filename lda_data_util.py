import random, math, collections
import numpy as np
import tensorflow as tf


def data_set(data_url):
	"""process data input."""
	data = []
	word_count = []
	fin = open(data_url)
	while True:
		line = fin.readline()
		if not line:
			break
		id_freqs = line.split()
		doc = {}
		count = 0
		for id_freq in id_freqs[1:]:
			items = id_freq.split(':')
			# python starts from 0
			if int(items[0])-1<0:
				print('WARNING INDICES!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
			doc[int(items[0])-1] = int(items[1])
			count += int(items[1])
		if count > 0:
			data.append(doc)
			word_count.append(count)
	fin.close()
	return data, word_count

def create_batches(data_size, batch_size, shuffle=True):
	"""create index by batches."""
	batches = []
	ids = list(range(data_size))
	if shuffle:
		random.shuffle(ids)
	for i in range(int(data_size / batch_size)):
		start = i * batch_size
		end = (i + 1) * batch_size
		batches.append(ids[start:end])
	# the batch of which the length is less than batch_size
	rest = data_size % batch_size
	if rest > 0:
		batches.append(list(ids[-rest:]) + [-1] * (batch_size - rest))  # -1 as padding
	return batches

def fetch_data(data, count, idx_batch, vocab_size):
	"""fetch input data by batch."""
	batch_size = len(idx_batch)
	data_batch = np.zeros((batch_size, vocab_size))
	count_batch = []
	mask = np.zeros(batch_size)
	indices = []
	values = []
	for i, doc_id in enumerate(idx_batch):
		if doc_id != -1:
			for word_id, freq in data[doc_id].items():
				data_batch[i, word_id] = freq
			count_batch.append(count[doc_id])
			mask[i]=1.0
		else:
			count_batch.append(0)
	return data_batch, count_batch, mask

def dict2sparse_tensor(data, vocab_size):
	indices = []
	values = []
	for _id, d in enumerate(data):
		idx0 = [_id] * len(d)
		idx1 = []
		for k, v in d.items():
			idx1.append(k)
			values.append(v)
		indices.extend(np.stack([idx0, idx1], axis=-1))
	
	return tf.sparse.reorder(tf.SparseTensor(
		indices=indices,
		values=np.array(values, dtype='float32'),
		dense_shape=(len(data), vocab_size)
	))

class lda_data:
	def __init__(self, stopwords_path, itos) -> None:
		super().__init__()
		self.stopwords_path = stopwords_path
		self.itos = itos
		self.stopwords = self._read_stopwords(stopwords_path)

	def _read_stopwords(self, stopwords_path):
		with open(stopwords_path, 'r', encoding='utf8') as f:
			ls = f.readlines()
		ls = [l.strip() for l in ls if l.strip() != '']
		return ls

	def idx2lda_format(self, datas):
		itos = self.itos
		stopwords = self.stopwords
		
		valid_idxs = set()
		for _id, word in enumerate(itos):
			if word not in stopwords and len(word) > 2:
				valid_idxs.add(_id)
		valid_idxs.remove(0)
		valid_idxs.remove(len(itos) - 1)
		
		self.valid_idxs = valid_idxs
		
		all_rs = []
		for domain in datas:
			domain_rs = []
			domain = np.reshape(domain, [len(domain), -1])
			for d in domain:
				summary = collections.Counter(d)
				lda_doc = {_id: cnt for _id, cnt in summary.items() if _id in valid_idxs}
				if len(lda_doc) == 0:
					print('Warning')
				else:
					domain_rs.append(lda_doc)
			all_rs.append(domain_rs)
		return all_rs

	def idx2lda_format_selection(self, datas):
		itos = self.itos
		stopwords = self.stopwords

		valid_idxs = set()
		for _id, word in enumerate(itos):
			if word not in stopwords and len(word) > 2:
				valid_idxs.add(_id)
		valid_idxs.remove(0)
		valid_idxs.remove(len(itos) - 1)

		self.valid_idxs = valid_idxs

		all_rs = []
		for domain in datas:
			domain_rs = []
			domain = np.reshape(domain, [len(domain), -1])
			for did, d in enumerate(domain):
				summary = collections.Counter(d)
				lda_doc = {_id: cnt for _id, cnt in summary.items() if _id in valid_idxs}
				if len(lda_doc) == 0:
					print('Warning')
				else:
					domain_rs.append(did)
			all_rs.append(domain_rs)
		return all_rs

	def prepare_dataset(self, datas, batch_size=200, shuffle=0, combine_data=True):
		ndomain = len(datas)
		min_data = np.min([len(domain) for domain in datas])
		domain_batch = batch_size // ndomain
		domain_take = min_data // domain_batch
		print('lda prepare dataset batch size', batch_size, 'shuffle', shuffle, 'map function', combine_data)
		print('min data size', min_data)
		print('domain batch', domain_batch, 'take', domain_take)
		
		sparse_datas = [dict2sparse_tensor(domain, len(self.itos)) for domain in datas]
		sparse_dataset = [tf.data.Dataset.from_tensor_slices(domain) for domain in sparse_datas]
		if shuffle:
			print('shuffle data with size', shuffle)
			sparse_dataset = [dataset.shuffle(shuffle) for dataset in sparse_dataset]
		if combine_data:
			def _mapping_fn(*data):
				input_data = [d for d in data]
				return tf.sparse.concat(axis=0, sp_inputs=input_data)
			sparse_dataset = [dataset.batch(domain_batch).take(domain_take) for dataset in sparse_dataset]
			dataset = tf.data.Dataset.zip(tuple(sparse_dataset)).map(_mapping_fn)
			return dataset
		else:
			return [dataset.batch(domain_batch) for dataset in sparse_dataset]
	
	def prepare_eval_dataset(self, datas, batch_size=200):
		ndomain = len(datas)
		max_data = np.max([len(domain) for domain in datas])
		domain_batch = batch_size // ndomain
		num_take = max_data // domain_batch + 1
		print('lda prepare evaluation dataset domain batch size', domain_batch, )
		print('max data size', max_data)
		print('num taken', num_take)
		
		sparse_datas = [dict2sparse_tensor(domain, len(self.itos)) for domain in datas]
		sparse_dataset = [tf.data.Dataset.from_tensor_slices(domain).repeat() for domain in sparse_datas]
		def _mapping_fn(*data):
			input_data = [d for d in data]
			return tf.sparse.concat(axis=0, sp_inputs=input_data)
		
		sparse_dataset = [dataset.batch(domain_batch) for dataset in sparse_dataset]
		dataset = tf.data.Dataset.zip(tuple(sparse_dataset)).map(_mapping_fn).take(num_take)
		return dataset
