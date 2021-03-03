import tensorflow as tf
import numpy as np
from mymodel.myconfig import *
from mymodel.mymodels import *
from mymodel import my_data_util
import os, sys, datetime, multiprocessing, pickle, itertools, copy, bisect
K = tf.keras

default_gpu = 5
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % default_gpu
print('GPU:', default_gpu)

def prepare_data(settings: Runtime_Settings):
	print('repeat seed', settings.repeat)
	np.random.seed(settings.repeat)
	tf.random.set_seed(settings.repeat)

	fpath = './mymodel/data/all_data_filtered.pickle'
	train_data, train_data_sentiment_label, val_data, val_data_sentiment_label, unlabeled_data, unlabeled_data_domain_label, val_unlabeled_data, val_unlabeled_data_domain_label, all_test_data, all_test_data_sentiment_label, embeddings, itos, stoi = my_data_util.read_pickle_data(fpath)

	print('selecting source domain from', settings.data_config.source_domain)
	train_data = [train_data[idx] for idx in settings.data_config.source_domain]
	train_data_sentiment_label = [train_data_sentiment_label[idx] for idx in settings.data_config.source_domain]
	print('selecting target domain from', settings.data_config.target_domain)
	test_data = all_test_data[settings.data_config.target_domain]
	test_data_sentiment_label = all_test_data_sentiment_label[settings.data_config.target_domain]

	train_data = [data[0] for data in train_data]
	unlabeled_data = [data[0] for data in unlabeled_data]

	def get_k_learner_domain_idx(n_topic, k, target, vec=0, selection_size=500, num_learners=5):
		lda_score_path = './mymodel/lda_model/lda_data_score_%d_%d_%d.pickle' % (n_topic[0], n_topic[1], settings.repeat)
		domain_topic_proportions, target_topic_proportions = my_data_util.read_pickle_data(lda_score_path)
		domain_sizes = [len(domain[vec]) for domain in domain_topic_proportions]
		print('domain size', domain_sizes)
		bisect_count = np.cumsum(domain_sizes)
		print('cumulated sum', bisect_count)

		target_topic_proportions = target_topic_proportions[target][vec][:, n_topic[0]:]
		target_topic_proportion_overall = np.sum(target_topic_proportions, axis=0)
		topic_idx = np.reshape(np.argsort(-target_topic_proportion_overall), [num_learners, -1])[k]

		domain_scores = [np.sum(domain[vec][:, n_topic[0] + topic_idx], axis=1) for domain in domain_topic_proportions]
		domain_scores = np.concatenate(domain_scores, axis=0)
		selections = np.argsort(-domain_scores)[:selection_size]
		rs = [[] for _ in domain_topic_proportions]
		for sel in selections:
			did = bisect.bisect_right(bisect_count, sel)
			if did > 0:
				did_idx = sel - bisect_count[did-1]
			else:
				did_idx = sel
			rs[did].append(did_idx)
		print([len(_) for _ in rs])
		return rs

	unlabeled_data_idxs = get_k_learner_domain_idx(settings.n_topic, settings.k_learner, settings.data_config.target_domain, settings.topic_vector_idx)
	unlabeled_data = [data[idxs] for idxs, data in zip(unlabeled_data_idxs, unlabeled_data)]
	unlabeled_data_domain_label = [data[idxs] for idxs, data in zip(unlabeled_data_idxs, unlabeled_data_domain_label)]

	random_orders = [(np.argsort(np.random.rand(data.shape[0] // 2)),
					  np.argsort(np.random.rand(data.shape[0] // 2)) + data.shape[0] // 2)
					 for data in train_data]
	print('training samples')
	for pos_order, neg_order in random_orders:
		print(pos_order[:settings.data_config.few_shot // 2], neg_order[:settings.data_config.few_shot // 2])
	samples = [np.concatenate([pos_order[:settings.data_config.few_shot // 2], neg_order[:settings.data_config.few_shot // 2]])
			   for pos_order, neg_order in random_orders]
	train_data = [data[sample] for data, sample in zip(train_data, samples)]
	train_data_sentiment_label = [data[sample] for data, sample in zip(train_data_sentiment_label, samples)]

	test_data = test_data[0]

	train_data = np.concatenate(train_data, axis=0)
	train_data_sentiment_label = np.concatenate(train_data_sentiment_label, axis=0)
	train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_data_sentiment_label)).repeat().batch(settings.batch_size)
	unlabeled_data = np.concatenate(unlabeled_data, axis=0)
	unlabeled_data_domain_label = np.concatenate(unlabeled_data_domain_label, axis=0)
	unlabeled_dataset = tf.data.Dataset.from_tensor_slices((unlabeled_data, unlabeled_data_domain_label)).shuffle(1000).batch(settings.batch_size, drop_remainder=True)
	all_training_dataset = tf.data.Dataset.zip((train_dataset, unlabeled_dataset)).prefetch(5)
	test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_data_sentiment_label)).batch(settings.testset_batch)


	return all_training_dataset, test_dataset, embeddings, itos, stoi

def run(settings: Runtime_Settings, others=None):
	all_data = prepare_data(settings)

	print('repeat seed', settings.repeat)
	np.random.seed(settings.repeat)
	tf.random.set_seed(settings.repeat)

	all_training_dataset, test_dataset, word_embedding, idx2word, stoi = all_data
	def prepare_submodels(settings: Runtime_Settings, embedding):
		config = settings.model_config
		query_settings = settings.query_settings

		embed_sm = my_embedding_submodel(embedding)
		encoder = my_encoder_submodel(config)

		weight_sm = my_document_based_attention_submodel(config, query_settings)

		sent_decoder = my_decoder_submodel(config, 'Sentiment', settings.model_config.sentiment_classes)
		dom_decoder = my_decoder_submodel(config, 'Domain', settings.model_config.domain_classes)

		return embed_sm, encoder, weight_sm, sent_decoder, dom_decoder

	def build_model_by_settings(settings, word_embedding):
		query_settings = settings.query_settings

		embed_sm, encoder, weight_sm, sent_decoder, dom_decoder = prepare_submodels(settings, word_embedding)

		linear_sm = my_linear_submodel()
		rnn_sm = None

		sent_composition = linear_sm
		doc_composition = linear_sm

		sent_fix_weight_dense_submodel = my_fix_weight_dense_submodel(sent_decoder)
		sent_model = my_model(settings, [_[1] for _ in query_settings], 'sent', embed_sm, encoder, weight_sm,
		                      sent_composition, doc_composition, sent_decoder,
		                      settings.model_config.sentiment_classes, sent_fix_weight_dense_submodel)
		dom_fix_weight_dense_submodel = my_fix_weight_dense_submodel(dom_decoder)
		dom_model = my_model(settings, [_[2] for _ in query_settings], 'dom', embed_sm, encoder, weight_sm, linear_sm,
		                     linear_sm, dom_decoder,
		                     settings.model_config.domain_classes, dom_fix_weight_dense_submodel)

		return [sent_model, dom_model]

	models = build_model_by_settings(settings, word_embedding)

	for model in models:
		for l in model.layers:
			print(l, l.trainable)
		print()

	for model in models:
		model.itos = idx2word

	sent_model, dom_model = models
	sent_model.compile(K.optimizers.Adam(settings.lr))
	dom_model.compile(K.optimizers.Adam(settings.lr))
	model = my_data_convert_model(settings, sent_model, dom_model)

	training_results = []
	testing_results = []
	testing_predicts = []

	epoch_count = 0
	while epoch_count < 20:
		for _data in all_training_dataset:
			model.clear_metrics()
			sentiment_data, domain_data = _data
			# model.train_step(_data, None, settings.discover_hidden_topic)
			model.sentiment_model.train_step(sentiment_data, None, False)
			model.domain_model.train_step(domain_data, None, settings.discover_hidden_topic)
			rs = model.get_results()

			training_results.append(rs)
			print(rs)

			epoch_count += 1

	print('testing')
	model.clear_metrics()
	for test_data in test_dataset:
		model.test_step(test_data)
	rs = model.get_test_results(2)
	testing_results.append(rs)
	testing_predicts.append(model.test_predicts)
	print(rs)


	if settings.save_result_to_pickle:
		print('saving results', training_results, testing_results)
		with open(settings.output_pickle_path, 'ab') as f:
			pickle.dump([settings, training_results, testing_results, testing_predicts], f)

	return


def baseline_model(debug=False):
	def _get_basic_settings():
		pickle_file_path = './mymodel/pickle/result/result.pickle'
		_word_dimension = 300
		settings = Runtime_Settings()
		settings.data_config = Data_Config()
		settings.model_config = Model_Config()

		settings.query_settings = [
			['pivot', False, True, -1, ],
			['domain_only', True, False, -1, ],
		]

		settings.sent_doc_composition = ['linear', 'linear']

		settings.lr = 0.002

		settings.data_config.embedding_model = 'word2vec'
		settings.model_config.word_projection_dimension = _word_dimension
		settings.model_config.embedding_dimension = _word_dimension
		settings.epoch = 10
		settings.num_steps = 2
		settings.data_config.few_shot = 30
		settings.batch_size = 120  # 40
		settings.data_config.word_dimension = _word_dimension
		settings.data_config.max_distinct_word = 40000
		settings.repeat = 0
		settings.output_pickle_path = pickle_file_path
		settings.data_config.target_domain = 2

		settings.save_result_to_pickle = True
		return settings

	settings = _get_basic_settings()
	n_repeats = 10
	settings.sentence_constant_weight = True
	settings.special_training_apply_main_task = False
	settings.use_random_sentence_label = True

	settings.data_config.few_shot = 30
	settings.testset_batch = 500
	settings.num_steps = 2
	settings.epoch = 13

	settings.n_topic = [20, 40]

	for tmp_repeat in list(range(n_repeats)):
		settings.repeat = tmp_repeat
		for few_shot in [10, 20, 30, 40, 50]:
			settings.data_config.few_shot = few_shot
			for target in range(5):
				all_sources = list(range(5))
				all_sources.remove(target)

				settings.batch_size = settings.data_config.few_shot * len(all_sources)
				settings.data_config.source_domain = all_sources
				settings.data_config.target_domain = target
				settings.model_config.domain_classes = len(all_sources) + 1

				for k_learner in range(5):
					settings.k_learner = k_learner

					print('running', tmp_repeat, few_shot, all_sources, target, k_learner)

					p = multiprocessing.Process(target=run, args=(settings, None))
					p.start()
					p.join()


if __name__ == '__main__':
	baseline_model(False)
