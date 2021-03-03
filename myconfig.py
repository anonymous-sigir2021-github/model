import hashlib
import numpy as np
import os

pos_tags = [
	'--PADDING--',
	'ADJ',
	'ADP',
	'ADV',
	'AUX',
	'CONJ',
	'CCONJ',
	'DET',
	'INTJ',
	'NOUN',
	'NUM',
	'PART',
	'PRON',
	'PROPN',
	'PUNCT',
	'SCONJ',
	'SYM',
	'VERB',
	'X',
	'SPACE',
]

class Runtime_Settings:
	def __init__(self):
		
		self.batch_size = 100
		self.epoch = 25
		self.lr = 0.001
		self.grl_adapt = None

		self.model_config: Model_Config = Model_Config()
		self.data_config: Data_Config = Data_Config()

		self.query_settings = [
			['pivot', False, True, -1, ],
			['domain_only', True, False, -1, ],
		]
		
		self.apply_topic_weight = 'doc'
		self.sent_doc_composition = ['linear', 'linear']

		self.output_pickle_path = None

		self.num_steps = 10
		
		self.shuffle_unlabeled_data = True # false for debugging, so as to match the adj-noun parent information

		self.auxiliary_task_discount = 0.2
		
		self.repeat = 0
		self.load_model = None
		self.save_model = None
		self.discover_hidden_topic = True
		self.apply_weight_transfer = False

		self.small_testset = False
		self.testset_batch = 100
		
		
		self.weight_times_num_word = True
		self.aspect_finding_affect_encoder = True
		self.domain_only_function_word_ratio = 0.5
		self.sample_reweight_power = 5
		
		self.aspect_model_epoch = -1
		self.aspect_model_num_steps = -1
		self.train_alpha = True
		
		self.aspect_model_config: Aspect_Model_Config = None
		self.perturb = None
		self.sentence_diff_threshold = 0.8
		self.sentence_constant_weight = True
		
		self.half_steps_for_sentence = False
		self.sentence_epoch = 4
		self.special_training_apply_main_task = False
		self.use_submodel_softmax = True
		self.sentence_weight_apply_uncertainty = False
		self.use_random_sentence_label = False
		
		self.sentence_weight_apply_uncertainty_threshold = None
		self.sentence_weight_apply_uncertainty_min_weight = None
		
		self.n_topic = None
		self.k_learner = None
		self.topic_vector_idx = 0
		self.unlabeled_data_top_k = 200
		
		self.save_result_to_pickle = True
		
class Aspect_Model_Config:
	def __init__(self):
		self.domain_specific_encoder_features = [300, 5, 5]
		self.domain_share_encoder_features = [300, 10, 10]
		
		self.domain_specific_decoder_features = [5]
		self.domain_share_decoder_features = [10]
		
		self.predicted_theta_weight = 0.7

class Data_Config:
	def __init__(self):
		self.source_domain = 0
		self.target_domain = 1
		self.max_sent = 25
		self.max_word = 35
		self.unlabeled_validation_size = 1000
		self.embedding_dimension = 200
		self.embedding_model = 'word2vec'
		
		self.pos_size = len(pos_tags)

		self.max_distinct_word = 40000

		self.few_shot = 30


class Model_Config:
	def __init__(self, summary_root_path='./mymodel/output/tensorboard_output/default', message='', raw_output_path='./mymodel/output/raw_output'):

		self.activation_function = 'tanh'


		self.rnn_ngram = None
		self.rnn_padding = [0, 0]

		self.word_minux_max = True
		self.normalize_word_weighting = True
		self.embedding_dimension = 200
		self.word_projection_dimension = 200

		self.multiple_sentence_querys = True
		self.sentence_minux_max = True
		self.normalize_sentence_weighting = True
		self.sentence_projection_dimension = int(self.word_projection_dimension / 1)

		self.max_sent = 25
		self.max_word = 35
		self.sentiment_classes = 2
		self.domain_classes = 5
		
		self.bias_apply_l2_regularization_loss = True
		self.orthogonal_loss_weight = 5
		self.l2_regularization_loss_weight = 0.005
		self.learning_rate = 0.0001
		self.domain_trainer_discount = 2.5

		self.summary_root_path = summary_root_path
		self.message = message
		self.raw_output_path = raw_output_path
		
		self.n_subquery = 1

		if not os.path.exists(raw_output_path):
			os.makedirs(raw_output_path)
