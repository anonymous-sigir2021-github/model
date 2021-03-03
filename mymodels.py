from mymodel.mylayers import my_subquery_attention_layer, get_word_embedding_layer, myReverseGradientLayer
from mymodel.myconfig import Model_Config, Runtime_Settings
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as K
import numpy as np
import copy, collections

class my_embedding_submodel(K.models.Model):
	def __init__(self, embeddings, *args, **kwargs):
		super().__init__(name='Embed_sM', *args, **kwargs)
		self.sent_start_idx = len(embeddings) - 1
		self.embedding_layer = get_word_embedding_layer(embeddings)

	def call(self, inputs, training=None, mask=None):
		emb = self.embedding_layer(inputs)
		rnn_zero_mask = tf.not_equal(inputs, 0)
		sent_mask = tf.not_equal(inputs, self.sent_start_idx)
		true_mask = tf.math.logical_and(rnn_zero_mask, sent_mask)

		return emb, rnn_zero_mask, true_mask

class my_encoder_submodel(K.models.Model):
	def __init__(self, config, *args, **kwargs):
		super().__init__(name='Encoder_sM', *args, **kwargs)
		self.config = config
		self.encoder_layer = K.layers.Dense(
			self.config.word_projection_dimension,
			activation='tanh',
			# kernel_regularizer=K.regularizers.l2(self.config.l2_regularization_loss_weight),
			# bias_regularizer=K.regularizers.l2(self.config.l2_regularization_loss_weight),
		)
	def call(self, inputs, training=None, mask=None):
		proj = self.encoder_layer(inputs)
		return proj
	
	def my_save_weights(self, fname):
		fname += '_encoder_submodel'
		self.save_weights(fname)
	
	def my_load_weights(self, fname):
		fname += '_encoder_submodel'
		self.load_weights(fname)
	
class my_encoder_submodel2(K.models.Model):
	def __init__(self, config, *args, **kwargs):
		super().__init__(name='Encoder_sM2', *args, **kwargs)
		self.config = config
		self.encoder_layer1 = K.layers.Dense(
			self.config.word_projection_dimension,
			activation='tanh',
		)
		self.encoder_layer2 = K.layers.Dense(
			self.config.word_projection_dimension,
			activation='tanh',
		)
	def call(self, inputs, training=None, mask=None):
		proj1 = self.encoder_layer1(inputs)
		proj = self.encoder_layer2(proj1)
		return proj
	
	def my_save_weights(self, fname):
		fname += '_encoder_submodel2'
		self.save_weights(fname)
		
	def my_load_weights(self, fname):
		fname += '_encoder_submodel2'
		self.load_weights(fname)

class my_document_based_attention_submodel(K.models.Model): # add query dimension
	def __init__(self, config: Model_Config, query_setting, *args, **kwargs):
		super().__init__(name='DocBased_Q_sM', *args, **kwargs)
		self.config = config
		self.query_setting = query_setting
		self.attention_layer = my_subquery_attention_layer([len(self.query_setting), self.config.n_subquery, self.config.word_projection_dimension])

	def call(self, inputs, training=None, mask=None):
		self.debug_input = inputs
		
		*_, _n3, _n2, _n1 = tf.shape(inputs)
		re_inputs = tf.reshape(inputs, [-1, _n3 * _n2, _n1])
		
		self.debug_re_input = re_inputs
		re_mask = tf.reshape(mask, [-1, _n3 * _n2])
		masked_weight = self.attention_layer(re_inputs, mask=re_mask)

		rs = tf.reshape(masked_weight, [len(self.query_setting), self.config.n_subquery, -1, _n3, _n2])
		# self.debug_weight = rs

		return rs

class my_linear_submodel(K.models.Model): # reduce dimension at -2 axis
	def __init__(self, *args, **kwargs):
		super().__init__(name='Linear_R_sM', *args, **kwargs)

	def call(self, inputs, mask=None): # should be the same with or without mask
		fmask = tf.cast(mask, 'float32') # b mask
		count = tf.reduce_sum(fmask, axis=-1, keepdims=True) + 1e-20 # b 1
		red = tf.reduce_sum(tf.multiply(inputs, tf.expand_dims(fmask, -1)), -2) / count
		return red


class my_decoder_submodel(K.models.Model):
	def __init__(self, config: Model_Config, task_name: str, nclass, model_softmax=True, *args, **kwargs):
		super().__init__(name=task_name + 'Decoder_sM', *args, **kwargs)
		self.config = config
		self.nclass = nclass
		self.model_softmax = model_softmax
		self.decoder = K.layers.Dense(
			nclass,
			# kernel_initializer=K.initializers.constant(decoder_kernel),
			# bias_initializer=K.initializers.constant(decoder_bias),
			kernel_regularizer=K.regularizers.l2(self.config.l2_regularization_loss_weight),
			bias_regularizer=K.regularizers.l2(self.config.l2_regularization_loss_weight),
			# trainable=False,
		)
	
	def call(self, inputs, training=None, mask=None):
		out = self.decoder(inputs)
		if self.model_softmax:
			out = tf.nn.softmax(out, -1)
		return out
	
	def my_save_weights(self, fname):
		fname += '_decoder_submodel'
		self.save_weights(fname)
	
	def my_load_weights(self, fname):
		fname += '_decoder_submodel'
		self.load_weights(fname)


class my_fix_weight_dense_submodel(K.models.Model):
	def __init__(self, decoder: my_decoder_submodel, model_softmax=True, *args, **kwargs):
		super().__init__(name='fix_weight_sM', *args, **kwargs)
		self.dense_layer = decoder.decoder
		self.model_softmax = model_softmax

	def call(self, inputs, ):
		kernel = tf.stop_gradient(self.dense_layer.kernel)
		bias = tf.stop_gradient(self.dense_layer.bias)
		output = inputs @ kernel + bias
		if self.model_softmax:
			output = tf.nn.softmax(output, axis=-1)
		return output

class my_domain_output_layer(K.layers.Layer):
	def __init__(self, settings: Runtime_Settings, name='DomReW_L', **kwargs):
		super().__init__(name=name, **kwargs)
		self.settings = settings
		nclasses = settings.model_config.domain_classes
		self.nclasses = nclasses
		
		other_v = 1. / (nclasses - 1)
		up = np.eye(nclasses, nclasses, dtype='float32')
		down = np.ones([nclasses, nclasses], dtype='float32')
		down = down - up
		down *= other_v
		matrix = np.concatenate([up, down], axis=0) / nclasses
		self.matrix = tf.constant(matrix)
		# self.matrix = self.add_weight(
		# 	shape=[nclasses * 2, nclasses],
		# 	dtype='float32',
		# 	initializer=K.initializers.constant(matrix),
		# 	trainable=False
		# )
		
	def call(self, inputs, **kwargs):
		re_inp = tf.reshape(tf.transpose(inputs, [1, 2, 0]), [-1, self.nclasses * 2])
		output = tf.matmul(re_inp, self.matrix)
		return output

@tf.function
def _clip_dirichlet_parameters(x):
	return tf.clip_by_value(x, 1e-3, 1e3)

class my_tfp_lda_encoder(K.models.Model):

	def __init__(self, settings: Runtime_Settings, features):
		super(my_tfp_lda_encoder, self).__init__()
		self.settings = settings
		self.input_layer = K.layers.InputLayer(
			[settings.model_config.word_projection_dimension],
			dtype='float32',
		)
		self.dense_layers = [K.layers.Dense(
			n_features,
			activation='tanh',
		) for n_features in features[:-1]]
		self.last_layer = K.layers.Dense(
			features[-1] + 1,
			activation=lambda x: _clip_dirichlet_parameters(tf.nn.softplus(x)),
		)
		self.m = K.models.Sequential(self.dense_layers)

	def call(self, inputs):
		inputs = self.input_layer(inputs)
		mid_outputs = self.m(inputs)
		mid_outputs_norm = tf.nn.softmax(mid_outputs, axis=-1)
		outputs = self.last_layer(mid_outputs)
		alphas = outputs[:, :-1]
		mus = outputs[:, -1]
		return alphas, mus, mid_outputs_norm


class my_tfp_lda_decoder(K.models.Model):
	def __init__(self, settings: Runtime_Settings, features, *args, **kwargs):
		super().__init__(name='LDA_Dec_M', *args, **kwargs)
		self.settings = settings
		self.input_layer = K.layers.InputLayer(
			[features[-1]],
			dtype='float32',
		)
		self.dense_layers = [K.layers.Dense(
			n_features,
			activation='tanh',
		) for n_features in features]
		self.m = K.models.Sequential(self.dense_layers)
		self.betas = self.add_weight(
			name='betas',
			shape=[features[-1], self.settings.model_config.embedding_dimension],
			dtype='float32',
		)

	def call(self, inputs):
		inputs = self.input_layer(inputs)
		thetas = inputs  # batch, ntopics, may not sum to one
		thetas_ = self.m(thetas)
		return tf.matmul(thetas_, self.betas)  # batch, nwords

class my_model(K.models.Model):
	def __init__(self,
	             settings: Runtime_Settings,
	             reverse_gradient_settings,
	             task_name: str,
				 embed_sm: my_embedding_submodel,
	             encoder: my_encoder_submodel,
				 query_w_layer,
				 word_layer,
	             sent_layer,
				 decoder: my_decoder_submodel,
				 # weight_transfer_model: weight_transfer_model,
				 # aspect_model: aspect_model,
				 nclasses: int,
	             fix_weight_dense_submodel,
				 *args, **kwargs):
		super().__init__(name=task_name + '_M', *args, **kwargs)
		self.settings = settings
		self.reverse_gradient_settings = reverse_gradient_settings
		self.task_name = task_name

		self.input_layer = K.layers.InputLayer(
			input_shape=[settings.data_config.max_sent, settings.data_config.max_word],
			# batch_size=settings.batch_size,
			dtype='int32',
			sparse=False
		)
		self.embed_sm = embed_sm
		self.encoder = encoder
		self.query_w_layer = query_w_layer

		self.word_layer = word_layer
		self.sent_layer = sent_layer
		
		self.decoder = decoder
		self.fixed_weight_decoder = fix_weight_dense_submodel

		self.grl_adapt = settings.grl_adapt
		if self.grl_adapt is not None:
			self.grl_layer = myReverseGradientLayer(self.grl_adapt)
		
		self.debug_weight = None
		
		self.special_call_output = None
		
		# self.weight_transfer_model = weight_transfer_model
		# self.aspect_model = aspect_model
		self.nclasses = nclasses
		
		@tf.function
		def my_normal_loss_function(ytrue, ypred, weight=None, model_softmax=settings.use_submodel_softmax):
			if not model_softmax:
				ypred = tf.nn.softmax(ypred, axis=-1)
			all_loss = K.losses.sparse_categorical_crossentropy(ytrue, ypred, from_logits=False)
			if weight is None:
				return tf.reduce_mean(all_loss)
			else:
				return tf.reduce_sum(tf.multiply(all_loss, weight)) / tf.reduce_sum(weight)
		
		@tf.function
		def my_adv_loss_function(ypred, weight=None, model_softmax=settings.use_submodel_softmax):
			if not model_softmax:
				ypred = tf.nn.softmax(ypred, axis=-1)
			all_loss = tf.reduce_sum(tf.math.square(ypred - 1. / self.nclasses), -1)
			if weight is None:
				return tf.reduce_mean(all_loss)
			else:
				return tf.reduce_sum(tf.multiply(all_loss, weight)) / tf.reduce_sum(weight)
		@tf.function
		def my_normal_binary_loss_function(ytrue, ypred, weight=None, model_softmax=settings.use_submodel_softmax):
			if not model_softmax:
				ypred = tf.nn.softmax(ypred, axis=-1)
			all_loss = K.losses.binary_crossentropy(ytrue, ypred, from_logits=False)
			if weight is None:
				return tf.reduce_mean(all_loss)
			else:
				return tf.reduce_sum(tf.multiply(all_loss, weight)) / tf.reduce_sum(weight)
		@tf.function
		def my_categorical_loss_function(ytrue, ypred, weight=None, model_softmax=settings.use_submodel_softmax):
			if not model_softmax:
				ypred = tf.nn.softmax(ypred, axis=-1)
			all_loss = tf.reduce_sum(-tf.math.xlogy(ytrue, ypred), axis=-1)
			if weight is None:
				return tf.reduce_mean(all_loss)
			else:
				return tf.reduce_sum(tf.multiply(all_loss, weight)) / tf.reduce_sum(weight)


		self.my_loss_fn = my_normal_loss_function
		self.my_adv_loss_fn = my_adv_loss_function
		self.my_continuous_loss_fn = my_categorical_loss_function
		self.my_metrics = [K.metrics.SparseCategoricalAccuracy() for _ in self.settings.query_settings]
		self.my_continuous_metrics = [K.metrics.CategoricalAccuracy() for _ in self.settings.query_settings]
		self.loss_tracker = K.metrics.Mean()
		self.task_loss_tracker = [K.metrics.Mean() for _ in self.settings.query_settings]
		self.aspect_loss_tracker = K.metrics.Mean()
		self.other_losses_tracker = K.metrics.Mean()

	def special_call(self, inputs):
		inputs = self.input_layer(inputs)
		emb, rnn_zero_mask, true_mask = self.embed_sm(inputs)
		proj = self.encoder(emb)
		sent_mask = tf.reduce_any(true_mask, axis=-1)
		
		m_weights = self.query_w_layer(proj, mask=true_mask)
		
		if self.settings.apply_weight_transfer:
			return m_weights, proj
		else:
			self.debug_weight = m_weights
			w_proj = tf.multiply(tf.expand_dims(proj, 0), tf.expand_dims(m_weights, -1))
			# tf.print('wproj', w_proj.shape, w_proj)
			# tf.print('zero mask', rnn_zero_mask.shape, rnn_zero_mask)
			
			re_w_proj = tf.reshape(w_proj, [-1, self.settings.model_config.max_word, self.settings.model_config.word_projection_dimension])
			add_nq_rnn_zero_mask = tf.tile(rnn_zero_mask, [len(self.settings.query_settings), 1, 1])
			re_rnn_zero_mask = tf.reshape(add_nq_rnn_zero_mask, [-1, self.settings.model_config.max_word])
			# tf.print('re_w_proj', re_w_proj)
			# tf.print('re_rnn_zero_mask', re_rnn_zero_mask)
			
			sent_rep = self.word_layer(re_w_proj, mask=re_rnn_zero_mask)
			# tf.print('sent_rep', sent_rep)
			
			re_sent_rep = tf.reshape(sent_rep, [-1, self.settings.model_config.max_sent, self.settings.model_config.word_projection_dimension])
			aad_nq_sent_mask = tf.tile(sent_mask, [len(self.settings.query_settings), 1])
			
			doc_reps = self.sent_layer(re_sent_rep, mask=aad_nq_sent_mask)
			# tf.print('doc_reps', doc_reps)
			doc_reps = tf.reshape(doc_reps, [len(self.settings.query_settings), -1, self.settings.model_config.word_projection_dimension])
			# tf.print('doc_reps', doc_reps)
			
			return self.settings.domain_only_function_word_ratio * doc_reps[1] + (1 - self.settings.domain_only_function_word_ratio) * doc_reps[2]
	
	def call(self, inputs, call_aspect_model=False, weight_transfer_input=None, get_aspect_model_output=False):
		inputs = self.input_layer(inputs)
		_batch_size, _num_sent, _num_word = tf.shape(inputs)
		# inputs = tf.sparse.to_dense(inputs)
		emb, rnn_zero_mask, true_mask = self.embed_sm(inputs)

		proj = self.encoder(emb)
		sent_mask = tf.reduce_any(true_mask, axis=-1)
		
		m_weights = self.query_w_layer(proj, mask=true_mask)
		
		
		self.debug_weight = m_weights
		w_proj = tf.multiply(proj[tf.newaxis, tf.newaxis, :, :, :, :], tf.expand_dims(m_weights, -1))
		
		# tf.print('wproj', w_proj.shape, w_proj)
		# tf.print('zero mask', rnn_zero_mask.shape, rnn_zero_mask)
		
		re_w_proj = tf.reshape(w_proj, [-1, _num_word, self.settings.model_config.word_projection_dimension])
		add_nq_rnn_zero_mask = tf.tile(rnn_zero_mask[tf.newaxis, tf.newaxis, :, :, :], [len(self.settings.query_settings), self.settings.model_config.n_subquery, 1, 1, 1])
		re_rnn_zero_mask = tf.reshape(add_nq_rnn_zero_mask, [-1, _num_word])
		# tf.print('re_w_proj', re_w_proj)
		# tf.print('re_rnn_zero_mask', re_rnn_zero_mask)
		
		sent_rep = self.word_layer(re_w_proj, mask=re_rnn_zero_mask)
		# tf.print('sent_rep', sent_rep)
		
		re_sent_rep = tf.reshape(sent_rep, [-1, _num_sent, self.settings.model_config.word_projection_dimension])
		aad_nq_sent_mask = tf.tile(sent_mask[tf.newaxis, tf.newaxis, :, :], [len(self.settings.query_settings), self.settings.model_config.n_subquery, 1, 1])
		re_nq_sent_mask = tf.reshape(aad_nq_sent_mask, [-1, _num_sent])
		
		doc_reps = self.sent_layer(re_sent_rep, mask=re_nq_sent_mask)
		# tf.print('doc_reps', doc_reps)
		doc_reps = tf.reshape(doc_reps, [len(self.settings.query_settings), self.settings.model_config.n_subquery, -1, self.settings.model_config.word_projection_dimension])
		# tf.print('doc_reps', doc_reps)
		
		
		
		# if call_aspect_model:
		# 	if self.settings.apply_weight_transfer and weight_transfer_input is not None:
		# 		dep_parent, dep_mask = weight_transfer_input
		# 		aspect_model_input = self.weight_transfer_model((m_weights, proj, dep_parent, dep_mask))
		# 	else:
		# 		aspect_model_input = self.special_call_output = self.settings.domain_only_function_word_ratio * doc_reps[1] + (1 - self.settings.domain_only_function_word_ratio) * doc_reps[2]
		#
		# 	asp_model_output = self.aspect_model(aspect_model_input)
		# 	if get_aspect_model_output:
		# 		return asp_model_output
		
		out0, out1, out2 = None, None, None
		reverse_gradient_out0, reverse_gradient_out1, reverse_gradient_out2 = None, None, None
		_unstack_output = tf.unstack(doc_reps, axis=0)
		if len(self.settings.query_settings) == 3:
			out0, out1, out2 = _unstack_output
		elif len(self.settings.query_settings) == 2:
			out0, out1 = _unstack_output
			
		if self.grl_adapt is not None:
			if self.reverse_gradient_settings[0]:
				_out0 = self.decoder(self.grl_layer(out0))
			if self.reverse_gradient_settings[1]:
				_out1 = self.decoder(self.grl_layer(out1))
			if out2 is not None:
				if self.reverse_gradient_settings[2]:
					_out2 = self.decoder(self.grl_layer(out2))
		else:
			if self.reverse_gradient_settings[0]:
				_out0 = self.decoder(tf.stop_gradient(out0))
				reverse_gradient_out0 = self.fixed_weight_decoder(out0)
			else:
				_out0 = self.decoder(out0)
			if self.reverse_gradient_settings[1]:
				_out1 = self.decoder(tf.stop_gradient(out1))
				reverse_gradient_out1 = self.fixed_weight_decoder(out1)
			else:
				_out1 = self.decoder(out1)
			if out2 is not None:
				if self.reverse_gradient_settings[2]:
					_out2 = self.decoder(tf.stop_gradient(out2))
					reverse_gradient_out2 = self.fixed_weight_decoder(out2)
				else:
					_out2 = self.decoder(out2)
		
		_out0 = tf.reduce_mean(_out0, axis=0)
		_out1 = tf.reduce_mean(_out1, axis=0)
		reverse_gradient_out0 = tf.reduce_mean(reverse_gradient_out0, axis=0) if reverse_gradient_out0 is not None else None
		reverse_gradient_out1 = tf.reduce_mean(reverse_gradient_out1, axis=0) if reverse_gradient_out1 is not None else None
		if out2 is not None:
			_out2 = tf.reduce_mean(_out2, axis=0)
			reverse_gradient_out2 = tf.reduce_mean(reverse_gradient_out2, axis=0) if reverse_gradient_out2 is not None else None

		if len(self.settings.query_settings) == 3:
			return (_out0, _out1, _out2), (reverse_gradient_out0, reverse_gradient_out1, reverse_gradient_out2)
		elif len(self.settings.query_settings) == 2:
			return (_out0, _out1), (reverse_gradient_out0, reverse_gradient_out1)
	
	@property
	def metrics(self):
		return [self.loss_tracker, *self.my_metrics, *self.my_continuous_metrics, *self.task_loss_tracker, self.aspect_loss_tracker, self.other_losses_tracker]
	
	def train_step(self, data, labeled_data_weight=None, train_aspect_model=False):
		if self.settings.apply_weight_transfer:
			_word_data, _, _dep_parent, _, _dep_mask, label = data
		else:
			_word_data, label = data
		with tf.GradientTape() as tape:
			
			weight_transfer_input = None
			if self.settings.apply_weight_transfer:
				weight_transfer_input = _dep_parent, _dep_mask
			
			normal_predict, reverse_gradient_predict = self.call(_word_data, call_aspect_model=train_aspect_model, weight_transfer_input=weight_transfer_input)

			adv_loss0 = self.my_adv_loss_fn(reverse_gradient_predict[0], labeled_data_weight) if reverse_gradient_predict[0] is not None else 0
			adv_loss1 = self.my_adv_loss_fn(reverse_gradient_predict[1], labeled_data_weight) if reverse_gradient_predict[1] is not None else 0
			loss0 = self.my_loss_fn(label, normal_predict[0], labeled_data_weight)
			loss1 = self.my_loss_fn(label, normal_predict[1], labeled_data_weight)

			if self.reverse_gradient_settings[0]:
				adv_loss0 = self.settings.auxiliary_task_discount * adv_loss0
			if self.reverse_gradient_settings[1]:
				adv_loss1 = self.settings.auxiliary_task_discount * adv_loss1

			if len(self.settings.query_settings) == 3:
				adv_loss2 = self.my_adv_loss_fn(reverse_gradient_predict[2], labeled_data_weight) if reverse_gradient_predict[2] is not None else 0
				loss2 = self.my_loss_fn(label, normal_predict[2], labeled_data_weight)
				if self.reverse_gradient_settings[2]:
					adv_loss2 = self.settings.auxiliary_task_discount * adv_loss2

			aspect_model_loss = 0
			if train_aspect_model:
				aspect_model_loss = self.losses[0]
				other_losses = tf.add_n(self.losses[1:])
			else:
				other_losses = tf.add_n(self.losses)
			
			all_loss = loss0 + loss1 + adv_loss0 + adv_loss1 +\
			           aspect_model_loss + other_losses
			
			if len(self.settings.query_settings) == 3:
				all_loss += loss2 + adv_loss2

		# for v in self.trainable_variables:
		# 	print(v)
		# print()
		gradients = tape.gradient(all_loss, self.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
		
		self.loss_tracker.update_state(all_loss)
		self.aspect_loss_tracker.update_state(aspect_model_loss)
		self.other_losses_tracker.update_state(other_losses)
		
		self.task_loss_tracker[0].update_state(loss0)
		self.task_loss_tracker[1].update_state(loss1)
		self.my_metrics[0].update_state(label, normal_predict[0])
		self.my_metrics[1].update_state(label, normal_predict[1])
		if len(self.settings.query_settings) == 3:
			self.task_loss_tracker[2].update_state(loss2)
			self.my_metrics[2].update_state(label, normal_predict[2])
			return {"loss": self.loss_tracker.result(),
					"CE_0": self.my_metrics[0].result(),
					"CE_1": self.my_metrics[1].result(),
					"CE_2": self.my_metrics[2].result(),
					'Task_Loss0': self.task_loss_tracker[0].result(),
					'Task_Loss1': self.task_loss_tracker[1].result(),
					'Task_Loss2': self.task_loss_tracker[2].result(),
					'Aspect_Model_Loss': self.aspect_loss_tracker.result(),
					'Other_Losses': self.other_losses_tracker.result(),
					}
		else:
			return {"loss": self.loss_tracker.result(),
			        "CE_0": self.my_metrics[0].result(),
			        "CE_1": self.my_metrics[1].result(),
			        'Task_Loss0': self.task_loss_tracker[0].result(),
			        'Task_Loss1': self.task_loss_tracker[1].result(),
			        'Aspect_Model_Loss': self.aspect_loss_tracker.result(),
			        'Other_Losses': self.other_losses_tracker.result(),
			        }

	def special_train_step(self, sentence_dataset):
		sentence_data, sentence_label, sentence_weight = sentence_dataset
		_batch_size, label_class = tf.shape(sentence_label)
		doc_label = tf.reshape(sentence_label, [-1, self.settings.data_config.max_sent, label_class])[:, 0, :]
		with tf.GradientTape() as tape:
			normal_predict, reverse_gradient_predict = self.call(sentence_data, call_aspect_model=False, weight_transfer_input=None)
			
			adv_loss0 = self.my_adv_loss_fn(reverse_gradient_predict[0], sentence_weight) if reverse_gradient_predict[0] is not None else 0
			adv_loss1 = self.my_adv_loss_fn(reverse_gradient_predict[1], sentence_weight) if reverse_gradient_predict[1] is not None else 0
			loss0 = self.my_continuous_loss_fn(sentence_label, normal_predict[0], sentence_weight)
			loss1 = self.my_continuous_loss_fn(sentence_label, normal_predict[1], sentence_weight)

			if self.reverse_gradient_settings[0]:
				adv_loss0 = self.settings.auxiliary_task_discount * adv_loss0
			if self.reverse_gradient_settings[1]:
				adv_loss1 = self.settings.auxiliary_task_discount * adv_loss1
				
			if len(self.settings.query_settings) == 3:
				adv_loss2 = self.my_adv_loss_fn(reverse_gradient_predict[2], sentence_weight) if reverse_gradient_predict[2] is not None else 0
				loss2 = self.my_continuous_loss_fn(sentence_label, normal_predict[2], sentence_weight)
				if self.reverse_gradient_settings[2]:
					adv_loss2 = self.settings.auxiliary_task_discount * adv_loss2
					
			if self.settings.special_training_apply_main_task:
				num_sentence = tf.reduce_sum(tf.reshape(sentence_weight, [-1, self.settings.data_config.max_sent]), axis=-1)
				np0 = tf.reduce_sum(tf.reshape(tf.multiply(normal_predict[0], sentence_weight[:, tf.newaxis]), [-1, self.settings.data_config.max_sent, label_class]), axis=-2) / num_sentence[:, tf.newaxis]
				np1 = tf.reduce_sum(tf.reshape(tf.multiply(normal_predict[1], sentence_weight[:, tf.newaxis]), [-1, self.settings.data_config.max_sent, label_class]), axis=-2) / num_sentence[:, tf.newaxis]
				rgp0 = tf.reduce_sum(tf.reshape(tf.multiply(reverse_gradient_predict[0], sentence_weight[:, tf.newaxis]), [-1, self.settings.data_config.max_sent, label_class]), axis=-2) / num_sentence[:, tf.newaxis] if reverse_gradient_predict[0] is not None else None
				rgp1 = tf.reduce_sum(tf.reshape(tf.multiply(reverse_gradient_predict[1], sentence_weight[:, tf.newaxis]), [-1, self.settings.data_config.max_sent, label_class]), axis=-2) / num_sentence[:, tf.newaxis] if reverse_gradient_predict[1] is not None else None
				
				main_task_adv_loss0 = self.my_adv_loss_fn(rgp0) if rgp0 is not None else 0
				main_task_adv_loss1 = self.my_adv_loss_fn(rgp1) if rgp1 is not None else 0
				main_task_loss0 = self.my_continuous_loss_fn(doc_label, np0)
				main_task_loss1 = self.my_continuous_loss_fn(doc_label, np1)
				
				if self.reverse_gradient_settings[0]:
					main_task_adv_loss0 = self.settings.auxiliary_task_discount * main_task_adv_loss0
				if self.reverse_gradient_settings[1]:
					main_task_adv_loss1 = self.settings.auxiliary_task_discount * main_task_adv_loss1

				if len(self.settings.query_settings) == 3:
					np2 = tf.reduce_sum(tf.reshape(tf.multiply(normal_predict[2], sentence_weight[:, tf.newaxis]), [-1, self.settings.data_config.max_sent, label_class]), axis=-2) / num_sentence[:, tf.newaxis]
					rgp2 = tf.reduce_sum(tf.reshape(tf.multiply(reverse_gradient_predict[2], sentence_weight[:, tf.newaxis]), [-1, self.settings.data_config.max_sent, label_class]), axis=-2) / num_sentence[:, tf.newaxis] if reverse_gradient_predict[2] is not None else None
					main_task_adv_loss2 = self.my_adv_loss_fn(rgp2) if rgp2 is not None else 0
					main_task_loss2 = self.my_continuous_loss_fn(doc_label, np2)
					if self.reverse_gradient_settings[2]:
						main_task_adv_loss2 = self.settings.auxiliary_task_discount * main_task_adv_loss2

			aspect_model_loss = 0
			other_losses = tf.add_n(self.losses)
				
			all_loss = loss0 + loss1 + adv_loss0 + adv_loss1 +\
			       aspect_model_loss + other_losses
			if len(self.settings.query_settings) == 3:
				all_loss += loss2 + adv_loss2
				
			if self.settings.special_training_apply_main_task:
				all_loss += main_task_loss0 + main_task_loss1 + main_task_adv_loss0 + main_task_adv_loss1
				if len(self.settings.query_settings) == 3:
					all_loss += main_task_loss2 + main_task_adv_loss2


		gradients = tape.gradient(all_loss, self.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

		self.loss_tracker.update_state(all_loss)
		self.aspect_loss_tracker.update_state(aspect_model_loss)
		self.other_losses_tracker.update_state(other_losses)

		self.task_loss_tracker[0].update_state(loss0)
		self.task_loss_tracker[1].update_state(loss1)
		_metric_sentence_weight = tf.cast(tf.not_equal(sentence_weight, 0), 'float32')
		self.my_continuous_metrics[0].update_state(sentence_label, normal_predict[0], _metric_sentence_weight)
		self.my_continuous_metrics[1].update_state(sentence_label, normal_predict[1], _metric_sentence_weight)
		
		if len(self.settings.query_settings) == 3:
			self.task_loss_tracker[2].update_state(loss2)
			self.my_continuous_metrics[2].update_state(sentence_label, normal_predict[2], _metric_sentence_weight)
			return {"loss": self.loss_tracker.result(),
					"CE_0": self.my_continuous_metrics[0].result(),
					"CE_1": self.my_continuous_metrics[1].result(),
					"CE_2": self.my_continuous_metrics[2].result(),
					'Task_Loss0': self.task_loss_tracker[0].result(),
					'Task_Loss1': self.task_loss_tracker[1].result(),
					'Task_Loss2': self.task_loss_tracker[2].result(),
					'Aspect_Model_Loss': self.aspect_loss_tracker.result(),
					'Other_Losses': self.other_losses_tracker.result(),
					}
		else:
			return {"loss": self.loss_tracker.result(),
			        "CE_0": self.my_continuous_metrics[0].result(),
			        "CE_1": self.my_continuous_metrics[1].result(),
			        'Task_Loss0': self.task_loss_tracker[0].result(),
			        'Task_Loss1': self.task_loss_tracker[1].result(),
			        'Aspect_Model_Loss': self.aspect_loss_tracker.result(),
			        'Other_Losses': self.other_losses_tracker.result(),
			        }



class my_data_convert_model(K.models.Model):
	def __init__(self,
	             settings: Runtime_Settings,
	             sentiment_model: my_model,
	             domain_model: my_model,
	             *args, **kwargs):
		super().__init__(name='Comb_M', *args, **kwargs)
		self.settings = settings
		self.sentiment_model = sentiment_model
		self.domain_model = domain_model
		
		self.test_predicts = []

	def call(self, inputs, special_call=False):
		sentiment_data, domain_data = inputs
		
		sentiment_output = self.sentiment_model(sentiment_data)  # Forward pass
		domain_output = self.domain_model(domain_data, special_call=special_call)  # Forward pass
		return sentiment_output, domain_output

	def modify_gradient(self, grads, norm, noise):
		new_grads = []
		for grad in grads:
			if norm:
				grad = tf.clip_by_norm(grad, norm)
			if noise:
				grad = grad + tf.random.normal(grad.shape, stddev=noise)
			new_grads.append(grad)
		return new_grads
	
	def train_step(self, data, labeled_data_weight=None, train_aspect_model=False):
		sentiment_data, domain_data = data
		self.sentiment_model.train_step(sentiment_data, labeled_data_weight, False)
		self.domain_model.train_step(domain_data, labeled_data_weight, train_aspect_model)
	
	def test_step(self, data):
		x, y = data
		y_pred, _ = self.sentiment_model(x)
		y_pred0 = y_pred[0]
		y_pred1 = y_pred[1]
		if not self.settings.use_submodel_softmax:
			y_pred0 = tf.nn.softmax(y_pred0, axis=-1)
			y_pred1 = tf.nn.softmax(y_pred1, axis=-1)
		# loss = self.my_loss_fn(y, y_pred)
		# self.loss_tracker.update_state(loss)
		self.test_predicts.append((y_pred0.numpy(), y_pred1.numpy()))
		self.sentiment_model.my_metrics[0].update_state(y, y_pred0)
		self.sentiment_model.my_metrics[1].update_state(y, y_pred1)
		return {#'Loss': self.loss_tracker.result(),
		        'Accuracy0': self.sentiment_model.my_metrics[0].result(),
				'Accuracy1': self.sentiment_model.my_metrics[1].result(),
		}
	
	def clear_metrics(self):
		for model in [self.sentiment_model, self.domain_model]:
			for m in model.metrics:
				m.reset_states()
		self.test_predicts = []

	def get_results(self):
		return [
			[m.result().numpy() for m in self.sentiment_model.my_metrics],
			[m.result().numpy() for m in self.domain_model.my_metrics],
		]
	def get_continuous_metrics_results(self):
		return [
			[m.result().numpy() for m in self.sentiment_model.my_continuous_metrics],
			[m.result().numpy() for m in self.domain_model.my_continuous_metrics],
		]

	def get_test_results(self, all_rs=None):
		if all_rs is None:
			return self.sentiment_model.my_metrics[0].result().numpy()
		else:
			return [self.sentiment_model.my_metrics[_].result().numpy() for _ in range(all_rs)]