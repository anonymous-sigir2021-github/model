import tensorflow as tf
import tensorflow.keras as K

class my_weight_layer(K.layers.Layer):
	def __init__(self, myshape, name='Weight_L', **kwargs):
		super().__init__(name=name, **kwargs)
		self.myshape = myshape
		self.querys = self.add_weight(
			name='querys',
			shape=myshape,
			dtype=tf.float32,
			initializer=tf.initializers.Orthogonal(),
			trainable=True
		)

	def call(self, inputs):
		projection = inputs
		# proj: batch, sent, word, word proj
		# mask: batch sent, word

		weight = tf.tensordot(self.querys, projection, axes=[[-1], [-1]])
		# weight: num query, batch, sent, word

		# max_weight = tf.reduce_max(weight, axis=-1, keepdims=True)
		# weight = weight - max_weight


		# exp_weight = tf.exp(weight)

		return weight

class my_attention_layer(K.layers.Layer):
	def __init__(self, myshape, name='MaskAtt_L', **kwargs):
		super().__init__(name=name, **kwargs)
		self.myshape = myshape
		self.querys = self.add_weight(
			name='querys',
			shape=myshape,
			dtype=tf.float32,
			initializer=tf.initializers.Orthogonal(),
			trainable=True
		)

	def call(self, inputs, mask=None):
		projection = inputs
		#proj: batch, sent, word, word proj
		#mask: batch sent, word

		mask = tf.cast(tf.expand_dims(mask, 0), dtype='float32')
		# mask: 1, batch sent, word
		weight = tf.tensordot(self.querys, projection, axes=[[-1], [-1]])
		num_word = tf.reduce_sum(mask, -1, keepdims=True)
		#weight: num query, batch, sent, word
		
		max_weight = tf.reduce_max(weight, axis=-1, keepdims=True)
		exp = tf.multiply(tf.exp(weight - max_weight), mask)
		norm = tf.reduce_sum(exp, axis=-1, keepdims=True) + 1e-20
		masked_weight = exp / norm * num_word

		# num query, batch, sent, word
		word_topic_distribution = tf.multiply(tf.nn.softmax(weight, axis=0), mask) # todo need to be modified

		return masked_weight, word_topic_distribution

class my_subquery_attention_layer(K.layers.Layer):
	def __init__(self, myshape, name='SqMaskAtt_L', **kwargs):
		super().__init__(name=name, **kwargs)
		self.myshape = myshape
		self.querys = self.add_weight(
			name='querys',
			shape=myshape,
			dtype=tf.float32,
			initializer=tf.initializers.Orthogonal(),
			trainable=True
		)
	
	def call(self, inputs, mask=None):
		projection = inputs
		# proj: batch, sent, word, word proj
		# mask: batch sent, word
		
		mask = tf.cast(mask[tf.newaxis, tf.newaxis, :, :], dtype='float32')
		# mask: 1, 1, batch sent, word
		weight = tf.tensordot(self.querys, projection, axes=[[-1], [-1]])
		num_word = tf.reduce_sum(mask, -1, keepdims=True)
		# weight: num query, batch, sent, word
		
		max_weight = tf.reduce_max(weight, axis=-1, keepdims=True)
		exp = tf.multiply(tf.exp(weight - max_weight), mask)
		norm = tf.reduce_sum(exp, axis=-1, keepdims=True) + 1e-20
		masked_weight = exp / norm * num_word
		
		# num query, batch, sent, word
		# word_topic_distribution = tf.multiply(tf.nn.softmax(weight, axis=0), mask)  # todo need to be modified
		
		return masked_weight#, word_topic_distribution


def get_word_embedding_layer(embeddings):
	return K.layers.Embedding(
		input_dim=len(embeddings),
		output_dim=len(embeddings[0]),
		embeddings_initializer=K.initializers.constant(embeddings),
		trainable=False
	)

# class my_clip_gradient

class myReverseGradientLayer(K.layers.Layer):
	def __init__(self, scale=0.1, name='ReGrad'):
		super().__init__(name=name)
		self.scale = scale

	@tf.custom_gradient
	def my_grad_reverse(self, x):
		y = tf.identity(x)

		def custom_grad(dy):
			return -dy * self.scale

		return y, custom_grad

	# @tf.function
	def call(self, x):
		return self.my_grad_reverse(x)

