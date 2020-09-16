import tensorflow as tf
import numpy as np
import json
import six
import tensorflow as tf
try:
	import bert_utils
except:
	from . import bert_utils

class BertConfig(object):
	"""Configuration for `BertModel`."""

	def __init__(self,
				 vocab_size=None,
				 emb_size=None,
				 cnn_num_layers=None,
				 cnn_dilation_rates=None,
				 cnn_num_filters=None,
				 cnn_filter_sizes=None,
				 is_casual=None,
				 padding=None,
				 pooling_method=None,
				 max_position_embeddings=1024,
				 embedding_dropout=False,
				 num_hidden_layers=4,
				 scope='textcnn'):
		
		self.vocab_size = vocab_size
		self.emb_size = emb_size
		self.cnn_num_layers = cnn_num_layers
		self.cnn_dilation_rates = cnn_dilation_rates
		self.cnn_num_filters = cnn_num_filters
		self.cnn_filter_sizes = cnn_filter_sizes
		self.is_casual = is_casual
		self.padding = padding
		self.scope = scope
		self.embedding_dropout = embedding_dropout
		self.pooling_method = pooling_method
		self.max_position_embeddings = max_position_embeddings
		self.num_hidden_layers = cnn_num_layers

	@classmethod
	def from_dict(cls, json_object):
		"""Constructs a `BertConfig` from a Python dictionary of parameters."""
		config = BertConfig(vocab_size=None)
		for (key, value) in six.iteritems(json_object):
			config.__dict__[key] = value
		return config

	@classmethod
	def from_json_file(cls, json_file):
		"""Constructs a `BertConfig` from a json file of parameters."""
		with tf.gfile.GFile(json_file, "r") as reader:
			text = reader.read()
		return cls.from_dict(json.loads(text))

	def to_dict(self):
		"""Serializes this instance to a Python dictionary."""
		output = copy.deepcopy(self.__dict__)
		return output

	def to_json_string(self):
		"""Serializes this instance to a JSON string."""
		return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class LightDGCNN(object):
	def __init__(self, config,
					 is_training,
					 input_ids, 
					 reuse=None,
					 num_train_steps=None,
					 **kargs):
		self.config = config
		self.vocab_size = int(self.config.vocab_size)
		self.emb_size = int(self.config.emb_size)
		self.scope = self.config.scope
		self.emb_dropout_count = 0

		init_mat = np.random.uniform(-0.1, 0.1, (self.vocab_size, self.emb_size))
		with tf.variable_scope(self.scope+"_token_embedding", reuse=reuse):
			self.emb_mat = tf.get_variable("emb_mat",
									[self.vocab_size, self.emb_size],
									dtype=tf.float32,
									initializer=tf.constant_initializer(init_mat, 
												dtype=tf.float32))
		
		if is_training:
			dropout_rate = self.config.dropout_rate
		else:
			dropout_rate = 0.0

		if self.config.embedding_dropout and is_training:
			embedding_matrix = tf.nn.dropout(self.emb_mat, 
										keep_prob=1-self.config.embedding_dropout, 
										noise_shape=[self.vocab_size,1])
			tf.logging.info("***** word drop out *****")
		else:
			embedding_matrix = self.emb_mat
			tf.logging.info("***** none word drop *****")

		self.word_emb = tf.nn.embedding_lookup(embedding_matrix, input_ids)

		if is_training:
			dropout_rate = self.config.dropout_rate
		else:
			dropout_rate = 0.0

		sent_repres = tf.nn.dropout(self.word_emb, 1)

		input_mask = tf.cast(tf.not_equal(input_ids, kargs.get('[PAD]', 0)), tf.int32)
		input_len = tf.reduce_sum(tf.cast(input_mask, tf.int32), -1)

		mask = tf.expand_dims(input_mask, -1)
		sent_repres *= tf.cast(mask, tf.float32)
		self.sent_repres = sent_repres

		with tf.variable_scope(self.config.scope+"_encoder", reuse=reuse):
			self.sequence_output = dgcnn(
								sent_repres, 
								input_mask,
								num_layers=self.config.cnn_num_layers, 
								dilation_rates=self.config.cnn_dilation_rates,
								strides=self.config.cnn_dilation_rates,
								num_filters=self.config.cnn_num_filters,
								kernel_sizes=self.config.cnn_filter_sizes,
								is_training=is_training,
								scope_name="textcnn/forward", 
								reuse=False, 
								activation=tf.nn.relu,
								is_casual=self.config.is_casual,
								padding=self.config.padding
								)
			pooled_output = []
			if self.config.is_casual:
				self.forward_backward_repres = self.sequence_output[:,:-2]
				seq_mask = tf.cast(input_mask[:, 2:], dtype=tf.int32)
				tf.logging.info("***** casual concat *****")
			else:
				self.forward_backward_repres = self.sequence_output
				tf.logging.info("***** none-casual concat *****")
				seq_mask = tf.cast(input_mask, dtype=tf.int32)

			input_mask = tf.cast(input_mask, tf.float32)
			for pooling_method in self.config.pooling_method:
				if pooling_method == 'avg':
					avg_repres = mean_pooling(self.forward_backward_repres, 
															seq_mask)
					pooled_output.append(avg_repres)
					tf.logging.info("***** avg pooling *****")
				elif pooling_method == 'max':
					max_repres = max_pooling(self.forward_backward_repres, 
															seq_mask)
					pooled_output.append(max_repres)
					tf.logging.info("***** max pooling *****")
				
			self.output = tf.concat(pooled_output, axis=-1)

	def get_pooled_output(self):
		return self.output

	def get_sequence_output(self):
		return self.sequence_output

	def get_embedding_table(self):
		return self.emb_mat

def mean_pooling(tensor, mask):
	mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
	avg_out = tf.reduce_sum(tensor*mask, axis=1)/(1e-10+tf.reduce_sum(mask, axis=1))
	return avg_out
	
def mask_logits(inputs, mask, mask_value = -1e30):
	shapes = inputs.shape.as_list()
	mask = tf.cast(mask, tf.float32)
	return mask * inputs + mask_value * (1 - mask)

def max_pooling(tensor, mask):
	mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
	max_out = tf.reduce_max(mask_logits(tensor, mask), axis=1)
	return max_out

initializer = tf.glorot_uniform_initializer()
initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
															 mode='FAN_AVG',
															 uniform=True,
															 dtype=tf.float32)

def layer_norm_compute_python(x, epsilon, scale, bias):
	"""Layer norm raw computation."""
	mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
	variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
	norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
	return norm_x * scale + bias

def layer_norm(x, filters=None, epsilon=1e-6, scope=None, reuse=None):
	"""Layer normalize the tensor x, averaging over the last dimension."""
	if filters is None:
		filters = x.get_shape()[-1]
	with tf.variable_scope(scope, default_name="layer_norm", values=[x], reuse=reuse):
		scale = tf.get_variable(
			"layer_norm_scale", [filters], initializer=tf.ones_initializer())
		bias = tf.get_variable(
			"layer_norm_bias", [filters], initializer=tf.zeros_initializer())
		result = layer_norm_compute_python(x, epsilon, scale, bias)
		return result

def create_initializer(initializer_range=0.02):
	"""Creates a `truncated_normal_initializer` with the given range."""
	return tf.truncated_normal_initializer(stddev=initializer_range)

def depthwise_separable_convolution(inputs, kernel_size, num_filters,
									scope = "depthwise_separable_convolution",
									padding='SAME',
									bias = True, 
									is_training = True, 
									reuse = None,
									activation = tf.nn.relu,
									kernel_initializer = None,
									strides = 1,
									dilation_rate = 1):
	outputs = tf.expand_dims(inputs, 2) # batch, seq, 1, dim
	shapes = bert_utils.get_shape_list(outputs, expected_rank=[4])
	with tf.variable_scope(scope, reuse = reuse):
		depthwise_filter = tf.get_variable("depthwise_filter",
										(kernel_size[0], kernel_size[1], shapes[-1], 1),
										dtype = tf.float32,
										initializer = kernel_initializer)
		pointwise_filter = tf.get_variable("pointwise_filter",
										(1, 1, shapes[-1], num_filters),
										dtype = tf.float32,
										initializer = kernel_initializer)

		outputs = tf.nn.separable_conv2d(outputs,
										depthwise_filter,
										pointwise_filter,
										strides = (1, strides, 1, 1),
										padding = padding,
										rate = (dilation_rate, 1))
		if bias:
			b = tf.get_variable("bias",
					outputs.shape[-1],
					# regularizer=regularizer,
					initializer = tf.zeros_initializer())
			outputs += b
		if activation:
			outputs = activation(outputs)

		return tf.squeeze(outputs,2)

def gated_conv1d_op(inputs, 
				filters=8, 
				kernel_size=3, 
				padding="same", 
				activation=None, 
				strides=1, 
				reuse=False, 
				name="", 
				kernel_initializer=None,
				dilation_rate=1,
				is_training=True):
	conv_linear = depthwise_separable_convolution(
		inputs = inputs,
		kernel_size = [kernel_size, 1],
		num_filters = filters,
		scope = name+"_linear",
		padding=padding,
		bias = True, 
		is_training = True, 
		reuse = reuse,
		activation = None,
		kernel_initializer = kernel_initializer,
		strides = strides,
		dilation_rate = dilation_rate,
		)
	conv_gated = depthwise_separable_convolution(
		inputs = inputs,
		kernel_size = [kernel_size, 1],
		num_filters = filters,
		scope = name+"_gated",
		padding=padding,
		bias = True, 
		is_training = True, 
		reuse = reuse,
		activation = tf.nn.sigmoid,
		kernel_initializer = kernel_initializer,
		strides = strides,
		dilation_rate = dilation_rate,
		)
	conv = conv_linear * conv_gated
	return conv

def residual_gated_conv1d_op(inputs, 
							residual_inputs,
							filters=8, kernel_size=3, 
							padding='same',
							activation=None, 
							strides=1, 
							reuse=False, 
							dilation_rate=1,
							name="",
							kernel_initializer=None, 
							is_training=False):
	conv_linear = depthwise_separable_convolution(
		inputs = inputs,
		kernel_size = [kernel_size, 1],
		num_filters = filters,
		scope = name+"_linear",
		padding=padding,
		bias = True, 
		is_training = True, 
		reuse = reuse,
		activation = None,
		kernel_initializer = kernel_initializer,
		strides = strides,
		dilation_rate = dilation_rate,
		)
	conv_gated = depthwise_separable_convolution(
		inputs = inputs,
		kernel_size = [kernel_size, 1],
		num_filters = filters,
		scope = name+"_gated",
		padding=padding,
		bias = True, 
		is_training = True, 
		reuse = reuse,
		activation = None,
		kernel_initializer = kernel_initializer,
		strides = strides,
		dilation_rate = dilation_rate,
		)
	if is_training:
		dropout_rate = 0.1
	else:
		dropout_rate = 0.0
	conv_gated = tf.nn.sigmoid(tf.nn.dropout(conv_gated, 1-dropout_rate))
	conv = residual_inputs * (1. - conv_gated) + conv_linear * conv_gated
	return conv

def dgcnn(x, input_mask,
			num_layers=2, 
			dilation_rates=[1,2],
			strides=[1,1],
			num_filters=[64,64],
			kernel_sizes=[3,3], 
			is_training=False,
			scope_name="textcnn", 
			reuse=False, 
			activation=tf.nn.relu,
			is_casual=False,
			padding='SAME'
			):

	print(num_filters, '===num_filters===')

	# input_mask: batch_size, seq

	# initializer = tf.glorot_uniform_initializer()
	initializer = create_initializer(initializer_range=0.02)

	input_mask = tf.cast(input_mask, dtype=tf.float32)
	input_mask = tf.expand_dims(input_mask, axis=-1)

	padding_type = padding

	if is_casual:
		left_pad = dilation_rates[0] * (kernel_sizes[0] - 1)
		inputs = tf.pad(x, [[0, 0], [left_pad, 0], [0, 0]])
		padding = 'VALID'
		tf.logging.info("==casual valid padding==")
	else:
		inputs = x
		# left_pad = int(dilation_rates[0] * (kernel_sizes[0] - 1) / 2)
		# right_pad = int(dilation_rates[0] * (kernel_sizes[0] - 1) / 2)
		# print(left_pad, right_pad, '===projection===')
		# inputs = tf.pad(x, [[0, 0], [left_pad, right_pad], [0, 0]])
		# padding = 'VALID'
		padding = 'SAME'

	if is_training:
		dropout_rate = 0.1
	else:
		dropout_rate = 0.0

	with tf.variable_scope(scope_name, reuse=reuse):
		inputs = gated_conv1d_op(inputs,
						filters=num_filters[0],
						kernel_size=kernel_sizes[0],
						padding=padding,
						activation=None,
						strides=1,
						reuse=reuse, 
						dilation_rate=1,
						name="gated_conv",
						kernel_initializer=initializer, #tf.truncated_normal_initializer(stddev=0.1),
						is_training=is_training)
		if padding_type == 'SAME':
			inputs *= input_mask
		residual_inputs = inputs

	for (dilation_rate, 
		layer, 
		kernel_size, 
		stride, 
		num_filter) in zip(dilation_rates, 
							range(num_layers), 
							kernel_sizes,
							strides, 
							num_filters):
		layer_scope_name = "%s_layer_%s"%(str(scope_name), str(layer))
		output_shape = bert_utils.get_shape_list(inputs, expected_rank=3)
		with tf.variable_scope(layer_scope_name, reuse=reuse):
			if dilation_rate > 1:
				stride = 1
			if not is_casual:
				padding = padding
				# padding = 'VALID'
				# left_pad = int(dilation_rate * (kernel_sizes[0] - 1) / 2)
				# right_pad = int(dilation_rate * (kernel_sizes[0] - 1) / 2)
				# inputs = tf.pad(inputs, [[0, 0, ], [left_pad, right_pad], [0, 0]])
				# tf.logging.info("==none-casual same padding==")
				# print(left_pad, right_pad, '===projection===')
			else:
				left_pad = dilation_rate * (kernel_size - 1)
				inputs = tf.pad(inputs, [[0, 0, ], [left_pad, 0], [0, 0]])
				padding = 'VALID'
				tf.logging.info("==casual valid padding==")

			tf.logging.info("==kernel_size:%s, num_filter:%s, stride:%s, dilation_rate:%s==", str(kernel_size), 
										str(num_filter), str(stride), str(dilation_rate))
			gatedcnn_outputs = residual_gated_conv1d_op(inputs,
									residual_inputs,
									filters=num_filter, 
									kernel_size=kernel_size, 
									padding=padding, 
									activation=None, 
									strides=stride, 
									reuse=False, 
									dilation_rate=dilation_rate,
									name="residual_gated_conv",
									kernel_initializer=initializer, #tf.truncated_normal_initializer(stddev=0.1), 
									is_training=is_training)

			# The activation is only applied to the "intermediate" hidden layer.
			with tf.variable_scope("intermediate"):
				intermediate_output = tf.layers.dense(
						gatedcnn_outputs,
						num_filter*4,
						activation=tf.nn.relu,
						kernel_initializer=create_initializer(0.02))

			# Down-project back to `hidden_size` then add the residual.
			with tf.variable_scope("output"):
				layer_output = tf.layers.dense(
						intermediate_output,
						num_filter,
						kernel_initializer=create_initializer(0.02))

			layer_output = tf.nn.dropout(layer_output, 1-dropout_rate)
			inputs = layer_norm(layer_output + gatedcnn_outputs)

			if padding_type == 'SAME':
				inputs *= input_mask
			residual_inputs = inputs
	
	return inputs
