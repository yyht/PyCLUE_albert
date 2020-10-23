
import tensorflow as tf
import numpy as np
from model import bert_utils

def dropout(input_tensor, dropout_prob, dropout_name=None):
	"""Perform dropout.

	Args:
		input_tensor: float Tensor.
		dropout_prob: Python float. The probability of dropping out a value (NOT of
			*keeping* a dimension as in `tf.nn.dropout`).

	Returns:
		A version of `input_tensor` with dropout applied.
	"""
	if dropout_prob is None or dropout_prob == 0.0:
		return tf.identity(input_tensor)
	
	output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
	return output

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
		is_training = is_training, 
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
		is_training = is_training, 
		reuse = reuse,
		activation = tf.nn.sigmoid,
		kernel_initializer = kernel_initializer,
		strides = strides,
		dilation_rate = dilation_rate,
		)
	conv = conv_linear * conv_gated
	return conv

def dynamic_conv_layer(from_tensor,
				to_tensor,
				attention_mask=None,
				from_mask=None,
				to_mask=None,
				num_attention_heads=1,
				size_per_head=512,
				query_act=None,
				key_act=None,
				value_act=None,
				attention_probs_dropout_prob=0.0,
				initializer_range=0.02,
				do_return_2d_tensor=False,
				batch_size=None,
				from_seq_length=None,
				to_seq_length=None,
				attention_fixed_size=None,
				dropout_name=None,
				structural_attentions="none",
				is_training=False,
				kernel_size=9,
				strides=1,
				dilation_rate=1,
				hidden_size=None,
				share_or_not=True):
	
	initializer = create_initializer(initializer_range=0.02)

	def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
													 seq_length, width):
		output_tensor = tf.reshape(
				input_tensor, [batch_size, seq_length, num_attention_heads, width])

		output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
		return output_tensor

	from_shape = bert_utils.get_shape_list(from_tensor, expected_rank=[2, 3])
	to_shape = bert_utils.get_shape_list(to_tensor, expected_rank=[2, 3])

	if len(from_shape) != len(to_shape):
		raise ValueError(
				"The rank of `from_tensor` must match the rank of `to_tensor`.")

	if len(from_shape) == 3:
		batch_size = from_shape[0]
		from_seq_length = from_shape[1]
		to_seq_length = to_shape[1]
	elif len(from_shape) == 2:
		if (batch_size is None or from_seq_length is None or to_seq_length is None):
			raise ValueError(
					"When passing in rank 2 tensors to attention_layer, the values "
					"for `batch_size`, `from_seq_length`, and `to_seq_length` "
					"must all be specified.")

	# Scalar dimensions referenced here:
	#   B = batch size (number of sequences)
	#   F = `from_tensor` sequence length
	#   T = `to_tensor` sequence length
	#   N = `num_attention_heads`
	#   H = `size_per_head`

	if attention_fixed_size:
		attention_head_size = attention_fixed_size
		tf.logging.info("==apply attention_fixed_size==")
	else:
		attention_head_size = size_per_head
		tf.logging.info("==apply attention_original_size==")

	from_tensor_2d = bert_utils.reshape_to_matrix(from_tensor)
	to_tensor_2d = bert_utils.reshape_to_matrix(to_tensor)

	# query_layer = [B*F, N*H]

	# query_layer = [B*T, N*H]
	linear_query_layer = tf.layers.dense(
			from_tensor_2d,
			num_attention_heads * attention_head_size,
			activation=None,
			name="conv_query",
			kernel_initializer=create_initializer(initializer_range))

	gated_query_layer = tf.layers.dense(
			from_tensor_2d,
			num_attention_heads * attention_head_size,
			activation=tf.nn.sigmoid,
			name="conv_query_gated",
			kernel_initializer=create_initializer(initializer_range))

	query_layer = linear_query_layer * gated_query_layer

	# query_layer = [B, N, F, H]
	query_layer = transpose_for_scores(query_layer, batch_size,
									 from_seq_length,
									 num_attention_heads, 
									 attention_head_size)

	# add zeros-query-mask for conv same padding
	from_query_mask = tf.expand_dims(from_mask, axis=-1)
	from_query_mask = tf.expand_dims(from_query_mask, axis=-1)
	from_query_mask = tf.cast(from_query_mask, dtype=tf.float32)
	query_layer *= from_query_mask

	# add zeros-value-mask for conv same padding
	# from_value_mask = tf.expand_dims(from_mask, axis=-1)
	# from_value_mask = tf.expand_dims(from_value_mask, axis=-1)
	# from_value_mask = tf.cast(from_value_mask, dtype=tf.float32)
	
	# add zeros-from_tensor-mask for conv same padding
	from_tensor_mask = tf.expand_dims(from_mask, axis=-1)
	from_tensor_mask = tf.cast(from_tensor_mask, dtype=tf.float32)

	if len(from_shape) == 3:
		query_layer *= from_tensor_mask
	else:
		query_layer = tf.reshape(
				from_tensor,
				[batch_size, from_seq_length, hidden_size])
		query_layer *= from_tensor_mask

	conv_key_layer = depthwise_separable_convolution(
						inputs=query_layer,
						kernel_size=[kernel_size, 1],
						num_filters=num_attention_heads*attention_head_size,
						scope="conv_key",
						padding="SAME", 
						bias=True, 
						is_training=is_training, 
						reuse=None,
						activation=None,
						kernel_initializer=initializer,
						strides=strides,
						dilation_rate=dilation_rate
						)

	conv_key_layer *= from_tensor_mask

	# [batch_size, from_seq_length, num_attention_heads, attention_head_size]
	conv_key_layer = tf.reshape(conv_key_layer, 
												[batch_size,
												from_seq_length,
												num_attention_heads,
												attention_head_size])
	# dynamic-kernel-generator
	dynamic_kernel_generator = tf.identity(conv_key_layer)

	# kernel-project-layer
	dynamic_kernel_generator = tf.reshape(dynamic_kernel_generator, 
			[batch_size*from_seq_length, 
			num_attention_heads*attention_head_size])
	
	dynamic_conv_kernel = tf.layers.dense(dynamic_kernel_generator, 
										num_attention_heads*kernel_size,
										name="dynamic_kernel_generation",
										use_bias=False)
	
	# [batch_size, num_attention_heads, from_seq_length, attention_head_size]
	# [num_attention_heads, kernel_size, attention_head_size]
	dynamic_conv_kernel = tf.reshape(dynamic_conv_kernel, 
												[batch_size,
												from_seq_length,
												num_attention_heads,
												kernel_size])
	dynamic_conv_kernel = tf.transpose(dynamic_conv_kernel, 
										[0,2,1,3])
	# [batch_size, num_attention_heads, from_seq_length, kernel_size]
	normalized_dynamic_kernel = tf.nn.softmax(dynamic_conv_kernel, axis=-1)
	normalized_dynamic_kernel = dropout(normalized_dynamic_kernel, 
										attention_probs_dropout_prob, 
										dropout_name=dropout_name+"_conv")

	indices_i = tf.range(from_seq_length+kernel_size-1, delta=1)
	# indices = tf.reverse(indices_i[0:kernel_size], axis=[-1])
	indices = indices_i[0:kernel_size]
	indices = tf.expand_dims(indices, axis=0)

	batch_one = tf.ones((from_seq_length, 1), dtype=indices.dtype)
	batch_index = tf.einsum("ab,bc->ac", 
							tf.cast(batch_one, dtype=tf.float32),
							tf.cast(indices, dtype=tf.float32)
							)

	incremental_index = tf.transpose(tf.expand_dims(indices_i[:from_seq_length], axis=0))
	indices += tf.cast(incremental_index, indices.dtype)
	
	indices = tf.reshape(indices, [-1])

	# padded_query_layer: [batch_size, from_seq_length+kernel_size-1, num_attention_heads, attention_head_size]
	# value_layer       : [batch_size, from_seq_length, num_attention_heads, attention_head_size]
	padded_query_layer = tf.pad(query_layer, 
							[[0, 0], 
							[int((kernel_size-1)/2) ,int((kernel_size-1)/2)],
							[0, 0], 
							[0, 0]])
	tf.logging.info(padded_query_layer)
	tf.logging.info("==padded_query_layer==")

	# [1, to_seq_length*kernel_size]
	padded_query_layer = tf.reshape(padded_query_layer, 
									[batch_size, -1, num_attention_heads * attention_head_size])
	
	tf.logging.info(padded_query_layer)
	tf.logging.info("==reshape padded_query_layer==")

	conv_span_output = bert_utils.gather_indexes(padded_query_layer, indices)
	
	tf.logging.info(conv_span_output)
	tf.logging.info("==conv_span_output==")

	conv_span_output = tf.reshape(conv_span_output, 
								[batch_size, 
								from_seq_length,
								kernel_size,
								num_attention_heads,
								attention_head_size
								])
	tf.logging.info(conv_span_output)
	conv_span_output = tf.transpose(conv_span_output, [0,3,1,2,4])
	tf.logging.info("==reshape conv_span_output==")

	# dynamic_conv_kernel: [batch_size, num_attention_heads, from_seq_length, kernel_size]
	# conv_span_output:    [batch_size, num_attention_heads, from_seq_length, kernel_size, attention_head_size]
	conv_output = tf.einsum("abcd,abcde->abce", normalized_dynamic_kernel, conv_span_output)
	tf.logging.info(conv_output)
	tf.logging.info("==conv_output==")

	# [batch_size, num_attention_heads, from_seq_length, attention_head_size]
	# [batch_size, from_seq_length, num_attention_heads, attention_head_size]
	conv_output = tf.transpose(conv_output, [0, 2, 1, 3])
	# conv_output *= tf.expand_dims(from_tensor_mask, axis=-1)
	conv_output_mask = tf.expand_dims(from_mask, axis=-1)
	conv_output_mask = tf.expand_dims(conv_output_mask, axis=-1)
	conv_output_mask = tf.cast(conv_output_mask, dtype=tf.float32)
	conv_output *= conv_output_mask
	if do_return_2d_tensor:
		# `context_layer` = [B*F, N*V]
		conv_output_layer = tf.reshape(
				conv_output,
				[batch_size * from_seq_length, num_attention_heads * attention_head_size])
	else:
		# `context_layer` = [B, F, N*V]
		conv_output_layer = tf.reshape(
				conv_output,
				[batch_size, from_seq_length, num_attention_heads * attention_head_size])

	return conv_output_layer