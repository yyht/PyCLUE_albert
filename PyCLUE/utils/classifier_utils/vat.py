
import tensorflow as tf
import numpy as np
from utils.bert import bert_utils

def kl_divergence_with_logit(q_logit, p_logit):
	# [batch_size, seq_length, classes]
	q_logit = tf.nn.log_softmax(q_logit, axis=-1)
	p_logit = tf.nn.log_softmax(p_logit, axis=-1)

	# [batch_size, seq_length]
	qlogq = tf.reduce_sum(tf.exp(q_logit) * q_logit, -1)
	qlogp = tf.reduce_sum(tf.exp(q_logit) * p_logit, -1)
	return qlogq - qlogp