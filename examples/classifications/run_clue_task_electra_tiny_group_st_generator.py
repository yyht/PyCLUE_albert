import os
import sys
sys.path.append("../..")
from PyCLUE.tasks.run_classifier import my_clue_tasks, configs
import tensorflow as tf

# assign GPU devices or CPU devices
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

flags = tf.flags

FLAGS = flags.FLAGS

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.logging.set_verbosity(tf.logging.ERROR)

flags.DEFINE_string("task_name", "", "oss buckets")
flags.DEFINE_string("gpu_id", "4", "oss buckets")

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id

# default configs: see PyCLUE.utils.classifier_utils.core
# below are some necessary paramters required in running this task

# task_name:
#     Support: 
#         chineseGLUE: bq, xnli, lcqmc, inews, thucnews, 
#         CLUE: afqmc, cmnli, copa, csl, iflytek, tnews, wsc
for task_name in FLAGS.task_name.split(","):
	from PyCLUE.tasks.run_classifier import configs
	if task_name == 'afqmc':
		configs["task_name"] = task_name
		# train parameters
		configs["max_seq_length"] = 128
		configs["train_batch_size"] = 32
		configs["learning_rate"] = 1e-4
		configs["warmup_proportion"] = 0.1
		configs["num_train_epochs"] = 10.0
	elif task_name == 'cmnli':
		configs["task_name"] = task_name
		# train parameters
		configs["max_seq_length"] = 128
		configs["train_batch_size"] = 32
		configs["learning_rate"] = 1e-4
		configs["warmup_proportion"] = 0.1
		configs["num_train_epochs"] = 10.0
	elif task_name == 'csl':
		configs["task_name"] = task_name
		# train parameters
		configs["max_seq_length"] = 256
		configs["train_batch_size"] = 32
		configs["learning_rate"] = 1e-4
		configs["warmup_proportion"] = 0.1
		configs["num_train_epochs"] = 20.0
	elif task_name == 'iflytek':
		configs["task_name"] = task_name
		# train parameters
		configs["max_seq_length"] = 512
		configs["train_batch_size"] = 32
		configs["learning_rate"] = 2e-4
		configs["warmup_proportion"] = 0.1
		configs["num_train_epochs"] = 20.0
	elif task_name == 'tnews':
		configs["task_name"] = task_name
		# train parameters
		configs["max_seq_length"] = 512
		configs["train_batch_size"] = 32
		configs["learning_rate"] = 1e-4
		configs["warmup_proportion"] = 0.1
		configs["num_train_epochs"] = 20.0
	elif task_name == 'wsc':
		configs["task_name"] = task_name
		# train parameters
		configs["max_seq_length"] = 128
		configs["train_batch_size"] = 32
		configs["learning_rate"] = 1e-4
		configs["warmup_proportion"] = 0.1
		configs["num_train_epochs"] = 10.0

	# pretrained_lm_name: 
	#     If None, should assign `vocab_file`, `bert_config_file`, `init_checkpoint`.
	#     Or you can choose the following models:
	#         bert, bert_wwm_ext, albert_xlarge, albert_large, albert_base, albert_base_ext, 
	#         albert_small, albert_tiny, roberta, roberta_wwm_ext, roberta_wwm_ext_large
	configs["pretrained_lm_name"] = "bert_electra_tiny_group_st_generator_1074200_mixed_mask"
	configs["vocab_file"] = "/data/group/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding_mixed_mask/generator/vocab.txt"
	configs["bert_config_file"] = "/data/group/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding_mixed_mask/generator/bert_config_tiny_large_embed.json"
	configs["init_checkpoint"] = "/data/group/st/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_pretrained_embedding_mixed_mask/generator/generator.ckpt-1074200"
	configs["verbose"] = 1

	configs["do_train"] = True
	configs["do_eval"] = True
	configs["do_predict"] = True

	my_clue_tasks(configs)
    

