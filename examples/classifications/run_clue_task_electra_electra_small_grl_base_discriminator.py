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
import os, sys

#vocab_path = "electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_scale_50_grl_auto_temp_fix_scratch"
#model_path = "electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_scale_50_grl_auto_temp_fix_scratch"
#config_path = "electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_scale_50_grl_auto_temp_fix_scratch"
config_path = "electra_bert_small_gen_bert_base_dis_joint_gumbel_no_sharing_base_grl_scale_50_2e-4"
vocab_path = "electra_bert_small_gen_bert_base_dis_joint_gumbel_no_sharing_base_grl_scale_50_2e-4"
model_path = "electra_bert_small_gen_bert_base_dis_joint_gumbel_no_sharing_base_grl_scale_50_2e-4"
model_name = "bert_electra_small_generator_scratch"

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
		configs["learning_rate"] = 3e-5
		configs["warmup_proportion"] = 0.1
		configs["num_train_epochs"] = 5.0
	elif task_name == 'cmnli':
		configs["task_name"] = task_name
		# train parameters
		configs["max_seq_length"] = 128
		configs["train_batch_size"] = 32
		configs["learning_rate"] = 3e-5
		configs["warmup_proportion"] = 0.1
		configs["num_train_epochs"] = 5.0
	elif task_name == 'csl':
		configs["task_name"] = task_name
		# train parameters
		configs["max_seq_length"] = 128
		configs["train_batch_size"] = 32
		configs["learning_rate"] = 3e-5
		configs["warmup_proportion"] = 0.1
		configs["num_train_epochs"] = 5.0
	elif task_name == 'iflytek':
		configs["task_name"] = task_name
		# train parameters
		configs["max_seq_length"] = 128
		configs["train_batch_size"] = 32
		configs["learning_rate"] = 3e-5
		configs["warmup_proportion"] = 0.1
		configs["num_train_epochs"] = 5.0
	elif task_name == 'tnews':
		configs["task_name"] = task_name
		# train parameters
		configs["max_seq_length"] = 128
		configs["train_batch_size"] = 32
		configs["learning_rate"] = 3e-5
		configs["warmup_proportion"] = 0.1
		configs["num_train_epochs"] = 5.0
	elif task_name == 'wsc':
		configs["task_name"] = task_name
		# train parameters
		configs["max_seq_length"] = 128
		configs["train_batch_size"] = 8
		configs["learning_rate"] = 3e-5
		configs["warmup_proportion"] = 0.1
		configs["num_train_epochs"] = 10.0
	elif task_name == 'lcqmc':
		configs["task_name"] = task_name
                # train parameters
		configs["max_seq_length"] = 128
		configs["train_batch_size"] = 32
		configs["learning_rate"] = 3e-5
		configs["warmup_proportion"] = 0.1
		configs["num_train_epochs"] = 5.0
	elif task_name == 'bq':
		configs["task_name"] = task_name
                # train paramet
		configs["max_seq_length"] = 128
		configs["train_batch_size"] = 32
		configs["learning_rate"] = 3e-5
		configs["warmup_proportion"] = 0.1
		configs["num_train_epochs"] = 5.0
	elif task_name == 'xnli':
		configs["task_name"] = task_name
                # train parameters
		configs["max_seq_length"] = 128
		configs["train_batch_size"] = 32
		configs["learning_rate"] = 3e-5
		configs["warmup_proportion"] = 0.1
		configs["num_train_epochs"] = 5.0
	elif task_name == 'inews':
		configs["task_name"] = task_name
                # train parameters
		configs["max_seq_length"] = 128
		configs["train_batch_size"] = 32
		configs["learning_rate"] = 3e-5
		configs["warmup_proportion"] = 0.1
		configs["num_train_epochs"] = 5.0
	elif task_name == 'thucnews':
		configs["task_name"] = task_name
                # train parameters
		configs["max_seq_length"] = 128
		configs["train_batch_size"] = 32
		configs["learning_rate"] = 3e-5
		configs["warmup_proportion"] = 0.1
		configs["num_train_epochs"] = 5.0
	# pretrained_lm_name: 
	#     If None, should assign `vocab_file`, `bert_config_file`, `init_checkpoint`.
	#     Or you can choose the following models:
	#         bert, bert_wwm_ext, albert_xlarge, albert_large, albert_base, albert_base_ext, 
	#         albert_small, albert_tiny, roberta, roberta_wwm_ext, roberta_wwm_ext_large
	#configs["pretrained_lm_name"] = "bert_tiny_electra_grl_scale_10_none_sharing_generator"
	#configs["vocab_file"] = "/data/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_scale_10_grl_auto_temp_fix/generator/vocab.txt"
	#configs["bert_config_file"] = "/data/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_scale_10_grl_auto_temp_fix/generator/bert_config_tiny.json"
	#configs["init_checkpoint"] = "/data/electra_bert_tiny_gen_bert_tiny_dis_joint_gumbel_no_sharing_scale_10_grl_auto_temp_fix/generator/generator"
	configs["verbose"] = 1
	configs["pretrained_lm_name"] = model_name+"_base_discriminator_grl_50"
	configs["vocab_file"] = os.path.join("/data", vocab_path, "discriminator", "vocab.txt")
	configs["bert_config_file"] = os.path.join("/data", config_path, "discriminator", "bert_config.json")
	configs["init_checkpoint"] = os.path.join("/data", config_path, "discriminator", "discriminator.ckpt-1289000")
	configs["do_train"] = True
	configs["do_eval"] = True
	configs["do_predict"] = True

	my_clue_tasks(configs)
    
