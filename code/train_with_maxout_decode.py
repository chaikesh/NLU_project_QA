"""
# python -2.7 
# Purely ours for understanding baseline code
# This is main file information
# Modified by: Chaikesh, Prakash and Sonu
"""

from __future__ import absolute_import
from __future__ import division

import os
import io
import json
import sys
import logging

# In[]
import tensorflow as tf
from data_batcher1 import get_batch_generator
from qa_model import QAModel
from vocab import get_glove
from official_eval_helper import get_json_data, generate_answers

# In[]
logging.basicConfig(level=logging.INFO)

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath("__file__")))) # relative path of the main directory
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") # relative path of data dir
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir


# High-level options
tf.app.flags.DEFINE_integer("gpu", 0, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_string("mode", "train", "Available modes: train / show_examples / official_eval")
tf.app.flags.DEFINE_string("experiment_name", "", "Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment")
tf.app.flags.DEFINE_integer("num_epochs", 0, "Number of epochs to train. 0 means train indefinitely")

# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use")
tf.app.flags.DEFINE_integer("hidden_size", 200, "Size of the hidden states")
tf.app.flags.DEFINE_integer("context_len", 600, "The maximum context length of your model")
tf.app.flags.DEFINE_integer("question_len", 30, "The maximum question length of your model")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained word vectors. This needs to be one of the available GloVe dimensions: 50/100/200/300")

# How often to print, save, eval
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("save_every", 500, "How many iterations to do per save.")
tf.app.flags.DEFINE_integer("eval_every", 500, "How many iterations to do per calculating loss/f1/em on dev set. Warning: this is fairly time-consuming so don't do it too often.")
tf.app.flags.DEFINE_integer("keep", 1, "How many checkpoints to keep. 0 indicates keep all (you shouldn't need to do keep all though - it's very storage intensive).")

# Reading and saving data
tf.app.flags.DEFINE_string("train_dir", "", "Training directory to save the model parameters and other info. Defaults to experiments/{experiment_name}")
tf.app.flags.DEFINE_string("glove_path", "", "Path to glove .txt file. Defaults to data/glove.6B.{embedding_size}d.txt")
tf.app.flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR, "Where to find preprocessed SQuAD data for training. Defaults to data/")
tf.app.flags.DEFINE_string("ckpt_load_dir", "", "For official_eval mode, which directory to load the checkpoint fron. You need to specify this for official_eval mode.")
tf.app.flags.DEFINE_string("json_in_path", "", "For official_eval mode, path to JSON input file. You need to specify this for official_eval_mode.")
tf.app.flags.DEFINE_string("json_out_path", "predictions.json", "Output path for official_eval mode. Defaults to predictions.json")

# In[]
FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)


FLAGS.train_dir = FLAGS.train_dir or os.path.join(EXPERIMENTS_DIR, FLAGS.experiment_name)

# Initialize bestmodel directory
bestmodel_dir = os.path.join(FLAGS.train_dir, "best_checkpoint")

# Define path for glove vecs
FLAGS.glove_path = FLAGS.glove_path or os.path.join(DEFAULT_DATA_DIR, "glove.6B.{}d.txt".format(FLAGS.embedding_size))

# Load embedding matrix and vocab mappings
emb_matrix, word2id, id2word = get_glove(FLAGS.glove_path, FLAGS.embedding_size)

# Get filepaths to train/dev datafiles for tokenized queries, contexts and answers
train_context_path = os.path.join(FLAGS.data_dir, "train.context")
train_qn_path = os.path.join(FLAGS.data_dir, "train.question")
train_ans_path = os.path.join(FLAGS.data_dir, "train.span")
dev_context_path = os.path.join(FLAGS.data_dir, "dev.context")
dev_qn_path = os.path.join(FLAGS.data_dir, "dev.question")
dev_ans_path = os.path.join(FLAGS.data_dir, "dev.span")

# In[]

# Initialize model
#qa_model = QAModel(FLAGS, id2word, word2id, emb_matrix)
from modules import RNNEncoder, SimpleSoftmaxLayer, BasicAttn, maybe_mask_affinity, RNNEncoder1
from modules import RNNEncoder2, dcn_decode, maybe_dropout
# In[]

context_ids = tf.placeholder(tf.int32, shape=[None, FLAGS.context_len])
context_mask = tf.placeholder(tf.int32, shape=[None, FLAGS.context_len])
qn_ids = tf.placeholder(tf.int32, shape=[None, FLAGS.question_len])
qn_mask = tf.placeholder(tf.int32, shape=[None, FLAGS.question_len])
ans_span = tf.placeholder(tf.int32, shape=[None, 2])

keep_prob = tf.placeholder_with_default(1.0, shape=())
# In[]
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops


# In[]
with vs.variable_scope("embeddings"):
    # Note: the embedding matrix is a tf.constant which means it's not a trainable parameter
    embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix") # shape (400002, embedding_size)
    # Get the word embeddings for the context and question,
    # using the placeholders self.context_ids and self.qn_ids
    context_embs = embedding_ops.embedding_lookup(embedding_matrix, context_ids) # shape (batch_size, context_len, embedding_size)
    qn_embs = embedding_ops.embedding_lookup(embedding_matrix, qn_ids) # shape (batch_size, question_len, embedding_size)


encoder = RNNEncoder(FLAGS.hidden_size, keep_prob)
context_hiddens = encoder.build_graph(context_embs, context_mask) # (batch_size, context_len, hidden_size*2)
question_hiddens = encoder.build_graph(qn_embs, qn_mask) # (batch_size, question_len, hidden_size*2)

question_variation = tf.layers.dense(question_hiddens, question_hiddens.get_shape()[2], activation=tf.tanh);
        

# In[]

question_length = tf.placeholder(tf.int32, (None,), name='question_length')
document_length = tf.placeholder(tf.int32, (None,), name='paragraph_length')
#question_length = tf.reduce_sum(qn_mask, reduction_indices=1) # shape (batch_size)
#document_length = tf.reduce_sum(context_mask, reduction_indices=1) # shape (batch_size)

unmasked_affinity = tf.einsum('ndh,nqh->ndq', context_hiddens, question_variation)  # [N, D, Q] or [N, 1+D, 1+Q] if sentinel
affinity = maybe_mask_affinity(unmasked_affinity, document_length)
attention_p = tf.nn.softmax(affinity, dim=1)
unmasked_affinity_t = tf.transpose(unmasked_affinity, [0, 2, 1])  # [N, Q, D] or [N, 1+Q, 1+D] if sentinel
affinity_t = maybe_mask_affinity(unmasked_affinity_t, question_length)
attention_q = tf.nn.softmax(affinity_t, dim=1)
summary_q = tf.einsum('ndh,ndq->nqh', context_hiddens, attention_p)  # [N, Q, 2H] or [N, 1+Q, 2H] if sentinel
summary_d = tf.einsum('nqh,nqd->ndh', question_variation, attention_q)  # [N, D, 2H] or [N, 1+D, 2H] if sentinel
coattention_d = tf.einsum('nqh,nqd->ndh', summary_q, attention_q)

encoder1 = RNNEncoder1(FLAGS.hidden_size, keep_prob)
context2 = encoder1.build_graph(summary_d, context_mask) # (batch_size, context_len, hidden_size*2)
question2 = encoder1.build_graph(summary_q, qn_mask) # (batch_size, question_len, hidden_size*2)


unmasked_affinity1 = tf.einsum('ndh,nqh->ndq', context2, question2)  # [N, D, Q] or [N, 1+D, 1+Q] if sentinel
affinity1 = maybe_mask_affinity(unmasked_affinity1, document_length)
attention_p1 = tf.nn.softmax(affinity1, dim=1)
unmasked_affinity_t1 = tf.transpose(unmasked_affinity1, [0, 2, 1])  # [N, Q, D] or [N, 1+Q, 1+D] if sentinel
affinity_t1 = maybe_mask_affinity(unmasked_affinity_t1, question_length)
attention_q1 = tf.nn.softmax(affinity_t1, dim=1)
summary_q1 = tf.einsum('ndh,ndq->nqh', context2, attention_p1)  # [N, Q, 2H] or [N, 1+Q, 2H] if sentinel
summary_d1 = tf.einsum('nqh,nqd->ndh', question2, attention_q1)  # [N, D, 2H] or [N, 1+D, 2H] if sentinel
coattention_d1 = tf.einsum('nqh,nqd->ndh', summary_q1, attention_q1)

# In[]
document_representations = [
    context_hiddens,  # E^D_1
    context2, # E^D_2
    summary_d,        # S^D_1
    summary_d1,        # S^D_2
    coattention_d,    # C^D_1
    coattention_d1,    # C^D_2
]

document_representation = tf.concat(document_representations, 2)
encoder2 = RNNEncoder2(FLAGS.hidden_size, keep_prob)
U = encoder2.build_graph(document_representation, context_mask)


# In[]

logits = dcn_decode(U, document_length, 100, 4, 4, keep_prob=maybe_dropout(keep_prob, True))






last_iter_logit = logits.read(4-1)
logits_start, logits_end = last_iter_logit[:,:,0], last_iter_logit[:,:,1]



# In[]

with vs.variable_scope("loss"):
    # Calculate loss for prediction of start position
    loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_start, labels=ans_span[:, 0]) # loss_start has shape (batch_size)
    loss_start = tf.reduce_mean(loss_start) # scalar. avg across batch
    tf.summary.scalar('loss_start', loss_start) # log to tensorboard

    # Calculate loss for prediction of end position
    loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_end, labels=ans_span[:, 1])
    loss_end = tf.reduce_mean(loss_end)
    tf.summary.scalar('loss_end', loss_end)

    # Add the two losses
    loss = loss_start + loss_end
    tf.summary.scalar('loss', loss)


params = tf.trainable_variables()
gradients = tf.gradients(loss, params)
gradient_norm = tf.global_norm(gradients)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
param_norm = tf.global_norm(params)


global_step = tf.Variable(0, name="global_step", trainable=False)
opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) # you can try other optimizers
updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

# Define savers (for checkpointing) and summaries (for tensorboard)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
summaries = tf.summary.merge_all()

# In[]


for batch in get_batch_generator(word2id, train_context_path, train_qn_path, train_ans_path, FLAGS.batch_size, context_len=FLAGS.context_len, question_len=FLAGS.question_len, discard_long=True):
    #loss, global_step, param_norm, grad_norm = run_train_iter(session, batch, summary_writer)
    # Match up our input data with the placeholders
    input_feed = {}
    print('1')
    input_feed[context_ids] = batch.context_ids
    print(batch.context_ids.shape)
    input_feed[context_mask] = batch.context_mask
    input_feed[qn_ids] = batch.qn_ids
    input_feed[qn_mask] = batch.qn_mask
    input_feed[ans_span] = batch.ans_span
    input_feed[keep_prob] = 1.0 - FLAGS.dropout # apply dropout
    input_feed[question_length] = batch.qn_length
    input_feed[document_length] = batch.context_length
    # output_feed contains the things we want to fetch.
    #output_feed = [updates, summaries, loss, global_step, param_norm, gradient_norm]

    # Run the model
    #[_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)

    # All summaries in the graph are added to Tensorboard
    #summary_writer.add_summary(summaries, global_step)










