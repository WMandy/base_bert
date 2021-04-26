#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import tensorflow as tf
from tensorflow.python.platform import gfile
import modeling
from tensorflow.python.framework import graph_util


if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)

if __name__ == '__main__':
    ckpt_dir = "./tmp/output2/"
    bert_config_file="./tmp/output2/bert_config.json"
    output_dir = ckpt_dir
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    is_training=True
    sequence_length = 32
    use_one_hot_embeddings = True
    num_labels = 29

    with tf.Session() as sess:
        input_ids = tf.placeholder(tf.int32, [1, sequence_length], name="input_x")
        input_mask = tf.placeholder(tf.int32, [1, sequence_length], name="input_mask")
        segment_ids = tf.placeholder(tf.int32, [1, sequence_length], name="segment_ids")
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)
        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        #
        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())
        #
        with tf.variable_scope("loss"):
        
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)

        print(probabilities.name)



        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["loss/Softmax"])
        with tf.gfile.FastGFile(output_dir + 'bert_model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())

