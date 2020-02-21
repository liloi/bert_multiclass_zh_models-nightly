# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""BERT classification finetuning runner in TF 2.x."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from official.nlp import optimization
from official.nlp.bert import bert_models
from official.nlp.bert import common_flags
from official.nlp.bert import configs as bert_configs
from official.nlp.bert import input_pipeline
#from official.utils.misc import keras_utils


flags.DEFINE_string('predict_data_path', None,
                    'Path to testing data for BERT classifier.')
flags.DEFINE_string(
    'input_meta_data_path', None,
    'Path to file that contains meta data about input '
    'to be used for training and evaluation.')
flags.DEFINE_integer('predict_batch_size', 32, 'Batch size for prediction.')

common_flags.define_common_bert_flags()

FLAGS = flags.FLAGS

def get_dataset_fn(input_file_pattern, max_seq_length, global_batch_size,
                   is_training):
  """Gets a closure to create a dataset."""

  def _dataset_fn(ctx=None):
    """Returns tf.data.Dataset for distributed BERT pretraining."""
    batch_size = ctx.get_per_replica_batch_size(
        global_batch_size) if ctx else global_batch_size
    dataset = input_pipeline.create_classifier_dataset(
        input_file_pattern,
        max_seq_length,
        batch_size,
        is_training=is_training,
        input_pipeline_context=ctx)
    return dataset

  return _dataset_fn


def main(_):
  # Users should always run this script under TF 2.x
  assert tf.version.VERSION.startswith('2.')

  with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))

  def _get_classifier_model():
    """Gets a classifier model."""
    bert_config = bert_configs.BertConfig.from_json_file(FLAGS.bert_config_file)
    num_classes = input_meta_data['num_labels']
    classifier_model, core_model = (
        bert_models.classifier_model(
            bert_config,
            num_classes,
            input_meta_data['max_seq_length'],
            hub_module_url=FLAGS.hub_module_url,
            hub_module_trainable=False))
    classifier_model.optimizer = optimization.create_optimizer(
        initial_lr, steps_per_epoch * epochs, warmup_steps)
    if FLAGS.fp16_implementation == 'graph_rewrite':
      classifier_model.optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
          classifier_model.optimizer)
    return classifier_model, core_model

  predict_input_fn = get_dataset_fn(
      FLAGS.predict_data_path,
      input_meta_data['max_seq_length'],
      FLAGS.predict_batch_size,
      is_training=False)

  predict_dataset = predict_input_fn()
  bert_model, sub_model = _get_classifier_model()
  optimizer = bert_model.optimizer
  new_model = tf.keras.models.load_model(
      FLAGS.model_export_path,
      custom_objects={"KerasLayer": sub_model,
                      "AdamWeightDecay": bert_model.optimizer,
                      "classification_loss_fn": loss_fn})
  predict_ret = new_model.predict(predict_dataset)
  logging.info('liran05:{}'.format(pre_ret))


if __name__ == '__main__':
  flags.mark_flag_as_required('input_meta_data_path')
  app.run(main)
