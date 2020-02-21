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
"""BERT finetuning task dataset generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json

from absl import app
from absl import flags
import tensorflow as tf

from official.nlp.bert import classifier_data_lib
from official.nlp.bert import tokenization

FLAGS = flags.FLAGS

# BERT classification specific flags.
flags.DEFINE_string(
    "input_data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

# Shared flags across BERT fine-tuning tasks.
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "predict_data_output_path", None,
    "The path in which generated predict input data will be written as tf"
    " records.")

flags.DEFINE_string("meta_data_file_path", None,
                    "The path in which input meta data will be written.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

def generate_classifier_dataset():
  """Generates classifier dataset and returns input meta data."""
  assert FLAGS.input_data_dir

  tokenizer = tokenization.FullTokenizer(
          vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  processor_text_fn = tokenization.convert_to_unicode
  processor = classifier_data_lib.BdbkProcessor(processor_text_fn)
  return classifier_data_lib.generate_predict_tf_record_from_data_file(
      processor,
      FLAGS.input_data_dir,
      tokenizer,
      predict_data_output_path=FLAGS.predict_data_output_path,
      max_seq_length=FLAGS.max_seq_length)


def main(_):
  if not FLAGS.vocab_file:
    raise ValueError("FLAG vocab_file for word-piece tokenizer is not specified.")

  input_meta_data = generate_classifier_dataset()

  with tf.io.gfile.GFile(FLAGS.meta_data_file_path, "w") as writer:
    writer.write(json.dumps(input_meta_data, indent=4) + "\n")


if __name__ == "__main__":
  flags.mark_flag_as_required("predict_data_output_path")
  flags.mark_flag_as_required("meta_data_file_path")
  app.run(main)
