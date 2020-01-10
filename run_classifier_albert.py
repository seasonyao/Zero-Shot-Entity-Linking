# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Entity Linking training/eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling_albert
import albert
from albert import optimization
from albert import tokenization
import tensorflow as tf
from tensorflow.python import debug as tf_debug

# Create a LocalCLIDebugHook and use it as a monitor when calling fit().
#hooks = [tf_debug.LocalCLIDebugHook()]

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "albert_config_file", None,
    "The config json file corresponding to the pre-trained ALBERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the ALBERT model was trained on.")

flags.DEFINE_string("spm_model_file", None,
                    "The model file for sentence piece tokenization.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("cached_dir", None,
                    "Path to cached training and dev tfrecord file. "
                    "The file will be generated if not exist.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "num_cands", 64,
    "Number of candidates. ")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_string("eval_domain", 'val_coronation', "Eval domain.")

flags.DEFINE_string("output_eval_file", '/tmp/eval.txt', "Eval output file.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 8, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 3000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("keep_checkpoint_max", 5,
                     "How many checkpoints to keep.")

flags.DEFINE_integer("iterations_per_loop", 3000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("optimizer", "adamw", "Optimizer to use")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer(
    "num_train_examples", 50000,
    "Number of training examples.")


def file_based_input_fn_builder(input_file, num_cands, seq_length, is_training,
                                drop_remainder, task_name, use_tpu, bsz,
                                multiple=1):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  labeltype = tf.int64

  name_to_features = {
      "input_ids": tf.FixedLenFeature([num_cands, seq_length * multiple], tf.int64),
      "input_mask": tf.FixedLenFeature([num_cands, seq_length * multiple], tf.int64),
      "segment_ids": tf.FixedLenFeature([num_cands, seq_length * multiple], tf.int64),
      "mention_id": tf.FixedLenFeature([num_cands, seq_length * multiple], tf.int64),
      "label_id": tf.FixedLenFeature([], labeltype),
      #"is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    if use_tpu:
      batch_size = params["batch_size"]
    else:
      batch_size = bsz

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def add_addictive_margin(embeddingFeature,one_hot_label, label, scale=30.,margin=-0.35,name="add_addictive_margin"):
    '''
        Adds margin to the embedding feature at the ground truth label if the score is less than margin.
        Then scales up the whole embedding feature by scale s
        The returned added_embeddingFeature is the fed to softmax to form the AM softmax
    '''
    with tf.name_scope(name):
        batch_range = tf.reshape(tf.range(tf.shape(embeddingFeature)[0]),shape=(-1,1))
        indices_of_groundtruth = tf.concat([batch_range, tf.reshape(label, shape=(-1,1))], axis=1)
        groundtruth_score = tf.gather_nd(embeddingFeature,indices_of_groundtruth)

        m = tf.constant(margin,name='m')
        s = tf.constant(scale,name='s')
        
        added_margin = tf.cast(tf.greater(groundtruth_score,-m),dtype=tf.float32)*m
        added_margin = tf.reshape(added_margin,shape=(-1,1))
        added_embeddingFeature = tf.add(embeddingFeature,one_hot_label*added_margin)*s
    return added_embeddingFeature


def create_zeshel_model(albert_config, is_training, input_ids, input_mask,
     segment_ids, mention_ids, labels, use_one_hot_embeddings):
    
  """Creates a classification model."""

  num_labels = input_ids.shape[1].value
  seq_len = input_ids.shape[-1].value

  input_ids = tf.reshape(input_ids, [-1, seq_len])
  segment_ids = tf.reshape(segment_ids, [-1, seq_len])
  input_mask = tf.reshape(input_mask, [-1, seq_len])
  mention_ids = tf.reshape(mention_ids, [-1, seq_len])

  model = modeling_albert.AlbertModel(
      config=albert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      mention_ids=mention_ids,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [1, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [1], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    # without addictive_margin
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    logits = tf.reshape(logits, [-1, num_labels])

    probabilities = tf.nn.softmax(logits, axis=-1)

    per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)

    loss = tf.reduce_mean(per_example_loss)

    #with addictive_margin
    # l2norm_output_layer = tf.nn.l2_normalize(output_layer,axis=1,name="l2norm_output_layer")
    # l2norm_output_weights = tf.nn.l2_normalize(output_weights,axis=0,name="l2norm_output_weights")

    # logits = tf.matmul(l2norm_output_layer, l2norm_output_weights, transpose_b=True)
    # logits = tf.nn.bias_add(logits, output_bias)
    # logits = tf.reshape(logits, [-1, num_labels])
    
    # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    
    # addictive_embedding_feature = add_addictive_margin(logits, one_hot_labels, labels, \
    #                                                scale=10.,margin=-0.1,name="add_addictive_margin")
    
    # probabilities = tf.nn.softmax(logits, axis=-1)
    # log_probs = tf.nn.log_softmax(logits, axis=-1)

    # per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=addictive_embedding_feature, labels=labels)
    # loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(albert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, optimizer="adamw"):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_id"]
    mention_ids = features["mention_id"]
#     is_real_example = None
#     if "is_real_example" in features:
#       is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
#     else:
#       is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_zeshel_model(
        albert_config, is_training, input_ids, input_mask, segment_ids, mention_ids,
        label_ids, use_one_hot_embeddings)
    

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling_albert.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps,
          use_tpu, optimizer)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
      # output_spec = tf.estimator.EstimatorSpec(
      #     mode=mode,
      #     loss=total_loss,
      #     train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions)
        loss = tf.metrics.mean(values=per_example_loss)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      #eval_metrics = metric_fn(per_example_loss, label_ids, logits)
      eval_metrics = (metric_fn,
                     [per_example_loss, label_ids, logits])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
      # output_spec = tf.estimator.EstimatorSpec(
      #     mode=mode,
      #     loss=total_loss,
      #     eval_metric_ops=eval_metrics)
    else:
#       output_spec = tf.contrib.tpu.TPUEstimatorSpec(
#           mode=mode,
#           predictions={"probabilities": probabilities},
#           scaffold_fn=scaffold_fn)
#       output_spec = tf.estimator.EstimatorSpec(
#           mode=mode,
#           predictions={"probabilities": probabilities})
      predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities,
                       "predictions": predictions,
                       "labels": label_ids},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  albert_config = modeling_albert.AlbertConfig.from_json_file(FLAGS.albert_config_file)

  if FLAGS.max_seq_length > albert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, albert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))
  # run_config = tf.estimator.RunConfig(
  #     model_dir=FLAGS.output_dir,
  #     save_summary_steps=100,
  #     save_checkpoints_steps=1000)

  num_train_examples = FLAGS.num_train_examples
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    num_train_steps = int(
        num_train_examples / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    if 1:                                       #if hnm
      num_train_steps = num_train_steps + 18750
      num_warmup_steps=0


  model_fn = model_fn_builder(
      albert_config=albert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)
  # estimator = tf.estimator.Estimator(
  #     model_fn=model_fn,
  #     config=run_config,
  #     params={"batch_size": FLAGS.train_batch_size})

  if FLAGS.do_train:
    #normal training
    #train_file = os.path.join(FLAGS.data_dir, "train_albert_ms256.tfrecord")
    #hnm training
    train_file = os.path.join(FLAGS.data_dir, "train_albert_ms256_hnmdata_after18750.tfrecord")
    
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Train file= %s", train_file)
    tf.logging.info("  Num examples = %d", num_train_examples)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        num_cands=FLAGS.num_cands,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True,
        task_name = 'entity_link',
        use_tpu=FLAGS.use_tpu,
        bsz=FLAGS.train_batch_size)
        
#     estimator.train(input_fn=train_input_fn, max_steps=num_train_steps, hooks=hooks)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    eval_file = os.path.join(FLAGS.data_dir, FLAGS.eval_domain + ".tfrecord")

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      eval_steps = 0
      for fn in [eval_file]:
        for record in tf.python_io.tf_record_iterator(fn):
          eval_steps += 1

      eval_steps = int(eval_steps // FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        num_cands=FLAGS.num_cands,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder,
        task_name=FLAGS.task_name,
        use_tpu=FLAGS.use_tpu,
        bsz=FLAGS.train_batch_size)
        

    result = estimator.evaluate(
        input_fn=eval_input_fn,
        steps=eval_steps,
        name=FLAGS.eval_domain)

    output_eval_file = FLAGS.output_eval_file
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))
    tf.logging.info("***** Eval results *****")
    for key in sorted(result.keys()):
      tf.logging.info("  %s = %s", key, str(result[key]))

  if FLAGS.do_predict:
    #predict the val data
    predict_file = os.path.join(FLAGS.data_dir, FLAGS.eval_domain + ".tfrecord")
    #predict training data for hnm
    #predict_file = os.path.join(FLAGS.data_dir, "train_albert_ms256.tfrecord")
    tf.logging.info(predict_file)
    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        num_cands=FLAGS.num_cands,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder,
        task_name=FLAGS.task_name,
        use_tpu=FLAGS.use_tpu,
        bsz=FLAGS.train_batch_size)
        
    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = FLAGS.output_eval_file
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        predictions = prediction["predictions"]
        labels = prediction["labels"]
        is_right = 0
        if predictions==labels:
            is_right=1
        
        output_line = "\t".join(
            str(class_probability)
            for class_probability in probabilities) + "\t" + str(is_right) + "\n"
        writer.write(output_line)
        num_written_lines += 1


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  #flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("albert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()