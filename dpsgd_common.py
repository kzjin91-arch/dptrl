# Copyright 2020, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Common tools for DP-SGD MNIST tutorials."""

# These are not necessary in a Python 3-only module.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

import os
import scipy.io as scio
import numpy as np


def get_lr_model(features,n_classes,dim=4032):
  """Given input features, returns the logits from a simple lr model."""
  # old cnn structure
  # input_layer = tf.reshape(features, [-1, 28, 28, 1])
  # y = tf.keras.layers.Conv2D(
  #     16, 8, strides=2, padding='same', activation='relu').apply(input_layer)
  # y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
  # y = tf.keras.layers.Conv2D(
  #     32, 4, strides=2, padding='valid', activation='relu').apply(y)
  # y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
  # y = tf.keras.layers.Flatten().apply(y)
  # y = tf.keras.layers.Dense(32, activation='relu').apply(y)
  # logits = tf.keras.layers.Dense(10).apply(y)
  input_layer = tf.reshape(features['x'], tuple([-1]) + tuple([dim]))
  y = tf.keras.layers.Dense(1024,kernel_regularizer=tf.keras.regularizers.l2(
    0),activation='relu').apply(input_layer)
  y = tf.keras.layers.Dropout(0.8).apply(y)
  y = tf.keras.layers.Dense(512,kernel_regularizer=tf.keras.regularizers.l2(
    0),activation='relu').apply(y)
  y = tf.keras.layers.Dropout(0.8).apply(y)
  logits = tf.keras.layers.Dense(n_classes,kernel_regularizer=tf.keras.regularizers.l2(
    0)).apply(y)

  return logits


def make_input_fn(split, input_batch_size=256, repetitions=-1, tpu=False):
  """Make input function on given MNIST split."""

  def input_fn(params=None):
    """A simple input function."""
    batch_size = params.get('batch_size', input_batch_size)

    def parser(example):
      image, label = example['image'], example['label']
      image = tf.cast(image, tf.float32)
      image /= 255.0
      label = tf.cast(label, tf.int32)
      return image, label

    dataset = tfds.load(name='mnist', split=split)
    dataset = dataset.map(parser).shuffle(60000).repeat(repetitions).batch(
        batch_size)
    print("datasset:{}".format(dataset))
    os.system("pause")
    # If this input function is not meant for TPUs, we can stop here.
    # Otherwise, we need to explicitly set its shape. Note that for unknown
    # reasons, returning the latter format causes performance regression
    # on non-TPUs.
    if not tpu:
      return dataset

    # Give inputs statically known shapes; needed for TPUs.
    images, labels = tf.data.make_one_shot_iterator(dataset).get_next()
    # return images, labels
    images.set_shape([batch_size, 28, 28, 1])
    labels.set_shape([
        batch_size,
    ])
    return images, labels

  return input_fn

def my_input_fn(isTrain = True):
  data_l2_norm = float('inf')
  a, b = 'amazon', 'dslr'
  src_domain, tar_domain = scio.loadmat(a), scio.loadmat(b)
  train_data, train_labels = src_domain["fts"], src_domain["labels"].reshape(src_domain["labels"].shape[1])
  test_data, test_labels = tar_domain["fts"], tar_domain["labels"].reshape(tar_domain["labels"].shape[1])
  train_labels = train_labels-1
  test_labels = test_labels-1
  idx = np.random.permutation(len(train_data))  # shuffle data once
  train_data = np.array(train_data[idx], dtype=np.float32)
  test_data = np.array(test_data, dtype=np.float32)
  train_labels = train_labels[idx]

  normalize_data(train_data, data_l2_norm)
  normalize_data(test_data, data_l2_norm)

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)
  print("train_labels:{}{}{}".format(train_labels.shape, train_labels[:10], type(train_labels[0])))
  print("train{}{}{}".format(train_data.shape, train_data[:10], type(train_data[0, 0])))
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': train_data},
    y=train_labels,
    batch_size=256,
    num_epochs=30,
    shuffle=False)
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': test_data}, y=test_labels, num_epochs=1, shuffle=False)
  if isTrain:
    return train_input_fn
  else:
    return eval_input_fn

def model_fn(features,n_classes,dim=4032):
  input_layer = tf.reshape(features['x'], tuple([-1]) + tuple([dim]))

  logits = tf.layers.dense(
    inputs=input_layer,
    units=n_classes,
    kernel_regularizer=tf.keras.regularizers.l2(
      0),
    bias_regularizer=tf.keras.regularizers.l2(
      0))
  return logits

def normalize_data(data, data_l2_norm):
  """Normalizes data such that each samples has bounded L2 norm.

  Args:
    data: the dataset. Each row represents one samples.
    data_l2_norm: the target upper bound on the L2 norm.
  """

  for i in range(data.shape[0]):
    norm = np.linalg.norm(data[i])
    if norm > data_l2_norm:
      data[i] = data[i] / norm * data_l2_norm

