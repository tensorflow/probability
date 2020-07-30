# Lint as: python2, python3
# Copyright 2020 The TensorFlow Probability Authors.
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
"""Datasets and dataset utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.inference_gym.internal.datasets import sp500_closing_prices as sp500_closing_prices_lib
from tensorflow_probability.python.experimental.inference_gym.internal.datasets import synthetic_item_response_theory as synthetic_item_response_theory_lib
from tensorflow_probability.python.experimental.inference_gym.internal.datasets import synthetic_log_gaussian_cox_process as synthetic_log_gaussian_cox_process_lib
from tensorflow_probability.python.util.deferred_tensor import DeferredTensor

__all__ = [
    'german_credit_numeric',
    'sp500_closing_prices',
    'synthetic_item_response_theory',
    'synthetic_log_gaussian_cox_process',
]


def _tfds():
  """Import the TFDS module lazily."""
  try:
    import tensorflow_datasets as tfds  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  except ImportError:
    # Print more informative error message, then reraise.
    print("\n\nFailed to import TensorFlow Datasets. This is needed by the "
          "Inference Gym targets conditioned on real data. "
          "TensorFlow Probability does not have it as a dependency by default, "
          "so you need to arrange for it to be installed yourself. If you're "
          "installing TensorFlow Probability pip package, this can be done by "
          "installing it as `pip install tensorflow_probability[tfds]` "
          "or `pip install tfp_nightly[tfds]`.\n\n")
    raise
  return tfds


def _defer(fn, shape, dtype):
  empty = np.zeros(0)
  return DeferredTensor(
      empty, lambda _: fn(), shape=shape, dtype=dtype)


def _normalize_zero_mean_one_std(train, test):
  """Normalizes the data columnwise to have mean of 0 and std of 1.

  Assumes that the first axis indexes independent datapoints. The mean and
  standard deviation are estimated from the training set and are used
  for both train and test data, to avoid leaking information.

  Args:
    train: A floating point numpy array representing the training data.
    test: A floating point numpy array representing the test data.

  Returns:
    normalized_train: The normalized training data.
    normalized_test: The normalized test data.
  """
  train = np.asarray(train)
  test = np.asarray(test)
  train_mean = train.mean(0, keepdims=True)
  train_std = train.std(0, keepdims=True)
  return (train - train_mean) / train_std, (test - train_mean) / train_std


def german_credit_numeric(
    train_fraction=1.,
    normalize_fn=_normalize_zero_mean_one_std,
):
  """The numeric German Credit dataset [1].

  This dataset contains 1000 data points with 24 features and 1 binary label. It
  can be optionally split into a training and testing sets. The train set will
  contain `num_train_points = int(train_fraction * 1000)` examples and test set
  will contain `num_test_points = 1000 - num_train_points` examples.

  Args:
    train_fraction: What fraction of the data to put in the training set.
    normalize_fn: A callable to normalize the data. This should take the train
      and test features and return the normalized versions of them.

  Returns:
    dataset: A Dict with the following keys:
      `train_features`: Floating-point `Tensor` with shape `[num_train_points,
        24]`. Training features.
      `train_labels`: Integer `Tensor` with shape `[num_train_points]`. Training
        labels.
      `test_features`: Floating-point `Tensor` with shape `[num_test_points,
        24]`. Testing features.
      `test_labels`: Integer `Tensor` with shape `[num_test_points]`. Testing
        labels.

  #### References

  1. https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
  """
  num_points = 1000
  num_train = int(num_points * train_fraction)
  num_test = num_points - num_train
  num_features = 24

  def load_dataset():
    """Function that actually loads the dataset."""
    if load_dataset.dataset is not None:
      return load_dataset.dataset

    with tf.name_scope('german_credit_numeric'), tf.init_scope():
      dataset = _tfds().load('german_credit_numeric:1.*.*')
      features = []
      labels = []
      for entry in _tfds().as_numpy(dataset)['train']:
        features.append(entry['features'])
        # We're reversing the labels to match what's in the original dataset,
        # rather the TFDS encoding.
        labels.append(1 - entry['label'])
      features = np.stack(features, axis=0)
      labels = np.stack(labels, axis=0)

      train_features = features[:num_train]
      test_features = features[num_train:]

      if normalize_fn is not None:
        train_features, test_features = normalize_fn(train_features,
                                                     test_features)

      load_dataset.dataset = dict(
          train_features=train_features,
          train_labels=labels[:num_train].astype(np.int32),
          test_features=test_features,
          test_labels=labels[num_train:].astype(np.int32),
      )

    return load_dataset.dataset

  load_dataset.dataset = None

  return dict(
      train_features=_defer(
          lambda: load_dataset()['train_features'],
          shape=[num_train, num_features],
          dtype=np.float64),
      train_labels=_defer(
          lambda: load_dataset()['train_labels'],
          shape=[num_train],
          dtype=np.int32),
      test_features=_defer(
          lambda: load_dataset()['test_features'],
          shape=[num_test, num_features],
          dtype=np.float64),
      test_labels=_defer(
          lambda: load_dataset()['test_labels'],
          shape=[num_test],
          dtype=np.int32),
  )


def sp500_closing_prices(num_points=None):
  """Dataset of mean-adjusted returns of the S&P 500 index.

  Each of the 2516 entries represents the adjusted return of the daily closing
  price relative to the previous close, for each (non-holiday) weekday,
  beginning 6/26/2010 and ending 6/24/2020.

  Args:
    num_points: Optional `int` length of the series to return. If specified,
      only the final `num_points` returns are centered and returned.
      Default value: `None`.
  Returns:
    dataset: A Dict with the following keys:
      `centered_returns`: float `Tensor` daily returns, minus the mean return.
  """
  returns = np.diff(sp500_closing_prices_lib.CLOSING_PRICES)
  num_points = num_points or len(returns)
  return dict(
      centered_returns=returns[-num_points:] - np.mean(returns[-num_points:]))


def synthetic_item_response_theory(
    train_fraction=1.,
    shuffle=True,
    shuffle_seed=1337,
):
  """Synthetic dataset sampled from the ItemResponseTheory model.

  This dataset is a simulation of 400 students each answering a subset of
  100 unique questions, with a total of 30012 questions answered.

  The dataset is split into train and test portions by randomly partitioning the
  student-question-response triplets. This has two consequences. First, the
  student and question ids are shared between test and train sets. Second, there
  is a possibility of some students or questions not being present in both sets.

  The train set will contain `num_train_points = int(train_fraction * 30012)`
  triples and test set will contain
  `num_test_points = 30012 - num_train_points` triplets.

  The triples are encoded into three parallel arrays per set. I.e.
  `*_correct[i]  == 1` means that student `*_student_ids[i]` answered question
  `*_question_ids[i]` correctly; `*_correct[i] == 0` means they didn't.

  Args:
    train_fraction: What fraction of the data to put in the training set.
    shuffle: Whether to shuffle the dataset.
    shuffle_seed: Seed to use when shuffling.

  Returns:
    dataset: A Dict with the following keys:
      `train_student_ids`: Integer `Tensor` with shape `[num_train_points]`.
        training student ids, ranging from 0 to `num_students`.
      `train_question_ids`: Integer `Tensor` with shape `[num_train_points]`.
        training question ids, ranging from 0 to `num_questions`.
      `train_correct`: Integer `Tensor` with shape `[num_train_points]`.
        Whether the student in the training set answered the question correctly,
        either 0 or 1.
      `test_student_ids`: Integer `Tensor` with shape `[num_test_points]`.
        Testing student ids, ranging from 0 to `num_students`.
      `test_question_ids`: Integer `Tensor` with shape `[num_test_points]`.
        Testing question ids, ranging from 0 to `num_questions`.
      `test_correct`: Integer `Tensor` with shape `[num_test_points]`.
        Whether the student in the testing set answered the question correctly,
        either 0 or 1.
  """
  student_ids = synthetic_item_response_theory_lib.STUDENT_IDS
  question_ids = synthetic_item_response_theory_lib.QUESTION_IDS
  correct = synthetic_item_response_theory_lib.CORRECT

  if shuffle:
    shuffle_idxs = np.arange(student_ids.shape[0])
    np.random.RandomState(shuffle_seed).shuffle(shuffle_idxs)
    student_ids = student_ids[shuffle_idxs]
    question_ids = question_ids[shuffle_idxs]
    correct = correct[shuffle_idxs]

  num_train = int(student_ids.shape[0] * train_fraction)

  return dict(
      train_student_ids=student_ids[:num_train],
      train_question_ids=question_ids[:num_train],
      train_correct=correct[:num_train],
      test_student_ids=student_ids[num_train:],
      test_question_ids=question_ids[num_train:],
      test_correct=correct[num_train:],
  )


def synthetic_log_gaussian_cox_process(
    shuffle=True,
    shuffle_seed=1337,
):
  """Synthetic dataset sampled from the LogGaussianCoxProcess model.

  This dataset was simulated by constructing a 10 by 10 grid of equidistant 2D
  locations with spacing = 1, and then sampling from the prior to determine the
  counts at those locations.

  The data are encoded into three parallel arrays. I.e.
  `train_counts[i]` and `train_extents[i]` correspond to `train_locations[i]`.

  Args:
    shuffle: Whether to shuffle the dataset.
    shuffle_seed: Seed to use when shuffling.

  Returns:
    dataset: A Dict with the following keys:
      train_locations: Float `Tensor` with shape `[num_train_points, 2]`.
        Training set locations where counts were measured.
      train_extents: Float `Tensor` with shape `[num_train_points]`. Training
        set location extents, must be positive.
      train_counts: Float `Tensor` with shape `[num_train_points]`. Training set
        counts, must be positive.
  """
  locations = synthetic_log_gaussian_cox_process_lib.LOCATIONS
  extents = synthetic_log_gaussian_cox_process_lib.EXTENTS
  counts = synthetic_log_gaussian_cox_process_lib.COUNTS

  if shuffle:
    shuffle_idxs = np.arange(locations.shape[0])
    np.random.RandomState(shuffle_seed).shuffle(shuffle_idxs)
    locations = locations[shuffle_idxs]
    counts = counts[shuffle_idxs]
    extents = extents[shuffle_idxs]

  return dict(
      train_locations=locations,
      train_extents=extents,
      train_counts=counts,
  )
