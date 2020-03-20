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

# Avoid rewriting these two for the Jax/Numpy backends.
# For TF in particular, as TFDS is TF-only, we always use the real TF when
# interacting with it.
import numpy as onp
import tensorflow.compat.v2 as otf

__all__ = [
    'german_credit_numeric',
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
  train = onp.asarray(train)
  test = onp.asarray(test)
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
  with otf.name_scope('german_credit_numeric'):
    dataset = _tfds().load(name='german_credit_numeric:1.*.*')
    features = []
    labels = []
    for entry in _tfds().as_numpy(dataset)['train']:
      features.append(entry['features'])
      # We're reversing the labels to match what's in the original dataset,
      # rather the TFDS encoding.
      labels.append(1 - entry['label'])
    features = onp.stack(features, axis=0)
    labels = onp.stack(labels, axis=0)

    num_train = int(features.shape[0] * train_fraction)

    train_features = features[:num_train]
    test_features = features[num_train:]

    if normalize_fn is not None:
      train_features, test_features = normalize_fn(train_features,
                                                   test_features)

    return dict(
        train_features=train_features,
        train_labels=labels[:num_train].astype(onp.int32),
        test_features=test_features,
        test_labels=labels[num_train:].astype(onp.int32),
    )
