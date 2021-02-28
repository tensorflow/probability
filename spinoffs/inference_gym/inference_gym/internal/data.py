# Lint as: python3
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

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from inference_gym.internal.datasets import brownian_motion_missing_middle_observations as brownian_motion_lib
from inference_gym.internal.datasets import convection_lorenz_bridge as convection_lorenz_bridge_lib
from inference_gym.internal.datasets import sp500_closing_prices as sp500_closing_prices_lib
from inference_gym.internal.datasets import synthetic_item_response_theory as synthetic_item_response_theory_lib
from inference_gym.internal.datasets import synthetic_log_gaussian_cox_process as synthetic_log_gaussian_cox_process_lib

__all__ = [
    'brownian_motion_missing_middle_observations',
    'convection_lorenz_bridge',
    'german_credit_numeric',
    'radon',
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
  return tfp.util.DeferredTensor(
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


def brownian_motion_missing_middle_observations():
  """Synthetic dataset sampled from the BrownianMotion model.

  This dataset is a simulation of a 30 timestep Brownian Motion where
  the loc paramters of the middle ten timesteps are unobservable.

  Returns:
    dataset: A Dict with the following keys:
      observed_locs: Float `Tensor` of observed loc parameters at each timestep.
      observation_noise: Float `Tensor` of observation noise, must be positive.
      innovation_noise: Float `Tensor` of innovation noise, must be positive.

  """
  return dict(
      observed_locs=brownian_motion_lib.OBSERVED_LOC,
      observation_noise_scale=brownian_motion_lib.OBSERVATION_NOISE,
      innovation_noise_scale=brownian_motion_lib.INNOVATION_NOISE)


def convection_lorenz_bridge():
  """Synthetic dataset sampled from a LorenzSystem model.

  This dataset simulates a Lorenz system for 30 timesteps with a step size
  of 0.02. Only the convection (the first component of the state) is observed
  with Gaussian observation noise and the middle ten timesteps are unobserved.

  This model is based on the Lorenz Bridge system from [1].

  #### References

  1. Ambrogioni, Luca, Max Hinne, and Marcel van Gerven. "Automatic structured
     variational inference." arXiv preprint arXiv:2002.00643 (2020).

  Returns:
    dataset: A `dict` with the following entries:
      values: Float `Tensor` of observed convection values at each timestep.
      observation_index: The index for the convection values in the underlying
        state.
      observation_mask: A 30-length Boolean `Tensor` that is `False` for the
        middle ten observations and `True` elsewhere.
      observation_scale : The `float` scale of the observation noise for the
        system.
      innovation_scale: The `float` scale of the innovation noise for the
        system.
      step_size: The `float` step size used to numerically integrate the system.
  """
  return dict(
      observed_values=convection_lorenz_bridge_lib.OBSERVED_VALUES,
      observation_index=convection_lorenz_bridge_lib.OBSERVATION_INDEX,
      observation_mask=convection_lorenz_bridge_lib.OBSERVATION_MASK,
      observation_scale=convection_lorenz_bridge_lib.OBSERVATION_SCALE,
      innovation_scale=convection_lorenz_bridge_lib.INNOVATION_SCALE,
      step_size=convection_lorenz_bridge_lib.STEP_SIZE)


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


def radon(
    state='MN',
    num_examples=919,
    num_counties=85,
    train_fraction=1.,
    shuffle=False,
    shuffle_seed=42):
  """The Radon dataset [1] loaded from `tensorflow_datasets`.

  Radon is a radioactive gas that enters homes through contact points with the
  ground. It is a carcinogen that is the primary cause of lung cancer in
  non-smokers. Radon levels vary greatly from household to household.

  This dataset contains measured radon concentrations from houses in the
  United States and associated features.

  Args:
    state: `str`, the two-character code for the U.S. state by which to filter
      the data. If `None`, data for all states are returned.
    num_examples: `int`, total number of examples in the filtered dataset. When
      the dataset is materialized, this value is verified against the size of
      the filtered dataset, and if there is a mismatch a ValueError is raised.
      (The value is not determined automatically from the dataset because it is
      needed before the dataset is materialized).
    num_counties: `int`, number of unique counties in the filtered dataset.
    train_fraction: What fraction of the data to put in the training set. When
      the dataset is materialized, this value is verified against the size of
      the filtered dataset, and if there is a mismatch a ValueError is raised.
      (The value is not determined automatically from the dataset because it is
      needed before the dataset is materialized).
    shuffle: `bool`. If `True`, shuffle the data.
    shuffle_seed: `int`, RNG seed to use if shuffling the data.

  Returns:
    dataset: A Dict with the following keys:
      `num_counties`: `int`, number of unique counties in the filtered dataset.
      `train_log_uranium`: Floating-point `Tensor` with shape
        `[num_train]`. Soil uranium measurements.
      `train_floor`: Integer `Tensor` with shape `[num_train]`. Floor of the
        house on which the measurement was taken.
      `train_county`: Integer `Tensor` with values in `range(0, num_counties)`
        of shape `[num_train]`. County in which the measurement was taken.
      `train_floor_by_county`: Floating-point `Tensor` with shape
        `[num_train]`. Average floor on which the measurement was taken for the
        county in which each house is located (the `Tensor` will have
        `num_counties` unique values). This represents the contextual effect.
      `train_log_radon`: Floating-point `Tensor` with shape `[num_train]`.
        Radon measurement for each house (the dependent variable in the model).
      `test_log_uranium`: Floating-point `Tensor` with shape `[num_test`. Soil
        Soil uranium measurements for the test set.
      `test_floor`: Integer `Tensor` with shape `[num_test]`. Floor of the house
        house on which the measurement was taken.
      `test_county`: Integer `Tensor` with values in `range(0, num_counties)` of
        shape `[num_test]`. County in which the measurement was taken. Can be
        `None`, in which case test-related sample transformations are not
        computed.
      `test_floor_by_county`: Floating-point `Tensor` with shape
        `[num_test]`. Average floor on which the measurement was taken
        (calculated from the training set) for the county in which each house is
        located (the `Tensor` will have `num_counties` unique values). This
        represents the contextual effect.
      `test_log_radon`: Floating-point `Tensor` with shape `[num_test]`. Radon
        measurement for each house (the dependent variable in the model).

  Raises:
    ValueError if `num_examples` is not equal to the number of examples in the
      materialized dataset.
    ValueError if `num_counties` is not equal to the number of unique counties
      in the materialized dataset.

  #### References

  [1] Gelman, A., & Hill, J. (2007). Data Analysis Using Regression and
      Multilevel/Hierarchical Models (1st ed.). Cambridge University Press.
      http://www.stat.columbia.edu/~gelman/arm/examples/radon/
  """
  num_train = int(num_examples * train_fraction)
  num_test = num_examples - num_train

  def load_dataset():
    """Function that actually loads the dataset."""
    if load_dataset.dataset is not None:
      return load_dataset.dataset

    with tf.name_scope('radon'), tf.init_scope():

      dataset = _tfds().load(name='radon:1.*.*', split='train', batch_size=-1)
      dataset = _tfds().as_numpy(dataset)

      states = dataset['features']['state'].astype('U13')
      floor = dataset['features']['floor'].astype(np.int32)
      radon_val = dataset['activity'].astype(np.float64)
      county_strings = dataset['features']['county'].astype('U13')
      uranium = dataset['features']['Uppm'].astype(np.float64)

      if state is not None:
        floor = floor[states == state]
        radon_val = radon_val[states == state]
        county_strings = county_strings[states == state]
        uranium = uranium[states == state]

      radon_val[radon_val <= 0.] = 0.1
      log_radon = np.log(radon_val)
      log_uranium = np.log(uranium)
      unique_counties, county = np.unique(county_strings, return_inverse=True)
      county = county.astype(np.int32)

      if log_radon.size != num_examples:
        raise ValueError(
            'The size of the filtered dataset must equal the input '
            '`num_examples`. Saw dataset size = {}, `num_examples` = {}'
            ''.format(log_radon.size, num_examples))
      if unique_counties.size != num_counties:
        raise ValueError(
            'The number of counties present in the filtered dataset must equal '
            'the input `num_counties`. Saw {} counties but `num_counties` = {}'
            ''.format(unique_counties.size, num_counties))

      if shuffle:
        shuffle_idxs = np.arange(num_examples)
        np.random.RandomState(shuffle_seed).shuffle(shuffle_idxs)
        log_uranium = log_uranium[shuffle_idxs]
        floor = floor[shuffle_idxs]
        county = county[shuffle_idxs]
        log_radon = log_radon[shuffle_idxs]

      train_floor = floor[:num_train]
      train_county = county[:num_train]
      test_floor = floor[num_train:]
      test_county = county[num_train:]

      # Create a new features for mean of floor across counties.
      xbar = []
      for i in range(num_counties):
        xbar.append(train_floor[county == i].mean())
      floor_by_county = np.array(xbar, dtype=log_radon.dtype)

      load_dataset.dataset = dict(
          train_log_uranium=log_uranium[:num_train],
          train_floor=train_floor,
          train_county=train_county,
          train_floor_by_county=floor_by_county[train_county],
          train_log_radon=log_radon[:num_train],
          test_log_uranium=log_uranium[num_train:],
          test_floor=test_floor,
          test_county=test_county,
          test_floor_by_county=floor_by_county[test_county],
          test_log_radon=log_radon[num_train:],
      )

    return load_dataset.dataset

  load_dataset.dataset = None

  return dict(
      num_counties=np.array(num_counties, dtype=np.int32),
      train_log_uranium=_defer(
          lambda: load_dataset()['train_log_uranium'],
          shape=[num_train],
          dtype=np.float64),
      train_floor=_defer(
          lambda: load_dataset()['train_floor'],
          shape=[num_train],
          dtype=np.int32),
      train_county=_defer(
          lambda: load_dataset()['train_county'],
          shape=[num_train],
          dtype=np.int32),
      train_floor_by_county=_defer(
          lambda: load_dataset()['train_floor_by_county'],
          shape=[num_train],
          dtype=np.float64),
      train_log_radon=_defer(
          lambda: load_dataset()['train_log_radon'],
          shape=[num_train],
          dtype=np.float64),
      test_log_uranium=_defer(
          lambda: load_dataset()['test_log_uranium'],
          shape=[num_test],
          dtype=np.float64),
      test_floor=_defer(
          lambda: load_dataset()['test_floor'],
          shape=[num_test],
          dtype=np.int32),
      test_county=_defer(
          lambda: load_dataset()['test_county'],
          shape=[num_test],
          dtype=np.int32),
      test_floor_by_county=_defer(
          lambda: load_dataset()['test_floor_by_county'],
          shape=[num_test],
          dtype=np.float64),
      test_log_radon=_defer(
          lambda: load_dataset()['test_log_radon'],
          shape=[num_test],
          dtype=np.float64),
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
