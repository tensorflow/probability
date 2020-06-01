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
"""1PL item-response theory model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as onp
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.experimental.inference_gym.internal import data
from tensorflow_probability.python.experimental.inference_gym.targets import bayesian_model
from tensorflow_probability.python.internal import prefer_static as ps

__all__ = [
    'ItemResponseTheory',
    'SyntheticItemResponseTheory',
]


class ItemResponseTheory(bayesian_model.BayesianModel):
  """One-parameter logistic item-response theory (IRT) model."""

  def __init__(
      self,
      train_student_ids,
      train_question_ids,
      train_correct,
      test_student_ids=None,
      test_question_ids=None,
      test_correct=None,
      name='item_response_theory',
      pretty_name='Item-Response Theory',
  ):
    """Construct the item-response theory model.

    This models a set of students answering a set of questions, and being scored
    whether they get the question correct or not. Each student is associated
    with a scalar `student_ability`, and each question is associated with a
    scalar `question_difficulty`. Additionally, a scalar `mean_student_ability`
    is shared between all the students. This corresponds to the [1PL
    item-response theory](1) model.

    The data are encoded into three parallel arrays per set. I.e.
    `*_correct[i]  == 1` means that student `*_student_ids[i]` answered question
    `*_question_ids[i]` correctly; `*_correct[i] == 0` means they didn't.

    Args:
      train_student_ids: integer `tensor` with shape `[num_train_points]`.
        training student ids, ranging from 0 to `num_students`.
      train_question_ids: integer `tensor` with shape `[num_train_points]`.
        training question ids, ranging from 0 to `num_questions`.
      train_correct: integer `tensor` with shape `[num_train_points]`.
        whether the student in the training set answered the question correctly,
        either 0 or 1.
      test_student_ids: Integer `Tensor` with shape `[num_test_points]`.
        Testing student ids, ranging from 0 to `num_students`. Can be `None`, in
        which case test-related sample transformations are not computed.
      test_question_ids: Integer `Tensor` with shape `[num_test_points]`.
        Testing question ids, ranging from 0 to `num_questions`. Can be `None`,
        in which case test-related sample transformations are not computed.
      test_correct: Integer `Tensor` with shape `[num_test_points]`.
        Whether the student in the testing set answered the question correctly,
        either 0 or 1. Can be `None`, in which case test-related sample
        transformations are not computed.
      name: Python `str` name prefixed to Ops created by this class.
      pretty_name: A Python `str`. The pretty name of this model.

    Raises:
      ValueError: If `test_student_ids`, `test_question_ids` or `test_correct`
        are not either all `None` or are all specified.
      ValueError: If the parallel arrays are not all of the same size.

    #### References

    1. https://en.wikipedia.org/wiki/Item_response_theory
    """
    with tf.name_scope(name):
      test_data_present = (
          e is not None
          for e in [test_student_ids, test_question_ids, test_correct])
      self._have_test = all(test_data_present)
      if not self._have_test and any(test_data_present):
        raise ValueError('`test_student_ids`, `test_question_ids` and '
                         '`test_correct` must either all be `None` or '
                         'all be specified. Got: test_student_ids={}, '
                         'test_question_ids={}, test_correct={}'.format(
                             test_student_ids, test_question_ids, test_correct))
      if not (train_student_ids.shape[0] == train_question_ids.shape[0] ==
              train_correct.shape[0]):
        raise ValueError('`train_student_ids`, `train_question_ids` and '
                         '`train_correct` must all have the same length. '
                         'Got: {} {} {}'.format(train_student_ids.shape[0],
                                                train_question_ids.shape[0],
                                                train_correct.shape[0]))

      max_student_id = train_student_ids.max()
      max_question_id = train_question_ids.max()
      if self._have_test:
        max_student_id = max(max_student_id, test_student_ids.max())
        max_question_id = max(max_question_id, test_question_ids.max())

      self._num_students = max_student_id + 1
      self._num_questions = max_question_id + 1

      # TODO(siege): Make it an option to use a sparse encoding. The dense
      # encoding is only efficient when the dataset is not very sparse to begin
      # with.
      train_dense_y, train_y_mask = self._sparse_to_dense(
          train_student_ids,
          train_question_ids,
          train_correct,
      )

      root = tfd.JointDistributionCoroutine.Root

      def model_fn(dense_y, y_mask):
        """Model definition."""
        mean_student_ability = yield root(
            tfd.Normal(0.75, 1., name='mean_student_ability'))
        student_ability = yield root(
            tfd.Sample(
                tfd.Normal(0., 1.),
                dense_y.shape[0],
                name='student_ability',
            ))
        question_difficulty = yield root(
            tfd.Sample(
                tfd.Normal(0., 1.),
                dense_y.shape[1],
                name='question_difficulty',
            ))
        logits = (
            mean_student_ability[..., tf.newaxis, tf.newaxis] +
            student_ability[..., tf.newaxis] -
            question_difficulty[..., tf.newaxis, :])
        # TODO(b/150949917): Use a more dedicated masking functionality.
        masked_logits = logits * y_mask - 1e10 * (1 - y_mask)
        yield tfd.Independent(tfd.Bernoulli(masked_logits), 2, name='correct')

      train_joint_dist = tfd.JointDistributionCoroutine(
          functools.partial(model_fn, train_dense_y, train_y_mask))
      dtype = self._tuple_to_dict(train_joint_dist.dtype[:-1])

      sample_transformations = {
          'identity':
              bayesian_model.BayesianModel.SampleTransformation(
                  fn=lambda params: params,
                  pretty_name='Identity',
                  dtype=dtype,
              )
      }
      if self._have_test:
        if not (test_student_ids.shape[0] == test_question_ids.shape[0] ==
                test_correct.shape[0]):
          raise ValueError('`test_student_ids`, `test_question_ids` and '
                           '`test_correct` must all have the same length. '
                           'Got: {} {} {}'.format(test_student_ids.shape[0],
                                                  test_question_ids.shape[0],
                                                  test_correct.shape[0]))
        test_dense_y, test_y_mask = self._sparse_to_dense(
            test_student_ids,
            test_question_ids,
            test_correct,
        )
        test_joint_dist = tfd.JointDistributionCoroutine(
            functools.partial(model_fn, test_dense_y, test_y_mask))

        def _get_label_dist(params):
          # TODO(b/150897904): The seed does nothing since the model is fully
          # conditioned.
          distributions, _ = test_joint_dist.sample_distributions(
              value=self._dict_to_tuple(params) + (test_dense_y,), seed=42)
          return distributions[-1]

        sample_transformations['test_nll'] = (
            bayesian_model.BayesianModel.SampleTransformation(
                fn=lambda params: -(  # pylint: disable=g-long-lambda
                    _get_label_dist(params).log_prob(test_dense_y)),
                pretty_name='Test NLL',
            ))

        def _per_example_test_nll(params):
          """Computes per-example test NLL."""
          dense_nll = _get_label_dist(params).distribution.log_prob(
              test_dense_y)
          return self._dense_to_sparse(test_student_ids, test_question_ids,
                                       dense_nll)

        sample_transformations['per_example_test_nll'] = (
            bayesian_model.BayesianModel.SampleTransformation(
                fn=_per_example_test_nll,
                pretty_name='Per-example Test NLL',
            ))

    self._train_joint_dist = train_joint_dist
    self._train_correct = train_correct
    self._train_student_ids = train_student_ids
    self._train_question_ids = train_question_ids
    self._test_student_ids = test_student_ids
    self._test_question_ids = test_question_ids
    self._evidence_val = train_dense_y

    super(ItemResponseTheory, self).__init__(
        default_event_space_bijector=self._tuple_to_dict(
            (tfb.Identity(), tfb.Identity(), tfb.Identity())),
        event_shape=self._tuple_to_dict(train_joint_dist.event_shape[:-1]),
        dtype=dtype,
        name=name,
        pretty_name=pretty_name,
        sample_transformations=sample_transformations,
    )

  def _tuple_to_dict(self, params):
    return {
        'mean_student_ability': params[0],
        'student_ability': params[1],
        'question_difficulty': params[2],
    }

  def _dict_to_tuple(self, params):
    return (
        params['mean_student_ability'],
        params['student_ability'],
        params['question_difficulty'],
    )

  def _sparse_to_dense(self, student_ids, question_ids, correct):
    # TODO(siege): This probably should support batching, for completeness.
    dense_y = onp.zeros([self._num_students, self._num_questions], onp.float32)
    y_mask = onp.zeros_like(dense_y)
    dense_y[student_ids, question_ids] = correct
    y_mask[student_ids, question_ids] = 1.
    return dense_y, y_mask

  def _dense_to_sparse(self, student_ids, question_ids, dense_correct):
    test_y_idx = onp.stack([student_ids, question_ids], axis=-1)
    # Need to tile the indices across the batch, for gather_nd.
    batch_shape = ps.shape(dense_correct)[:-2]
    broadcast_shape = ps.concat([ps.ones_like(batch_shape), test_y_idx.shape],
                                axis=-1)
    test_y_idx = tf.reshape(test_y_idx, broadcast_shape)
    test_y_idx = tf.tile(test_y_idx, ps.concat([batch_shape, [1, 1]], axis=-1))
    return tf.gather_nd(
        dense_correct, test_y_idx, batch_dims=ps.size(batch_shape))

  def _sample_dataset(self, seed):
    dataset = dict(
        train_student_ids=self._train_student_ids,
        train_question_ids=self._train_question_ids,
        test_question_ids=self._test_question_ids,
        test_student_ids=self._test_student_ids,
    )
    train_samples = self._train_joint_dist.sample(seed=seed)
    train_correct = self._dense_to_sparse(self._train_student_ids,
                                          self._train_question_ids,
                                          train_samples[-1])
    dataset['train_correct'] = train_correct
    if self._have_test:
      test_samples = self._test_joint_dist.sample(value=train_samples[:-1])
      test_correct = self._dense_to_sparse(self._test_student_ids,
                                           self._test_question_ids,
                                           test_samples[-1])
      dataset['test_correct'] = test_correct
    return dataset

  def _joint_distribution(self):
    return self._train_joint_dist

  def _evidence(self):
    return self._evidence_val

  def _unnormalized_log_prob(self, value):
    return self.joint_distribution().log_prob(
        self._dict_to_tuple(value) + (self.evidence(),))


class SyntheticItemResponseTheory(ItemResponseTheory):
  """One-parameter logistic item-response theory (IRT) model.

  This uses a dataset sampled from the prior. This dataset is a simulation of
  400 students each answering a subset of 100 unique questions, with a total of
  30012 questions answered.
  """

  def __init__(self):
    dataset = data.synthetic_item_response_theory()
    del dataset['test_student_ids']
    del dataset['test_question_ids']
    del dataset['test_correct']
    super(SyntheticItemResponseTheory, self).__init__(
        name='synthetic_item_response_theory',
        pretty_name='Synthetic Item-Response Theory',
        **dataset)
