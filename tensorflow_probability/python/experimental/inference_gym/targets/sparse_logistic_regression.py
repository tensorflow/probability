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
"""Sparse logistic regression models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd

from tensorflow_probability.python.experimental.inference_gym.internal import data
from tensorflow_probability.python.experimental.inference_gym.targets import bayesian_model

__all__ = [
    'GermanCreditNumericSparseLogisticRegression',
    'SparseLogisticRegression',
]


def _add_bias(features):
  return tf.concat([features, tf.ones([tf.shape(features)[0], 1])], axis=-1)


class SparseLogisticRegression(bayesian_model.BayesianModel):
  """Bayesian logistic regression with a sparsity-inducing prior."""

  def __init__(self,
               train_features,
               train_labels,
               test_features=None,
               test_labels=None,
               name='sparse_logistic_regression',
               pretty_name='Sparse Logistic Regression'):
    """Construct the sparse logistic regression model.

    Args:
      train_features: Floating-point `Tensor` with shape `[num_train_points,
        num_features]`. Training features.
      train_labels: Integer `Tensor` with shape `[num_train_points]`. Training
        labels.
      test_features: Floating-point `Tensor` with shape `[num_test_points,
        num_features]`. Testing features. Can be `None`, in which case
        test-related sample transformations are not computed.
      test_labels: Integer `Tensor` with shape `[num_test_points]`. Testing
        labels. Can be `None`, in which case test-related sample transformations
        are not computed.
      name: Python `str` name prefixed to Ops created by this class.
      pretty_name: A Python `str`. The pretty name of this model.

    Raises:
      ValueError: If `test_features` and `test_labels` are either not both
        `None` or not both specified.
    """
    with tf.name_scope(name):
      train_features = _add_bias(train_features)
      train_labels = tf.convert_to_tensor(train_labels)
      num_features = int(train_features.shape[1])

      root = tfd.JointDistributionCoroutine.Root
      zero = tf.zeros(num_features)
      one = tf.ones(num_features)
      half = tf.fill([num_features], 0.5)

      def model_fn(features):
        """Model definition."""
        unscaled_weights = yield root(
            tfd.Independent(tfd.Normal(zero, one), 1, name='unscaled_weights'))
        local_scales = yield root(
            tfd.Independent(tfd.Gamma(half, half), 1, name='local_scales'))
        global_scale = yield root(tfd.Gamma(0.5, 0.5, name='global_scale'))

        weights = unscaled_weights * local_scales * global_scale[...,
                                                                 tf.newaxis]

        logits = tf.einsum('nd,...d->...n', features, weights)
        yield tfd.Independent(tfd.Bernoulli(logits=logits), 1, name='labels')

      train_joint_dist = tfd.JointDistributionCoroutine(
          functools.partial(model_fn, features=train_features))

      sample_transformations = {
          'identity':
              bayesian_model.BayesianModel.SampleTransformation(
                  fn=lambda params: params,
                  pretty_name='Identity',
              )
      }
      if (test_features is not None) != (test_labels is not None):
        raise ValueError('`test_features` and `test_labels` must either both '
                         'be `None` or both specified. Got: test_features={}, '
                         'test_labels={}'.format(test_features, test_labels))

      if test_features is not None and test_labels is not None:
        test_features = _add_bias(test_features)
        test_labels = tf.convert_to_tensor(test_labels)
        test_joint_dist = tfd.JointDistributionCoroutine(
            functools.partial(model_fn, features=test_features))

        def _get_label_dist(params):
          # TODO(b/150897904): The seed does nothing since the model is fully
          # conditioned.
          distributions, _ = test_joint_dist.sample_distributions(
              value=self._dict_to_tuple(params) + (test_labels,), seed=42)
          return distributions[-1]

        sample_transformations['test_nll'] = (
            bayesian_model.BayesianModel.SampleTransformation(
                fn=lambda params: -(  # pylint: disable=g-long-lambda
                    _get_label_dist(params).log_prob(test_labels)),
                pretty_name='Test NLL',
            ))
        sample_transformations['per_example_test_nll'] = (
            bayesian_model.BayesianModel.SampleTransformation(
                fn=lambda params: -(  # pylint: disable=g-long-lambda
                    _get_label_dist(params).distribution.log_prob(test_labels)),
                pretty_name='Per-example Test NLL',
            ))

    self._train_joint_dist = train_joint_dist
    self._train_labels = train_labels

    super(SparseLogisticRegression, self).__init__(
        default_event_space_bijector=self._tuple_to_dict(
            (tfb.Identity(), tfb.Exp(), tfb.Exp())),
        event_shape=self._tuple_to_dict(train_joint_dist.event_shape[:-1]),
        dtype=self._tuple_to_dict(train_joint_dist.dtype[:-1]),
        name=name,
        pretty_name=pretty_name,
        sample_transformations=sample_transformations,
    )

  def _tuple_to_dict(self, params):
    return {
        'unscaled_weights': params[0],
        'local_scales': params[1],
        'global_scale': params[2],
    }

  def _dict_to_tuple(self, params):
    return (
        params['unscaled_weights'],
        params['local_scales'],
        params['global_scale'],
    )

  def _joint_distribution(self):
    return self._train_joint_dist

  def _evidence(self):
    return self._train_labels

  def _unnormalized_log_prob(self, params):
    return self.joint_distribution().log_prob(
        self._dict_to_tuple(params) + (self.evidence(),))


class GermanCreditNumericSparseLogisticRegression(SparseLogisticRegression):
  """Bayesian logistic regression with a sparsity-inducing prior.

  This model uses the German Credit (numeric) data set [1].

  #### References

  1. https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
  """

  def __init__(self):
    dataset = data.german_credit_numeric()
    del dataset['test_features']
    del dataset['test_labels']
    super(GermanCreditNumericSparseLogisticRegression, self).__init__(
        name='german_credit_sparse_logistic_regression',
        pretty_name='German Credit Numeric Logistic Regression',
        **dataset)
