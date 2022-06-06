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

import functools

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import prefer_static as ps

from inference_gym.internal import data
from inference_gym.targets import bayesian_model
from inference_gym.targets import model
from inference_gym.targets.ground_truth import german_credit_numeric_sparse_logistic_regression

tfb = tfp.bijectors
tfd = tfp.distributions

__all__ = [
    'GermanCreditNumericSparseLogisticRegression',
    'SparseLogisticRegression',
]


def _add_bias(features):
  return tf.concat([features, tf.ones([ps.shape(features)[0], 1])], axis=-1)


class SparseLogisticRegression(bayesian_model.BayesianModel):
  """Bayesian logistic regression with a sparsity-inducing prior.

  ```none
  global_scale ~ Gamma(0.5, 0.5)

  # The `+ 1` is for the bias term.
  for i in range(num_features + 1):
    unscaled_weights[i] ~ Normal(loc=0, scale=1)
    local_scales[i] ~ Gamma(0.5, 0.5)
    weights[i] = unscaled_weights[i] * local_scales[i] * global_scale

  for j in range(num_datapoints):
    label[j] ~ Bernoulli(logit=concat([features[j], [1]) @ weights)
  ```
  """

  def __init__(self,
               train_features,
               train_labels,
               test_features=None,
               test_labels=None,
               positive_constraint_fn='exp',
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
      positive_constraint_fn: Python `str`. Which squashing function to use to
        enforce positivity of scales. Can be either `exp` or `softplus`.
      name: Python `str` name prefixed to Ops created by this class.
      pretty_name: A Python `str`. The pretty name of this model.

    Raises:
      ValueError: If `test_features` and `test_labels` are either not both
        `None` or not both specified.
    """
    with tf.name_scope(name):
      num_features = int(train_features.shape[1] + 1)

      self._prior_dist = tfd.JointDistributionNamed(
          dict(
              unscaled_weights=tfd.Sample(tfd.Normal(0., 1.), num_features),
              local_scales=tfd.Sample(tfd.Gamma(0.5, 0.5), num_features),
              global_scale=tfd.Gamma(0.5, 0.5),
          ))

      def log_likelihood_fn(unscaled_weights,
                            local_scales,
                            global_scale,
                            features,
                            labels,
                            reduce_sum=True):
        """The log_likelihood function."""
        features = tf.convert_to_tensor(features, tf.float32)
        features = _add_bias(features)
        labels = tf.convert_to_tensor(labels)

        weights = (
            unscaled_weights * local_scales * global_scale[..., tf.newaxis])

        logits = tf.einsum('nd,...d->...n', features, weights)
        log_likelihood = tfd.Bernoulli(logits=logits).log_prob(labels)
        if reduce_sum:
          return tf.reduce_sum(log_likelihood, [-1])
        else:
          return log_likelihood

      self._train_log_likelihood_fn = functools.partial(
          log_likelihood_fn, features=train_features, labels=train_labels)

      sample_transformations = {
          'identity':
              model.Model.SampleTransformation(
                  fn=lambda params: params,
                  pretty_name='Identity',
                  dtype=self._prior_dist.dtype,
              )
      }
      if (test_features is not None) != (test_labels is not None):
        raise ValueError('`test_features` and `test_labels` must either both '
                         'be `None` or both specified. Got: test_features={}, '
                         'test_labels={}'.format(test_features, test_labels))

      if test_features is not None and test_labels is not None:
        test_log_likelihood_fn = functools.partial(
            log_likelihood_fn, features=test_features, labels=test_labels)

        sample_transformations['test_nll'] = (
            model.Model.SampleTransformation(
                fn=lambda params: test_log_likelihood_fn(**params),
                pretty_name='Test NLL',
            ))
        sample_transformations['per_example_test_nll'] = (
            model.Model.SampleTransformation(
                fn=lambda params: test_log_likelihood_fn(  # pylint: disable=g-long-lambda
                    reduce_sum=False,
                    **params),
                pretty_name='Per-example Test NLL',
            ))

    if positive_constraint_fn == 'exp':
      scale_bijector = tfb.Exp()
    elif positive_constraint_fn == 'softplus':
      scale_bijector = tfb.Softplus()
    else:
      raise ValueError(
          f'Unknown positive_constraint_fn={positive_constraint_fn}')

    super(SparseLogisticRegression, self).__init__(
        default_event_space_bijector=dict(
            unscaled_weights=tfb.Identity(),
            local_scales=scale_bijector,
            global_scale=scale_bijector,
        ),
        event_shape=self._prior_dist.event_shape,
        dtype=self._prior_dist.dtype,
        name=name,
        pretty_name=pretty_name,
        sample_transformations=sample_transformations,
    )

  def _prior_distribution(self):
    return self._prior_dist

  def log_likelihood(self, value):
    return self._train_log_likelihood_fn(**value)


class GermanCreditNumericSparseLogisticRegression(SparseLogisticRegression):
  """Bayesian logistic regression with a sparsity-inducing prior.

  This model uses the German Credit (numeric) data set [1].

  #### References

  1. https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
  """

  GROUND_TRUTH_MODULE = german_credit_numeric_sparse_logistic_regression

  def __init__(self, positive_constraint_fn='exp'):
    dataset = data.german_credit_numeric()
    del dataset['test_features']
    del dataset['test_labels']
    super(GermanCreditNumericSparseLogisticRegression, self).__init__(
        name='german_credit_numeric_sparse_logistic_regression',
        pretty_name='German Credit Numeric Sparse Logistic Regression',
        positive_constraint_fn=positive_constraint_fn,
        **dataset)
