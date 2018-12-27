# Copyright 2018 The TensorFlow Probability Authors.
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
# ============================================================================
"""Layers for combining `tfp.distributions` and `tf.keras`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# Dependency imports
import numpy as np
import tensorflow as tf

# By importing `distributions` as `tfd`, docstrings will show
# `tfd.Distribution`. We import `bijectors` the same way, for consistency.
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.layers.internal import distribution_tensor_coercible as dtc
from tensorflow.python.keras.utils import tf_utils as keras_tf_utils


__all__ = [
    'CategoricalMixtureOfOneHotCategorical',
    'DistributionLambda',
    'IndependentBernoulli',
    'KLDivergenceAddLoss',
    'KLDivergenceRegularizer',
    'MultivariateNormalTriL',
    'OneHotCategorical',
]


keras_tf_utils.register_symbolic_tensor_type(
    dtc._TensorCoercible)  # pylint: disable=protected-access


class DistributionLambda(tf.keras.layers.Lambda):
  """Keras layer enabling plumbing TFP distributions through Keras models.

  A `DistributionLambda` is minimially characterized by a function that returns
  a `tfp.distributions.Distribution` instance.

  Since subsequent Keras layers are functions of tensors, a `DistributionLambda`
  also defines how the `tfp.distributions.Distribution` shall be "concretized"
  as a tensor. By default, a distribution is represented as a tensor via a
  random draw, e.g., `tfp.distributions.Distribution.sample`. Alternatively the
  user may provide a `callable` taking the distribution instance and producing a
  `tf.Tensor`.

  #### Examples

  ```python
  tfk = tf.keras
  tfkl = tf.keras.layers
  tfd = tfp.distributions
  tfpl = tfp.layers

  event_size = 7

  model = tfk.Sequential([
    tfkl.Dense(2),
    tfpl.DistributionLambda(
      make_distribution_fn=lambda t: tfd.Normal(
          loc=t[..., 0:1], scale=tf.exp(t[..., 1:2])),
      convert_to_tensor_fn=lambda s: s.sample(5))
  ])
  # ==> Normal (batch_shape=[1]) instance parametrized by mean and log scale.
  ```

  """

  def __init__(self,
               make_distribution_fn,
               convert_to_tensor_fn=tfd.Distribution.sample,
               **kwargs):
    """Create a `DistributionLambda` Keras layer.

    Args:
      make_distribution_fn: Python `callable` that takes previous layer outputs
        and returns a `tfp.Distribution` instance.
      convert_to_tensor_fn: Python `callable` that takes a `tfp.Distribution`
        instance and returns a `tf.Tensor`-like object. For examples, see
        `class` docstring.
        Default value: `tfd.Distribution.sample`.
      **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
    """
    # TODO(b/120440642): See if something like this code block is needed.
    # if output_shape is None:
    #   def default_output_shape(keras_input_shape):
    #     output_shape = map(
    #         _expand_rank_at_least_one,
    #         [sample_shape, keras_input_shape[:-1], event_shape])
    #     return tf.concat(output_shape, axis=0)
    #   output_shape = default_output_shape

    if isinstance(convert_to_tensor_fn, property):
      convert_to_tensor_fn = convert_to_tensor_fn.fget

    def _fn(*fargs, **fkwargs):
      """Wraps `make_distribution_fn` to return both dist and concrete value."""
      distribution = dtc._TensorCoercible(  # pylint: disable=protected-access
          distribution=make_distribution_fn(*fargs, **fkwargs),
          convert_to_tensor_fn=convert_to_tensor_fn)
      value = tf.convert_to_tensor(distribution)
      # TODO(b/120153609): Keras is incorrectly presuming everything is a
      # `tf.Tensor`. Closing this bug entails ensuring Keras only accesses
      # `tf.Tensor` properties after calling `tf.convert_to_tensor`.
      distribution.shape = value.shape
      distribution.get_shape = value.get_shape
      return distribution, value

    super(DistributionLambda, self).__init__(_fn, **kwargs)

    # We'll need to keep track of who's calling who since the functional
    # API has a different way of injecting `_keras_history` than the
    # `keras.Sequential` way.
    self._enter_dunder_call = False

  def __call__(self, inputs, *args, **kwargs):
    self._enter_dunder_call = True
    distribution, _ = super(DistributionLambda, self).__call__(
        inputs, *args, **kwargs)
    self._enter_dunder_call = False
    return distribution

  def call(self, inputs, *args, **kwargs):
    distribution, value = super(DistributionLambda, self).call(
        inputs, *args, **kwargs)
    if self._enter_dunder_call:
      # Its critical to return both distribution and concretization
      # so Keras can inject `_keras_history` to both. This is what enables
      # either to be used as an input to another Keras `Model`.
      return distribution, value
    return distribution


# TODO(b/120160878): Add more shape validation logic to each layer. Consider
# also adding additional functionality to help the user determine the
# appropriate size of the parameterizing vector.


class MultivariateNormalTriL(DistributionLambda):
  """A `d`-variate MVNTriL Keras layer from `d + d * (d + 1) // 2` params.

  Typical choices for `convert_to_tensor_fn` include:

  - `tfd.Distribution.sample`
  - `tfd.Distribution.mean`
  - `tfd.Distribution.mode`
  - `lambda s: s.mean() + 0.1 * s.stddev()`


  #### Example

  ```python
  tfk = tf.keras
  tfkl = tf.keras.layers
  tfd = tfp.distributions
  tfpl = tfp.layers

  # Load data.
  n = int(1e3)
  scale_tril = np.array([[1.6180, 0.],
                         [-2.7183, 3.1416]]).astype(np.float32)
  x = tfd.Normal(loc=0, scale=1).sample([n, 2])
  eps = tfd.Normal(loc=0, scale=0.01).sample([n, 2])
  y = tf.matmul(x, scale_tril) + eps

  # Create model.
  d = y.shape[-1].value
  model = tfk.Sequential([
      tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(d)),
      tfpl.MultivariateNormalTriL(d),
  ])

  # Fit.
  model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.02),
                loss=lambda y, model: -model.log_prob(y),
                metrics=[])
  batch_size = 100
  model.fit(x, y,
            batch_size=batch_size,
            epochs=500,
            steps_per_epoch=n // batch_size,
            verbose=True,
            shuffle=True)
  model.get_weights()[0][:, :2]
  # ==> [[  1.61842895e+00   1.34138885e-04]
  #      [ -2.71818233e+00   3.14186454e+00]]
  ```

  """

  def __init__(self,
               event_size,
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    """Initialize the `MultivariateNormalTriL` layer.

    Args:
      event_size: Scalar `int` representing the size of single draw from this
        distribution.
      convert_to_tensor_fn: Python `callable` that takes a `tfp.Distribution`
        instance and returns a `tf.Tensor`-like object. For examples, see
        `class` docstring.
        Default value: `tfd.Distribution.sample`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
    """
    super(MultivariateNormalTriL, self).__init__(
        lambda t: type(self).new(t, event_size, validate_args),
        convert_to_tensor_fn,
        **kwargs)

  @staticmethod
  def new(params, event_size, validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.name_scope(name, 'MultivariateNormalTriL', [params, event_size]):
      return tfd.MultivariateNormalTriL(
          loc=params[..., :event_size],
          scale_tril=tfb.ScaleTriL(validate_args=validate_args)(
              params[..., event_size:]),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_size, name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.name_scope(name, 'MultivariateNormalTriL_params_size',
                       [event_size]):
      return event_size + event_size * (event_size + 1) // 2


class OneHotCategorical(DistributionLambda):
  """A `d`-variate OneHotCategorical Keras layer from `d` params.

  Typical choices for `convert_to_tensor_fn` include:

  - `tfd.Distribution.sample`
  - `tfd.Distribution.mean`
  - `tfd.Distribution.mode`
  - `tfd.OneHotCategorical.logits`


  #### Example

  ```python
  tfk = tf.keras
  tfkl = tf.keras.layers
  tfd = tfp.distributions
  tfpl = tfp.layers

  # Load data.
  n = int(1e4)
  scale_noise = 0.01
  x = tfd.Normal(loc=0, scale=1).sample([n, 2])
  eps = tfd.Normal(loc=0, scale=scale_noise).sample([n, 1])
  y = tfd.OneHotCategorical(
      logits=tf.pad(0.3142 + 1.6180 * x[..., :1] - 2.7183 * x[..., 1:] + eps,
                    paddings=[[0, 0], [1, 0]]),
      dtype=tf.float32).sample()

  # Create model.
  d = y.shape[-1].value
  model = tfk.Sequential([
      tfk.layers.Dense(tfpl.OneHotCategorical.params_size(d) - 1),
      tfk.layers.Lambda(lambda x: tf.pad(x, paddings=[[0, 0], [1, 0]])),
      tfpl.OneHotCategorical(d),
  ])

  # Fit.
  model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.5),
                loss=lambda y, model: -model.log_prob(y),
                metrics=[])
  batch_size = 100
  model.fit(x, y,
            batch_size=batch_size,
            epochs=10,
            steps_per_epoch=n // batch_size,
            shuffle=True)
  model.get_weights()
  # ==> [np.array([[1.6180],
  #                [-2.7183]], np.float32),
  #      np.array([0.3142], np.float32)]   # Within 15% rel. error.
  ```

  """

  def __init__(self,
               event_size,
               convert_to_tensor_fn=tfd.Distribution.sample,
               sample_dtype=None,
               validate_args=False,
               **kwargs):
    """Initialize the `OneHotCategorical` layer.

    Args:
      event_size: Scalar `int` representing the size of single draw from this
        distribution.
      convert_to_tensor_fn: Python `callable` that takes a `tfp.Distribution`
        instance and returns a `tf.Tensor`-like object. For examples, see
        `class` docstring.
        Default value: `tfd.Distribution.sample`.
      sample_dtype: `dtype` of samples produced by this distribution.
        Default value: `None` (i.e., previous layer's `dtype`).
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
    """
    super(OneHotCategorical, self).__init__(
        lambda t: type(self).new(t, event_size, sample_dtype, validate_args),
        convert_to_tensor_fn,
        **kwargs)

  @staticmethod
  def new(params, event_size, dtype=None, validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.name_scope(name, 'OneHotCategorical', [params, event_size]):
      return tfd.OneHotCategorical(
          logits=params,
          dtype=dtype or params.dtype.base_dtype,
          validate_args=validate_args)

  @staticmethod
  def params_size(event_size, name=None):
    """The number of `params` needed to create a single distribution."""
    return event_size


class CategoricalMixtureOfOneHotCategorical(DistributionLambda):
  """A OneHotCategorical mixture Keras layer from `k * (1 + d)` params.

  `k` (i.e., `num_components`) represents the number of component
  `OneHotCategorical` distributions and `d` (i.e., `event_size`) represents the
  number of categories within each `OneHotCategorical` distribution.

  Typical choices for `convert_to_tensor_fn` include:

  - `tfd.Distribution.sample`
  - `tfd.Distribution.mean`
  - `tfd.Distribution.mode`
  - `lambda s: s.log_mean()`


  #### Example

  ```python
  tfk = tf.keras
  tfkl = tf.keras.layers
  tfd = tfp.distributions
  tfpl = tfp.layers

  # Load data.
  n = int(1e4)
  scale_noise = 0.01
  x = tfd.Normal(loc=0, scale=1).sample([n, 2])
  eps = tfd.Normal(loc=0, scale=scale_noise).sample([n, 1])
  y = tfd.OneHotCategorical(
      logits=tf.pad(0.3142 + 1.6180 * x[..., :1] - 2.7183 * x[..., 1:] + eps,
                    paddings=[[0, 0], [1, 0]]),
      dtype=tf.float32).sample()

  # Create model.
  d = y.shape[-1].value
  k = 2
  model = tfk.Sequential([
      tfkl.Dense(tfpl.CategoricalMixtureOfOneHotCategorical.params_size(d, k)),
      tfpl.CategoricalMixtureOfOneHotCategorical(d, k),
  ])

  # Fit.
  model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.5),
                loss=lambda y, model: -tf.reduce_mean(model.log_prob(y)),
                metrics=[])
  batch_size = 100
  model.fit(x, y,
            batch_size=batch_size,
            epochs=10,
            steps_per_epoch=n // batch_size,
            shuffle=True)
  print(model.get_weights())
  ```

  """

  def __init__(self,
               event_size,
               num_components,
               convert_to_tensor_fn=tfd.Distribution.sample,
               sample_dtype=None,
               validate_args=False,
               **kwargs):
    """Initialize the `CategoricalMixtureOfOneHotCategorical` layer.

    Args:
      event_size: Scalar `int` representing the size of single draw from this
        distribution.
      num_components: Scalar `int` representing the number of mixture
        components. Must be at least 1. (If `num_components=1`, it's more
        efficient to use the `OneHotCategorical` layer.)
      convert_to_tensor_fn: Python `callable` that takes a `tfp.Distribution`
        instance and returns a `tf.Tensor`-like object. For examples, see
        `class` docstring.
        Default value: `tfd.Distribution.sample`.
      sample_dtype: `dtype` of samples produced by this distribution.
        Default value: `None` (i.e., previous layer's `dtype`).
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
    """
    super(CategoricalMixtureOfOneHotCategorical, self).__init__(
        lambda t: type(self).new(  # pylint: disable=g-long-lambda
            t, event_size, num_components, sample_dtype, validate_args),
        convert_to_tensor_fn,
        **kwargs)

  @staticmethod
  def new(params, event_size, num_components,
          dtype=None, validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.name_scope(name, 'CategoricalMixtureOfOneHotCategorical',
                       [params, event_size, num_components]):
      components_shape = tf.concat(
          [tf.shape(params)[:-1], [num_components, event_size]], axis=0)
      dist = tfd.MixtureSameFamily(
          mixture_distribution=tfd.Categorical(
              logits=params[..., :num_components],
              validate_args=validate_args),
          components_distribution=tfd.OneHotCategorical(
              logits=tf.reshape(params[..., num_components:],
                                components_shape),
              dtype=dtype or params.dtype.base_dtype,
              validate_args=False),  # So we can eval on simplex interior.
          # TODO(b/120154797): Change following to `validate_args=True` after
          # fixing: "ValueError: `mixture_distribution` must have scalar
          # `event_dim`s." assertion in MixtureSameFamily.
          validate_args=False)
      # pylint: disable=protected-access
      dist._mean = functools.partial(
          _eval_all_one_hot, tfd.Distribution.prob, dist)
      dist.log_mean = functools.partial(
          _eval_all_one_hot, tfd.Distribution.log_prob, dist)
      # pylint: enable=protected-access
      return dist

  @staticmethod
  def params_size(event_size, num_components, name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.name_scope(
        name, 'CategoricalMixtureOfOneHotCategorical_params_size',
        [event_size, num_components]):
      return num_components * (1 + event_size)


class IndependentBernoulli(DistributionLambda):
  """An Independent-Bernoulli Keras layer from `prod(event_shape)` params.

  Typical choices for `convert_to_tensor_fn` include:

  - `tfd.Distribution.sample`
  - `tfd.Distribution.mean`
  - `tfd.Distribution.mode`
  - `tfd.Bernoulli.logits`


  #### Example

  ```python
  tfk = tf.keras
  tfkl = tf.keras.layers
  tfd = tfp.distributions
  tfpl = tfp.layers

  # Load data.
  n = int(1e4)
  scale_tril = np.array([[1.6180, 0.],
                         [-2.7183, 3.1416]]).astype(np.float32)
  scale_noise = 0.01
  x = tfd.Normal(loc=0, scale=1).sample([n, 2])
  eps = tfd.Normal(loc=0, scale=scale_noise).sample([n, 2])
  y = tfd.Bernoulli(logits=tf.reshape(
      tf.matmul(x, scale_tril) + eps,
      shape=[n, 1, 2, 1])).sample()

  # Create model.
  event_shape = y.shape[1:].as_list()
  model = tfk.Sequential([
      tfkl.Dense(tfpl.IndependentBernoulli.params_size(event_shape)),
      tfpl.IndependentBernoulli(event_shape),
  ])

  # Fit.
  model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.5),
                loss=lambda y, model: -model.log_prob(y),
                metrics=[])
  batch_size = 100
  model.fit(x, y,
            batch_size=batch_size,
            epochs=10,
            steps_per_epoch=n // batch_size,
            shuffle=True)
  print(model.get_weights())
  # ==> [np.array([[1.6180, 0.],
  #                [-2.7183, 3.1416]], np.float32),
  #      array([0., 0.], np.float32)]   # Within 15% rel. error.
  ```

  """

  def __init__(self,
               event_shape,
               convert_to_tensor_fn=tfd.Distribution.sample,
               sample_dtype=None,
               validate_args=False,
               **kwargs):
    """Initialize the `IndependentBernoulli` layer.

    Args:
      event_shape: `int` vector representing the size of single draw from this
        distribution.
      convert_to_tensor_fn: Python `callable` that takes a `tfp.Distribution`
        instance and returns a `tf.Tensor`-like object. For examples, see
        `class` docstring.
        Default value: `tfd.Distribution.sample`.
      sample_dtype: `dtype` of samples produced by this distribution.
        Default value: `None` (i.e., previous layer's `dtype`).
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
    """
    super(IndependentBernoulli, self).__init__(
        lambda t: type(self).new(t, event_shape, sample_dtype, validate_args),
        convert_to_tensor_fn,
        **kwargs)

  @staticmethod
  def new(params, event_shape, dtype=None, validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.name_scope(name, 'IndependentBernoulli', [params, event_shape]):
      new_shape = tf.concat([
          tf.shape(params)[:-1],
          event_shape,
      ], axis=0)
      dist = tfd.Independent(
          tfd.Bernoulli(logits=tf.reshape(params, new_shape),
                        dtype=dtype or params.dtype.base_dtype,
                        validate_args=validate_args),
          reinterpreted_batch_ndims=tf.size(event_shape),
          validate_args=validate_args)
      dist._logits = dist.distribution._logits  # pylint: disable=protected-access
      dist._probs = dist.distribution._probs  # pylint: disable=protected-access
      dist.logits = tfd.Bernoulli.logits
      dist.probs = tfd.Bernoulli.probs
      return dist

  @staticmethod
  def params_size(event_shape, name=None):
    """The number of `params` needed to create a single distribution."""
    try:
      return np.prod(np.array(event_shape, dtype=np.int32))
    finally:
      pass   # Fallback to TensorFlow. pylint: disable=lost-exception
    with tf.name_scope(name, 'IndependentBernoulli_params_size', [event_shape]):
      event_shape = tf.convert_to_tensor(
          event_shape, dtype=tf.int32, name='event_shape')
      event_shape_ = tf.contrib.util.constant_value(event_shape)
      return (tf.reduce_prod(event_shape) if event_shape_ is None
              else np.prod(event_shape_))


def _eval_all_one_hot(fn, dist, name=None):
  """OneHotCategorical helper computing probs, cdf, etc over its support."""
  with tf.name_scope(name, 'eval_all_one_hot'):
    event_size = dist.event_shape_tensor()[-1]
    batch_ndims = tf.size(dist.batch_shape_tensor())
    # Reshape `eye(d)` to: `[d] + [1]*batch_ndims + [d]`.
    x = tf.reshape(
        tf.eye(event_size, dtype=dist.dtype),
        shape=tf.pad(tf.ones(batch_ndims, tf.int32),
                     paddings=[[1, 1]],
                     constant_values=event_size))
    # Compute `fn(x)` then cyclically left-transpose one dim.
    perm = tf.pad(tf.range(1, batch_ndims + 1), paddings=[[0, 1]])
    return tf.transpose(fn(dist, x), perm)


class KLDivergenceRegularizer(tf.keras.regularizers.Regularizer):
  """Regularizer that adds a KL divergence penalty to the model loss.

  When using Monte Carlo approximation (e.g., `use_exact=False`), it is presumed
  that the input distribution's concretization (i.e.,
  `tf.convert_to_tensor(distribution)`) corresponds to a random sample. To
  override this behavior, set `test_points_fn`.

  #### Example

  ```python
  tfd = tfp.distributions
  tfpl = tfp.layers
  tfk = tf.keras
  tfkl = tf.keras.layers

  # Create a variational encoder and add a KL Divergence penalty to the
  # loss that encourages marginal coherence with a unit-MVN (the "prior").
  input_shape = [28, 28, 1]
  encoded_size = 2
  variational_encoder = tfk.Sequential([
      tfkl.InputLayer(input_shape=input_shape),
      tfkl.Flatten(),
      tfkl.Dense(10, activation='relu'),
      tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size)),
      tfpl.MultivariateNormalTriL(
          encoded_size,
          lambda s: s.sample(10),
          activity_regularizer=tfpl.KLDivergenceRegularizer(
             tfd.MultivariateNormalDiag(loc=tf.zeros(encoded_size)),
             weight=num_train_samples)),
  ])
  ```

  """

  def __init__(self,
               distribution_b,
               use_exact_kl=False,
               test_points_reduce_axis=(),  # `None` == "all"; () == "none".
               test_points_fn=tf.convert_to_tensor,
               weight=None):
    """Initialize the `KLDivergenceRegularizer` regularizer.

    Args:
      distribution_b: distribution instance corresponding to `b` as in
        `KL[a, b]`. The previous layer's output is presumed to be a
        `Distribution` instance and is `a`).
      use_exact_kl: Python `bool` indicating if KL divergence should be
        calculated exactly via `tfp.distributions.kl_divergence` or via Monte
        Carlo approximation.
        Default value: `False`.
      test_points_reduce_axis: `int` vector or scalar representing dimensions
        over which to `reduce_mean` while calculating the Monte Carlo
        approximation of the KL divergence.  As is with all `tf.reduce_*` ops,
        `None` means reduce over all dimensions; `()` means reduce over none of
        them.
        Default value: `()` (i.e., no reduction).
      test_points_fn: Python `callable` taking a `Distribution` instance and
        returning a `Tensor` used for random test points to approximate the KL
        divergence.
        Default value: `tf.convert_to_tensor`.
      weight: Multiplier applied to the calculated KL divergence for each Keras
        batch member.
        Default value: `None` (i.e., do not weight each batch member).
    """
    super(KLDivergenceRegularizer, self).__init__()
    self._kl_divergence_fn = _make_kl_divergence_fn(
        distribution_b,
        use_exact_kl=use_exact_kl,
        test_points_reduce_axis=test_points_reduce_axis,
        test_points_fn=test_points_fn,
        weight=weight)

  def __call__(self, distribution_a):
    return self._kl_divergence_fn(distribution_a)


# TODO(b/120307671): Once this bug is resolved, consider deprecating
# `KLDivergenceAddLoss` and instead having users do:
# `activity_regularizer=tfp.layers.KLDivergenceRegularizer`


class KLDivergenceAddLoss(tf.keras.layers.Layer):
  """Pass-through layer that adds a KL divergence penalty to the model loss.

  When using Monte Carlo approximation (e.g., `use_exact=False`), it is presumed
  that the input distribution's concretization (i.e.,
  `tf.convert_to_tensor(distribution)`) corresponds to a random sample. To
  override this behavior, set `test_points_fn`.

  #### Example

  ```python
  tfd = tfp.distributions
  tfpl = tfp.layers
  tfk = tf.keras
  tfkl = tf.keras.layers

  # Create a variational encoder and add a KL Divergence penalty to the
  # loss that encourages marginal coherence with a unit-MVN (the "prior").
  input_shape = [28, 28, 1]
  encoded_size = 2
  variational_encoder = tfk.Sequential([
      tfkl.InputLayer(input_shape=input_shape),
      tfkl.Flatten(),
      tfkl.Dense(10, activation='relu'),
      tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size)),
      tfpl.MultivariateNormalTriL(encoded_size, lambda s: s.sample(10)),
      tfpl.KLDivergenceAddLoss(
          tfd.MultivariateNormalDiag(loc=tf.zeros(encoded_size)),
          weight=num_train_samples),
  ])
  ```

  """

  def __init__(self,
               distribution_b,
               use_exact_kl=False,
               test_points_reduce_axis=None,
               test_points_fn=tf.convert_to_tensor,
               weight=None,
               **kwargs):
    """Initialize the `KLDivergenceAddLoss` (placeholder) layer.

    Args:
      distribution_b: distribution instance corresponding to `b` as in
        `KL[a, b]`. The previous layer's output is presumed to be a
        `Distribution` instance and is `a`).
      use_exact_kl: Python `bool` indicating if KL divergence should be
        calculated exactly via `tfp.distributions.kl_divergence` or via Monte
        Carlo approximation.
        Default value: `False`.
      test_points_reduce_axis: `int` vector or scalar representing dimensions
        over which to `reduce_mean` while calculating the Monte Carlo
        approximation of the KL divergence.  As is with all `tf.reduce_*` ops,
        `None` means reduce over all dimensions; `()` means reduce over none of
        them.
        Default value: `()` (i.e., no reduction).
      test_points_fn: Python `callable` taking a `Distribution` instance and
        returning a `Tensor` used for random test points to approximate the KL
        divergence.
        Default value: `tf.convert_to_tensor`.
      weight: Multiplier applied to the calculated KL divergence for each Keras
        batch member.
        Default value: `None` (i.e., do not weight each batch member).
      **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
    """
    super(KLDivergenceAddLoss, self).__init__(**kwargs)
    self.is_placeholder = True
    # TODO(b/120307671): Call `_make_kl_divergence_fn` directly once this bug is
    # closed. Chaining things this way means we can have just one unit-test for
    # both `KLDivergenceAddLoss` and `KLDivergenceRegularizer`. That this is
    # good is because its not possible to idiomatically test
    # `KLDivergenceRegularizer` because of b/120307671.
    self._kl_divergence_fn = KLDivergenceRegularizer(
        distribution_b,
        use_exact_kl=use_exact_kl,
        test_points_reduce_axis=test_points_reduce_axis,
        test_points_fn=test_points_fn,
        weight=weight).__call__

  def call(self, distribution_a):
    self.add_loss(self._kl_divergence_fn(distribution_a),
                  inputs=[distribution_a])
    return distribution_a


def _make_kl_divergence_fn(
    distribution_b,
    use_exact_kl=False,
    test_points_reduce_axis=(),  # `None` == "all"; () == "none".
    test_points_fn=tf.convert_to_tensor,
    weight=None):
  """Creates a callable computing `KL[a,b]` from `a`, a `tfd.Distribution`."""

  if use_exact_kl is None:
    kl_divergence_fn = tfd.kl_divergence
  else:
    # Closure over: test_points_fn, test_points_reduce_axis.
    def kl_divergence_fn(distribution_a, distribution_b):
      z = test_points_fn(distribution_a)
      return tf.reduce_mean(
          distribution_a.log_prob(z) - distribution_b.log_prob(z),
          axis=test_points_reduce_axis)

  # Closure over: distribution_b, kl_divergence_fn, weight.
  def _fn(distribution_a):
    """Closure that computes KLDiv as a function of `a` as in `KL[a, b]`."""
    with tf.name_scope('kldivergence_loss'):
      # TODO(b/119756336): Due to eager/graph Jacobian graph caching bug
      # we add here the capability for deferred construction of the prior.
      # This capability can probably be removed once b/119756336 is resolved.
      distribution_b_ = (distribution_b() if callable(distribution_b)
                         else distribution_b)
      kl = kl_divergence_fn(distribution_a, distribution_b_)
      if weight is not None:
        kl = tf.cast(weight, dtype=kl.dtype) * kl
      # Losses appended with the model.add_loss and are expected to be a single
      # scalar, unlike model.loss, which is expected to be the loss per sample.
      # Therefore, we reduce over all dimensions, regardless of the shape.
      return tf.reduce_mean(kl)

  return _fn
