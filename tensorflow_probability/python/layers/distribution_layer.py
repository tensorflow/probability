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

import codecs
import collections
import functools
import io
import pickle

# Dependency imports
from cloudpickle import CloudPickler
import numpy as np
import six
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.bijectors import fill_scale_tril as fill_scale_tril_lib
from tensorflow_probability.python.bijectors import transpose as transpose_lib
from tensorflow_probability.python.distributions import bernoulli as bernoulli_lib
from tensorflow_probability.python.distributions import categorical as categorical_lib
from tensorflow_probability.python.distributions import distribution as tfd
from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
from tensorflow_probability.python.distributions import logistic as logistic_lib
from tensorflow_probability.python.distributions import mixture_same_family as mixture_same_family_lib
from tensorflow_probability.python.distributions import mvn_tril as mvn_tril_lib
from tensorflow_probability.python.distributions import normal as normal_lib
from tensorflow_probability.python.distributions import onehot_categorical as onehot_categorical_lib
from tensorflow_probability.python.distributions import poisson as poisson_lib
from tensorflow_probability.python.distributions import transformed_distribution as transformed_distribution_lib
from tensorflow_probability.python.distributions import variational_gaussian_process as variational_gaussian_process_lib
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.layers.internal import distribution_tensor_coercible as dtc
from tensorflow_probability.python.layers.internal import tensor_tuple
from tensorflow.python.keras.utils import tf_utils as keras_tf_utils  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'CategoricalMixtureOfOneHotCategorical',
    'DistributionLambda',
    'IndependentBernoulli',
    'IndependentLogistic',
    'IndependentNormal',
    'IndependentPoisson',
    'KLDivergenceAddLoss',
    'KLDivergenceRegularizer',
    'MixtureLogistic',
    'MixtureNormal',
    'MixtureSameFamily',
    'MultivariateNormalTriL',
    'OneHotCategorical',
    'VariationalGaussianProcess',
]


keras_tf_utils.register_symbolic_tensor_type(dtc._TensorCoercible)  # pylint: disable=protected-access


def _event_size(event_shape, name=None):
  """Computes the number of elements in a tensor with shape `event_shape`.

  Args:
    event_shape: A tensor shape.
    name: The name to use for the tensor op to compute the number of elements
      (if such an op needs to be created).

  Returns:
    event_size: The number of elements in `tensor_shape`.  Returns a numpy int
    when the number of elements can be computed immediately.  Otherwise, returns
    a scalar tensor.
  """
  with tf.name_scope(name or 'event_size'):
    event_shape = tf.convert_to_tensor(
        event_shape, dtype=tf.int32, name='event_shape')

    event_shape_const = tf.get_static_value(event_shape)
    if event_shape_const is not None:
      return np.prod(event_shape_const, dtype=np.int32)
    else:
      return tf.reduce_prod(event_shape)


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

  model = tfk.Sequential([
    tfkl.Dense(2, input_dim=2),
    tfpl.DistributionLambda(
      make_distribution_fn=lambda t: tfd.Normal(
          loc=t[..., 0], scale=tf.exp(t[..., 1])),
      convert_to_tensor_fn=lambda s: s.sample(5))
  ])
  # model.call(x), where x.shape = B + [2] will produce
  # ==> Normal (batch_shape=[B]) instance parameterized by mean and log scale.
  ```

  """

  def __init__(self,
               make_distribution_fn,
               convert_to_tensor_fn=tfd.Distribution.sample,
               **kwargs):
    """Create a `DistributionLambda` Keras layer.

    Args:
      make_distribution_fn: Python `callable` that takes previous layer outputs
        and returns a `tfd.Distribution` instance.
      convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
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

    if isinstance(make_distribution_fn, six.string_types):
      # We are being called from `from_config`, so need to un-pickle.
      make_distribution_fn = _deserialize_function(make_distribution_fn)

    convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

    # If there is a 'function' keyword argument (e.g., because we are being
    # called from a `from_config` method), remove it.  We pass a function to
    # `Lambda.__init__` below as the first positional argument.
    kwargs.pop('function', None)

    def _fn(*fargs, **fkwargs):
      """Wraps `make_distribution_fn` to return both dist and concrete value."""
      d = make_distribution_fn(*fargs, **fkwargs)
      value_is_seq = isinstance(d.dtype, collections.Sequence)
      maybe_composite_convert_to_tensor_fn = (
          (lambda d: tensor_tuple.TensorTuple(convert_to_tensor_fn(d)))
          if value_is_seq else convert_to_tensor_fn)
      distribution = dtc._TensorCoercible(  # pylint: disable=protected-access
          distribution=d,
          convert_to_tensor_fn=maybe_composite_convert_to_tensor_fn)

      # Calling `distrbution._value()` is equivalent to:
      # from tensorflow.python.framework import ops
      # value = ops.convert_to_tensor_or_composite(distribution)
      # We'd prefer to call ops.convert_to_tensor_or_composite but do not,
      # favoring our own non-public API over TF's.
      value = distribution._value()  # pylint: disable=protected-access

      # TODO(b/126056144): Remove silent handle once we identify how/why Keras
      # is losing the distribution handle for activity_regularizer.
      value._tfp_distribution = distribution  # pylint: disable=protected-access
      # TODO(b/120153609): Keras is incorrectly presuming everything is a
      # `tf.Tensor`. Closing this bug entails ensuring Keras only accesses
      # `tf.Tensor` properties after calling `tf.convert_to_tensor`.
      if value_is_seq:
        value.shape = value[-1].shape
        value.get_shape = value[-1].get_shape
        value.dtype = value[-1].dtype
        distribution.shape = value[-1].shape
        distribution.get_shape = value[-1].get_shape
      else:
        distribution.shape = value.shape
        distribution.get_shape = value.get_shape
      return distribution, value

    super(DistributionLambda, self).__init__(_fn, **kwargs)

    # We need to ensure Keras tracks variables (eg, from activity regularizers
    # for type-II MLE). To accomplish this, we add the built distribution
    # variables and kwargs as members so `vars` picks them up (this is how
    # tf.Module implements its introspection).
    # Note also that we track all variables to support the user pattern:
    # `v.initializer for v in model.variable]`.
    self._most_recently_built_distribution_vars = None
    self._kwargs = kwargs

    self._make_distribution_fn = make_distribution_fn
    self._convert_to_tensor_fn = convert_to_tensor_fn

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
    # We always save the most recently built distribution variables for tracking
    # purposes.
    self._most_recently_built_distribution_vars = distribution.variables
    if self._enter_dunder_call:
      # Its critical to return both distribution and concretization
      # so Keras can inject `_keras_history` to both. This is what enables
      # either to be used as an input to another Keras `Model`.
      return distribution, value
    return distribution

  def get_config(self):
    """Returns the config of this layer.

    This Layer's `make_distribution_fn` is serialized via a library built on
    Python pickle.  This serialization of Python functions is provided for
    convenience, but:

      1. The use of this format for long-term storage of models is discouraged.
         In particular, it may not be possible to deserialize in a different
         version of Python.

      2. While serialization is generally supported for lambdas, local
         functions, and static methods (and closures over these constructs),
         complex functions may fail to serialize.

      3. `Tensor` objects (and functions referencing `Tensor` objects) can only
         be serialized when the tensor value is statically known.  (Such Tensors
         are serialized as numpy arrays.)

    Instead of relying on `DistributionLambda.get_config`, consider subclassing
    `DistributionLambda` and directly implementing Keras serialization via
    `get_config` / `from_config`.

    NOTE: At the moment, `DistributionLambda` can only be serialized if the
    `convert_to_tensor_fn` is a serializable Keras object (i.e., implements
    `get_config`) or one of the standard values:
     - `Distribution.sample` (or `"sample"`)
     - `Distribution.mean` (or `"mean"`)
     - `Distribution.mode` (or `"mode"`)
     - `Distribution.stddev` (or `"stddev"`)
     - `Distribution.variance` (or `"variance"`)
    """
    config = {
        'make_distribution_fn': _serialize_function(self._make_distribution_fn),
        'convert_to_tensor_fn': _serialize(self._convert_to_tensor_fn),
    }
    base_config = super(DistributionLambda, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _variable_creator(self, next_creator, **kwargs):
    """Overrides `tfkl.Lambda` method causing error in weight initialization."""
    return next_creator(**kwargs)


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
  d = tf.dimension_value(y.shape[-1])
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
      convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
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
        lambda t: MultivariateNormalTriL.new(t, event_size, validate_args),
        convert_to_tensor_fn,
        **kwargs)

  @staticmethod
  def new(params, event_size, validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.name_scope(name or 'MultivariateNormalTriL'):
      params = tf.convert_to_tensor(params, name='params')
      scale_tril = fill_scale_tril_lib.FillScaleTriL(
          diag_shift=np.array(1e-5, params.dtype.as_numpy_dtype),
          validate_args=validate_args)
      return mvn_tril_lib.MultivariateNormalTriL(
          loc=params[..., :event_size],
          scale_tril=scale_tril(params[..., event_size:]),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_size, name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.name_scope(name or 'MultivariateNormalTriL_params_size'):
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
  d = tf.dimension_value(y.shape[-1])
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
      convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
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
        lambda t: OneHotCategorical.new(  # pylint: disable=g-long-lambda
            t, event_size, sample_dtype, validate_args),
        convert_to_tensor_fn,
        **kwargs)

  @staticmethod
  def new(params, event_size, dtype=None, validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.name_scope(name or 'OneHotCategorical'):
      return onehot_categorical_lib.OneHotCategorical(
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
  d = tf.dimension_value(y.shape[-1])
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
      convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
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
        # pylint: disable=g-long-lambda
        lambda t: CategoricalMixtureOfOneHotCategorical.new(
            t, event_size, num_components, sample_dtype, validate_args),
        convert_to_tensor_fn,
        **kwargs)

  @staticmethod
  def new(params, event_size, num_components,
          dtype=None, validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.name_scope(name or 'CategoricalMixtureOfOneHotCategorical'):
      params = tf.convert_to_tensor(params, name='params')
      dist = MixtureSameFamily.new(
          params,
          num_components,
          OneHotCategorical(
              event_size,
              validate_args=False,  # So we can eval on simplex interior.
              dtype=params.dtype,
              name=name),
          validate_args=validate_args,
          name=name)
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
        name or 'CategoricalMixtureOfOneHotCategorical_params_size'):
      return MixtureSameFamily.params_size(
          num_components,
          OneHotCategorical.params_size(event_size, name=name),
          name=name)


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
               event_shape=(),
               convert_to_tensor_fn=tfd.Distribution.sample,
               sample_dtype=None,
               validate_args=False,
               **kwargs):
    """Initialize the `IndependentBernoulli` layer.

    Args:
      event_shape: integer vector `Tensor` representing the shape of single
        draw from this distribution.
      convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
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
    convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

    # If there is a 'make_distribution_fn' keyword argument (e.g., because we
    # are being called from a `from_config` method), remove it.  We pass the
    # distribution function to `DistributionLambda.__init__` below as the first
    # positional argument.
    kwargs.pop('make_distribution_fn', None)

    super(IndependentBernoulli, self).__init__(
        lambda t: IndependentBernoulli.new(  # pylint: disable=g-long-lambda
            t, event_shape, sample_dtype, validate_args),
        convert_to_tensor_fn,
        **kwargs)

    self._event_shape = event_shape
    self._convert_to_tensor_fn = convert_to_tensor_fn
    self._sample_dtype = sample_dtype
    self._validate_args = validate_args

  @staticmethod
  def new(params, event_shape=(), dtype=None, validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.name_scope(name or 'IndependentBernoulli'):
      params = tf.convert_to_tensor(params, name='params')
      event_shape = dist_util.expand_to_vector(
          tf.convert_to_tensor(
              event_shape, name='event_shape', dtype_hint=tf.int32),
          tensor_name='event_shape')
      new_shape = tf.concat([
          tf.shape(params)[:-1],
          event_shape,
      ], axis=0)
      dist = independent_lib.Independent(
          bernoulli_lib.Bernoulli(
              logits=tf.reshape(params, new_shape),
              dtype=dtype or params.dtype.base_dtype,
              validate_args=validate_args),
          reinterpreted_batch_ndims=tf.size(event_shape),
          validate_args=validate_args)
      dist._logits = dist.distribution._logits  # pylint: disable=protected-access
      dist._probs = dist.distribution._probs  # pylint: disable=protected-access
      dist.logits = bernoulli_lib.Bernoulli.logits
      dist.probs = bernoulli_lib.Bernoulli.probs
      return dist

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.name_scope(name or 'IndependentBernoulli_params_size'):
      event_shape = tf.convert_to_tensor(
          event_shape, name='event_shape', dtype_hint=tf.int32)
      return _event_size(
          event_shape, name=name or 'IndependentBernoulli_params_size')

  def get_config(self):
    """Returns the config of this layer.

    NOTE: At the moment, this configuration can only be serialized if the
    Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
    implements `get_config`) or one of the standard values:
     - `Distribution.sample` (or `"sample"`)
     - `Distribution.mean` (or `"mean"`)
     - `Distribution.mode` (or `"mode"`)
     - `Distribution.stddev` (or `"stddev"`)
     - `Distribution.variance` (or `"variance"`)
    """
    config = {
        'event_shape': self._event_shape,
        'convert_to_tensor_fn': _serialize(self._convert_to_tensor_fn),
        'sample_dtype': self._sample_dtype,
        'validate_args': self._validate_args
    }
    base_config = super(IndependentBernoulli, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def _eval_all_one_hot(fn, dist, name=None):
  """OneHotCategorical helper computing probs, cdf, etc over its support."""
  with tf.name_scope(name or 'eval_all_one_hot'):
    event_size = dist.event_shape_tensor()[-1]
    batch_ndims = tf.size(dist.batch_shape_tensor())
    # Reshape `eye(d)` to: `[d] + [1]*batch_ndims + [d]`.
    x = tf.reshape(
        tf.eye(event_size, dtype=dist.dtype),
        shape=tf.pad(
            tf.ones(batch_ndims, tf.int32),
            paddings=[[1, 1]],
            constant_values=event_size))
    # Compute `fn(x)` then cyclically left-transpose one dim.
    perm = tf.pad(tf.range(1, batch_ndims + 1), paddings=[[0, 1]])
    return tf.transpose(a=fn(dist, x), perm=perm)


class IndependentLogistic(DistributionLambda):
  """An independent logistic Keras layer.

  ### Example

  ```python
  tfd = tfp.distributions
  tfpl = tfp.layers
  tfk = tf.keras
  tfkl = tf.keras.layers

  # Create a stochastic encoder -- e.g., for use in a variational auto-encoder.
  input_shape = [28, 28, 1]
  encoded_shape = 2
  encoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=input_shape),
    tfkl.Flatten(),
    tfkl.Dense(10, activation='relu'),
    tfkl.Dense(tfpl.IndependentLogistic.params_size(encoded_shape)),
    tfpl.IndependentLogistic(encoded_shape)
  ])
  ```

  """

  def __init__(self,
               event_shape=(),
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    """Initialize the `IndependentLogistic` layer.

    Args:
      event_shape: integer vector `Tensor` representing the shape of single
        draw from this distribution.
      convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
        instance and returns a `tf.Tensor`-like object.
        Default value: `tfd.Distribution.sample`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
    """
    convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

    # If there is a 'make_distribution_fn' keyword argument (e.g., because we
    # are being called from a `from_config` method), remove it.  We pass the
    # distribution function to `DistributionLambda.__init__` below as the first
    # positional argument.
    kwargs.pop('make_distribution_fn', None)

    super(IndependentLogistic, self).__init__(
        lambda t: IndependentLogistic.new(t, event_shape, validate_args),
        convert_to_tensor_fn,
        **kwargs)

    self._event_shape = event_shape
    self._convert_to_tensor_fn = convert_to_tensor_fn
    self._validate_args = validate_args

  @staticmethod
  def new(params, event_shape=(), validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.name_scope(name or 'IndependentLogistic'):
      params = tf.convert_to_tensor(params, name='params')
      event_shape = dist_util.expand_to_vector(
          tf.convert_to_tensor(
              event_shape, name='event_shape', dtype_hint=tf.int32),
          tensor_name='event_shape')
      output_shape = tf.concat([
          tf.shape(params)[:-1],
          event_shape,
      ],
                               axis=0)
      loc_params, scale_params = tf.split(params, 2, axis=-1)
      return independent_lib.Independent(
          logistic_lib.Logistic(
              loc=tf.reshape(loc_params, output_shape),
              scale=tf.math.softplus(tf.reshape(scale_params, output_shape)),
              validate_args=validate_args),
          reinterpreted_batch_ndims=tf.size(event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.name_scope(name or 'IndependentLogistic_params_size'):
      event_shape = tf.convert_to_tensor(
          event_shape, name='event_shape', dtype_hint=tf.int32)
      return np.int32(2) * _event_size(
          event_shape, name=name or 'IndependentLogistic_params_size')

  def get_config(self):
    """Returns the config of this layer.

    NOTE: At the moment, this configuration can only be serialized if the
    Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
    implements `get_config`) or one of the standard values:
     - `Distribution.sample` (or `"sample"`)
     - `Distribution.mean` (or `"mean"`)
     - `Distribution.mode` (or `"mode"`)
     - `Distribution.stddev` (or `"stddev"`)
     - `Distribution.variance` (or `"variance"`)
    """
    config = {
        'event_shape': self._event_shape,
        'convert_to_tensor_fn': _serialize(self._convert_to_tensor_fn),
        'validate_args': self._validate_args
    }
    base_config = super(IndependentLogistic, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class IndependentNormal(DistributionLambda):
  """An independent normal Keras layer.

  ### Example

  ```python
  tfd = tfp.distributions
  tfpl = tfp.layers
  tfk = tf.keras
  tfkl = tf.keras.layers

  # Create a stochastic encoder -- e.g., for use in a variational auto-encoder.
  input_shape = [28, 28, 1]
  encoded_shape = 2
  encoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=input_shape),
    tfkl.Flatten(),
    tfkl.Dense(10, activation='relu'),
    tfkl.Dense(tfpl.IndependentNormal.params_size(encoded_shape)),
    tfpl.IndependentNormal(encoded_shape)
  ])
  ```

  """

  def __init__(self,
               event_shape=(),
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    """Initialize the `IndependentNormal` layer.

    Args:
      event_shape: integer vector `Tensor` representing the shape of single
        draw from this distribution.
      convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
        instance and returns a `tf.Tensor`-like object.
        Default value: `tfd.Distribution.sample`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
    """
    convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

    # If there is a 'make_distribution_fn' keyword argument (e.g., because we
    # are being called from a `from_config` method), remove it.  We pass the
    # distribution function to `DistributionLambda.__init__` below as the first
    # positional argument.
    kwargs.pop('make_distribution_fn', None)

    super(IndependentNormal, self).__init__(
        lambda t: IndependentNormal.new(t, event_shape, validate_args),
        convert_to_tensor_fn,
        **kwargs)

    self._event_shape = event_shape
    self._convert_to_tensor_fn = convert_to_tensor_fn
    self._validate_args = validate_args

  @staticmethod
  def new(params, event_shape=(), validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.name_scope(name or 'IndependentNormal'):
      params = tf.convert_to_tensor(params, name='params')
      event_shape = dist_util.expand_to_vector(
          tf.convert_to_tensor(
              event_shape, name='event_shape', dtype_hint=tf.int32),
          tensor_name='event_shape')
      output_shape = tf.concat([
          tf.shape(params)[:-1],
          event_shape,
      ],
                               axis=0)
      loc_params, scale_params = tf.split(params, 2, axis=-1)
      return independent_lib.Independent(
          normal_lib.Normal(
              loc=tf.reshape(loc_params, output_shape),
              scale=tf.math.softplus(tf.reshape(scale_params, output_shape)),
              validate_args=validate_args),
          reinterpreted_batch_ndims=tf.size(event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.name_scope(name or 'IndependentNormal_params_size'):
      event_shape = tf.convert_to_tensor(
          event_shape, name='event_shape', dtype_hint=tf.int32)
      return np.int32(2) * _event_size(
          event_shape, name=name or 'IndependentNormal_params_size')

  def get_config(self):
    """Returns the config of this layer.

    NOTE: At the moment, this configuration can only be serialized if the
    Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
    implements `get_config`) or one of the standard values:
     - `Distribution.sample` (or `"sample"`)
     - `Distribution.mean` (or `"mean"`)
     - `Distribution.mode` (or `"mode"`)
     - `Distribution.stddev` (or `"stddev"`)
     - `Distribution.variance` (or `"variance"`)
    """
    config = {
        'event_shape': self._event_shape,
        'convert_to_tensor_fn': _serialize(self._convert_to_tensor_fn),
        'validate_args': self._validate_args
    }
    base_config = super(IndependentNormal, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class IndependentPoisson(DistributionLambda):
  """An independent Poisson Keras layer.

  ### Example

  ```python
  tfd = tfp.distributions
  tfpl = tfp.layers
  tfk = tf.keras
  tfkl = tf.keras.layers

  # Create example data.
  n = 2000
  d = 4
  x = tfd.Uniform(low=1., high=10.).sample([n, d])
  w = [[3.14], [2.72], [-1.62], [0.577]]
  log_rate = tf.matmul(x, w) - 0.141
  y = tfd.Poisson(log_rate=log_rate).sample()

  # Poisson regression model.
  model = tfk.Sequential([
      tfkl.Dense(tfpl.IndependentPoisson.params_size(1)),
      tfpl.IndependentPoisson(1)
  ])

  # Fit.
  model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.05),
                loss=lambda y, model: -model.log_prob(y),
                metrics=[])
  batch_size = 50
  model.fit(x, y,
            batch_size=batch_size,
            epochs=20,
            steps_per_epoch=n // batch_size,
            verbose=True,
            shuffle=True)
  print(model.get_weights())
  ```

  """

  def __init__(self,
               event_shape=(),
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    """Initialize the `IndependentPoisson` layer.

    Args:
      event_shape: integer vector `Tensor` representing the shape of single
        draw from this distribution.
      convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
        instance and returns a `tf.Tensor`-like object.
        Default value: `tfd.Distribution.sample`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
    """
    convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

    # If there is a 'make_distribution_fn' keyword argument (e.g., because we
    # are being called from a `from_config` method), remove it.  We pass the
    # distribution function to `DistributionLambda.__init__` below as the first
    # positional argument.
    kwargs.pop('make_distribution_fn', None)

    super(IndependentPoisson, self).__init__(
        lambda t: IndependentPoisson.new(t, event_shape, validate_args),
        convert_to_tensor_fn,
        **kwargs)

    self._event_shape = event_shape
    self._convert_to_tensor_fn = convert_to_tensor_fn
    self._validate_args = validate_args

  @staticmethod
  def new(params, event_shape=(), validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.name_scope(name or 'IndependentPoisson'):
      params = tf.convert_to_tensor(params, name='params')
      event_shape = dist_util.expand_to_vector(
          tf.convert_to_tensor(
              event_shape, name='event_shape', dtype_hint=tf.int32),
          tensor_name='event_shape')
      output_shape = tf.concat([
          tf.shape(params)[:-1],
          event_shape,
      ],
                               axis=0)
      return independent_lib.Independent(
          poisson_lib.Poisson(
              log_rate=tf.reshape(params, output_shape),
              validate_args=validate_args),
          reinterpreted_batch_ndims=tf.size(event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.name_scope(name or 'IndependentPoisson_params_size'):
      event_shape = tf.convert_to_tensor(
          event_shape, name='event_shape', dtype_hint=tf.int32)
      return _event_size(
          event_shape, name=name or 'IndependentPoisson_params_size')

  def get_config(self):
    """Returns the config of this layer.

    NOTE: At the moment, this configuration can only be serialized if the
    Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
    implements `get_config`) or one of the standard values:
     - `Distribution.sample` (or `"sample"`)
     - `Distribution.mean` (or `"mean"`)
     - `Distribution.mode` (or `"mode"`)
     - `Distribution.stddev` (or `"stddev"`)
     - `Distribution.variance` (or `"variance"`)
    """
    config = {
        'event_shape': self._event_shape,
        'convert_to_tensor_fn': _serialize(self._convert_to_tensor_fn),
        'validate_args': self._validate_args
    }
    base_config = super(IndependentPoisson, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


# We mix-in `tf.Module` since Keras `Regularizer` base class tracks neither
# tf.Variables nor tf.Modules.
class KLDivergenceRegularizer(tf.keras.regularizers.Regularizer, tf.Module):
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
    # Note: Any argument which might be a `tf.Variable` needs to be assigned to
    # self in order to activate automatic tracking. We expect this is necessary
    # for at least {weight, distribution_b}.
    self._distribution_b = distribution_b
    self._use_exact_kl = use_exact_kl
    self._test_points_reduce_axis = test_points_reduce_axis
    self._test_points_fn = test_points_fn
    self._weight = weight
    self._kl_divergence_fn = _make_kl_divergence_fn(
        distribution_b,
        use_exact_kl=use_exact_kl,
        test_points_reduce_axis=test_points_reduce_axis,
        test_points_fn=test_points_fn,
        weight=weight)

  @property
  def distribution_b(self):
    return self._distribution_b

  @property
  def use_exact_kl(self):
    return self._use_exact_kl

  @property
  def test_points_reduce_axis(self):
    return self._test_points_reduce_axis

  @property
  def test_points_fn(self):
    return self._test_points_fn

  @property
  def weight(self):
    return self._weight

  def __call__(self, distribution_a):
    # TODO(b/126056144): Remove reacquisition of distribution handle once we
    # identify how/why Keras lost it.
    distribution_a = getattr(
        distribution_a, '_tfp_distribution', distribution_a)
    return self._kl_divergence_fn(distribution_a)


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
    self._regularizer = KLDivergenceRegularizer(
        distribution_b,
        use_exact_kl=use_exact_kl,
        test_points_reduce_axis=test_points_reduce_axis,
        test_points_fn=test_points_fn,
        weight=weight)
    # Make sure to explicitly make a reference for variable tracking. (Keras
    # does not track `tf.Module` tracked variables, only its own.)
    # We need to ensure Keras tracks variables (eg, from activity regularizers
    # for type-II MLE).  Note that we track all variables to support the user
    # pattern: `v.initializer for v in model.variable]`.
    self._extra_variables = self._regularizer.variables

  def call(self, distribution_a):
    self.add_loss(self._regularizer(distribution_a), inputs=[distribution_a])
    return distribution_a


def _make_kl_divergence_fn(
    distribution_b,
    use_exact_kl=False,
    test_points_reduce_axis=(),  # `None` == "all"; () == "none".
    test_points_fn=tf.convert_to_tensor,
    weight=None):
  """Creates a callable computing `KL[a,b]` from `a`, a `tfd.Distribution`."""

  if use_exact_kl:
    kl_divergence_fn = kl_lib.kl_divergence
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
      if isinstance(distribution_b, tf.keras.Model):
        distribution_b_ = distribution_b(0.)  # Pass a dummy arg.
      elif callable(distribution_b):
        # TODO(b/119756336): Due to eager/graph Jacobian graph caching bug we
        # add here the capability for deferred construction of the prior.  This
        # capability can probably be removed once b/119756336 is resolved.
        distribution_b_ = distribution_b()
      else:
        distribution_b_ = distribution_b
      kl = kl_divergence_fn(distribution_a, distribution_b_)
      if weight is not None:
        kl = tf.cast(weight, dtype=kl.dtype) * kl
      # Losses appended with the model.add_loss and are expected to be a single
      # scalar, unlike model.loss, which is expected to be the loss per sample.
      # Therefore, we reduce over all dimensions, regardless of the shape.
      # We take the sum because (apparently) Keras will add this to the *post*
      # `reduce_sum` (total) loss.
      # TODO(b/126259176): Add end-to-end Keras/TFP test to ensure the API's
      # align, particularly wrt how losses are aggregated (across batch
      # members).
      return tf.reduce_sum(kl, name='batch_total_kl_divergence')

  return _fn


class MixtureSameFamily(DistributionLambda):
  """A mixture (same-family) Keras layer.

  ### Example

  ```python
  tfd = tfp.distributions
  tfpl = tfp.layers
  tfk = tf.keras
  tfkl = tf.keras.layers

  # Load data -- graph of a [cardioid](https://en.wikipedia.org/wiki/Cardioid).
  n = 2000
  t = tfd.Uniform(low=-np.pi, high=np.pi).sample([n, 1])
  r = 2 * (1 - tf.cos(t))
  x = r * tf.sin(t) + tfd.Normal(loc=0., scale=0.1).sample([n, 1])
  y = r * tf.cos(t) + tfd.Normal(loc=0., scale=0.1).sample([n, 1])

  # Model the distribution of y given x with a Mixture Density Network.
  event_shape = [1]
  num_components = 5
  params_size = tfpl.MixtureSameFamily.params_size(
      num_components,
      component_params_size=tfpl.IndependentNormal.params_size(event_shape))
  model = tfk.Sequential([
    tfkl.Dense(12, activation='relu'),
    tfkl.Dense(params_size, activation=None),
    tfpl.MixtureSameFamily(num_components, tfpl.IndependentNormal(event_shape)),
  ])

  # Fit.
  batch_size = 100
  model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.02),
                loss=lambda y, model: -model.log_prob(y))
  model.fit(x, y,
            batch_size=batch_size,
            epochs=20,
            steps_per_epoch=n // batch_size)
  ```

  """

  def __init__(self,
               num_components,
               component_layer,
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    """Initialize the `MixtureSameFamily` distribution layer.

    Args:
      num_components: Number of component distributions in the mixture
        distribution.
      component_layer: Python `callable` that, given a tensor of shape
        `batch_shape + [num_components, component_params_size]`, returns a
        `tfd.Distribution`-like instance that implements the component
        distribution (with batch shape `batch_shape + [num_components]`) --
        e.g., a TFP distribution layer.
      convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
        instance and returns a `tf.Tensor`-like object.
        Default value: `tfd.Distribution.sample`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
    """
    super(MixtureSameFamily, self).__init__(
        lambda t: MixtureSameFamily.new(  # pylint: disable=g-long-lambda
            t, num_components, component_layer, validate_args),
        convert_to_tensor_fn,
        **kwargs)

  @staticmethod
  def new(params, num_components, component_layer,
          validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.name_scope(name or 'MixtureSameFamily'):
      params = tf.convert_to_tensor(params, name='params')
      num_components = tf.convert_to_tensor(
          num_components, name='num_components', dtype_hint=tf.int32)

      components_dist = component_layer(
          tf.reshape(
              params[..., num_components:],
              tf.concat([tf.shape(params)[:-1], [num_components, -1]],
                        axis=0)))
      mixture_dist = categorical_lib.Categorical(
          logits=params[..., :num_components])
      return mixture_same_family_lib.MixtureSameFamily(
          mixture_dist,
          components_dist,
          # TODO(b/120154797): Change following to `validate_args=True` after
          # fixing: "ValueError: `mixture_distribution` must have scalar
          # `event_dim`s." assertion in MixtureSameFamily.
          validate_args=False)

  @staticmethod
  def params_size(num_components, component_params_size, name=None):
    """Number of `params` needed to create a `MixtureSameFamily` distribution.

    Args:
      num_components: Number of component distributions in the mixture
        distribution.
      component_params_size: Number of parameters needed to create a single
        component distribution.
      name: The name to use for the op to compute the number of parameters
        (if such an op needs to be created).

    Returns:
     params_size: The number of parameters needed to create the mixture
       distribution.
    """
    with tf.name_scope(name or 'MixtureSameFamily_params_size'):
      num_components = tf.convert_to_tensor(
          num_components, name='num_components', dtype_hint=tf.int32)
      component_params_size = tf.convert_to_tensor(
          component_params_size, name='component_params_size')

      num_components = dist_util.prefer_static_value(num_components)
      component_params_size = dist_util.prefer_static_value(
          component_params_size)

      return num_components * (1 + component_params_size)


class MixtureNormal(DistributionLambda):
  """A mixture distribution Keras layer, with independent normal components.

  ### Example

  ```python
  tfd = tfp.distributions
  tfpl = tfp.layers
  tfk = tf.keras
  tfkl = tf.keras.layers

  # Load data -- graph of a [cardioid](https://en.wikipedia.org/wiki/Cardioid).
  n = 2000
  t = tfd.Uniform(low=-np.pi, high=np.pi).sample([n, 1])
  r = 2 * (1 - tf.cos(t))
  x = r * tf.sin(t) + tfd.Normal(loc=0., scale=0.1).sample([n, 1])
  y = r * tf.cos(t) + tfd.Normal(loc=0., scale=0.1).sample([n, 1])

  # Model the distribution of y given x with a Mixture Density Network.
  event_shape = [1]
  num_components = 5
  params_size = tfpl.MixtureNormal.params_size(num_components, event_shape)
  model = tfk.Sequential([
    tfkl.Dense(12, activation='relu'),
    tfkl.Dense(params_size, activation=None),
    tfpl.MixtureNormal(num_components, event_shape)
  ])

  # Fit.
  batch_size = 100
  model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.02),
                loss=lambda y, model: -model.log_prob(y))
  model.fit(x, y,
            batch_size=batch_size,
            epochs=20,
            steps_per_epoch=n // batch_size)
  ```

  """

  def __init__(self,
               num_components,
               event_shape=(),
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    """Initialize the `MixtureNormal` distribution layer.

    Args:
      num_components: Number of component distributions in the mixture
        distribution.
      event_shape: integer vector `Tensor` representing the shape of single
        draw from this distribution.
      convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
        instance and returns a `tf.Tensor`-like object.
        Default value: `tfd.Distribution.sample`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
    """
    convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

    # If there is a 'make_distribution_fn' keyword argument (e.g., because we
    # are being called from a `from_config` method), remove it.  We pass the
    # distribution function to `DistributionLambda.__init__` below as the first
    # positional argument.
    kwargs.pop('make_distribution_fn', None)

    super(MixtureNormal, self).__init__(
        lambda t: MixtureNormal.new(  # pylint: disable=g-long-lambda
            t, num_components, event_shape, validate_args),
        convert_to_tensor_fn,
        **kwargs)

    self._num_components = num_components
    self._event_shape = event_shape
    self._convert_to_tensor_fn = convert_to_tensor_fn
    self._validate_args = validate_args

  @staticmethod
  def new(params, num_components, event_shape=(),
          validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(params, name='params')
    return MixtureSameFamily.new(
        params,
        num_components,
        IndependentNormal(event_shape, validate_args=validate_args,
                          dtype=params.dtype, name=name),
        validate_args=validate_args,
        name=name)

  @staticmethod
  def params_size(num_components, event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    return MixtureSameFamily.params_size(
        num_components,
        IndependentNormal.params_size(event_shape, name=name),
        name=name)

  def get_config(self):
    """Returns the config of this layer.

    NOTE: At the moment, this configuration can only be serialized if the
    Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
    implements `get_config`) or one of the standard values:
     - `Distribution.sample` (or `"sample"`)
     - `Distribution.mean` (or `"mean"`)
     - `Distribution.mode` (or `"mode"`)
     - `Distribution.stddev` (or `"stddev"`)
     - `Distribution.variance` (or `"variance"`)
    """
    config = {
        'num_components': self._num_components,
        'event_shape': self._event_shape,
        'convert_to_tensor_fn': _serialize(self._convert_to_tensor_fn),
        'validate_args': self._validate_args
    }
    base_config = super(MixtureNormal, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class MixtureLogistic(DistributionLambda):
  """A mixture distribution Keras layer, with independent logistic components.

  ### Example

  ```python
  tfd = tfp.distributions
  tfpl = tfp.layers
  tfk = tf.keras
  tfkl = tf.keras.layers

  # Load data -- graph of a [cardioid](https://en.wikipedia.org/wiki/Cardioid).
  n = 2000
  t = tfd.Uniform(low=-np.pi, high=np.pi).sample([n, 1])
  r = 2 * (1 - tf.cos(t))
  x = r * tf.sin(t) + tfd.Normal(loc=0., scale=0.1).sample([n, 1])
  y = r * tf.cos(t) + tfd.Normal(loc=0., scale=0.1).sample([n, 1])

  # Model the distribution of y given x with a Mixture Density Network.
  event_shape = [1]
  num_components = 5
  params_size = tfpl.MixtureLogistic.params_size(num_components, event_shape)
  model = tfk.Sequential([
    tfkl.Dense(12, activation='relu'),
    tfkl.Dense(params_size, activation=None),
    tfpl.MixtureLogistic(num_components, event_shape)
  ])

  # Fit.
  batch_size = 100
  model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.02),
                loss=lambda y, model: -model.log_prob(y))
  model.fit(x, y,
            batch_size=batch_size,
            epochs=20,
            steps_per_epoch=n // batch_size)
  ```

  """

  def __init__(self,
               num_components,
               event_shape=(),
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    """Initialize the `MixtureLogistic` distribution layer.

    Args:
      num_components: Number of component distributions in the mixture
        distribution.
      event_shape: integer vector `Tensor` representing the shape of single
        draw from this distribution.
      convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
        instance and returns a `tf.Tensor`-like object.
        Default value: `tfd.Distribution.sample`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
    """
    convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

    # If there is a 'make_distribution_fn' keyword argument (e.g., because we
    # are being called from a `from_config` method), remove it.  We pass the
    # distribution function to `DistributionLambda.__init__` below as the first
    # positional argument.
    kwargs.pop('make_distribution_fn', None)

    super(MixtureLogistic, self).__init__(
        lambda t: MixtureLogistic.new(  # pylint: disable=g-long-lambda
            t, num_components, event_shape, validate_args),
        convert_to_tensor_fn,
        **kwargs)

    self._num_components = num_components
    self._event_shape = event_shape
    self._convert_to_tensor_fn = convert_to_tensor_fn
    self._validate_args = validate_args

  @staticmethod
  def new(params, num_components, event_shape=(),
          validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(params, name='params')
    return MixtureSameFamily.new(
        params,
        num_components,
        IndependentLogistic(
            event_shape, validate_args=validate_args, dtype=params.dtype,
            name=name),
        validate_args=validate_args,
        name=name)

  @staticmethod
  def params_size(num_components, event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    return MixtureSameFamily.params_size(
        num_components,
        IndependentLogistic.params_size(event_shape, name=name),
        name=name)

  def get_config(self):
    """Returns the config of this layer.

    NOTE: At the moment, this configuration can only be serialized if the
    Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
    implements `get_config`) or one of the standard values:
     - `Distribution.sample` (or `"sample"`)
     - `Distribution.mean` (or `"mean"`)
     - `Distribution.mode` (or `"mode"`)
     - `Distribution.stddev` (or `"stddev"`)
     - `Distribution.variance` (or `"variance"`)
    """
    config = {
        'num_components': self._num_components,
        'event_shape': self._event_shape,
        'convert_to_tensor_fn': _serialize(self._convert_to_tensor_fn),
        'validate_args': self._validate_args
    }
    base_config = super(MixtureLogistic, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class VariationalGaussianProcess(DistributionLambda):
  """A VariationalGaussianProcess Layer.

  Create a VariationalGaussianProcess distribtuion whose `index_points` are the
  inputs to the layer. Parameterized by number of inducing points and a
  `kernel_provider`, which should be a `tf.keras.Layer` with an @property that
  late-binds variable parameters to a
  `tfp.positive_semidefinite_kernel.PositiveSemidefiniteKernel` instance (this
  requirement has to do with the way that variables must be created in a keras
  model). The `mean_fn` is an optional argument which, if omitted, will be
  automatically configured to be a constant function with trainable variable
  output.
  """

  def __init__(
      self,
      num_inducing_points,
      kernel_provider,
      event_shape=(1,),
      inducing_index_points_initializer=None,
      unconstrained_observation_noise_variance_initializer=(
          tf.initializers.constant(-10.)),
      variational_inducing_observations_scale_initializer=None,
      mean_fn=None,
      jitter=1e-6,
      convert_to_tensor_fn=tfd.Distribution.sample,
      name=None):
    """Construct a VariationalGaussianProcess Layer.

    Args:
      num_inducing_points: number of inducing points in the
        VariationalGaussianProcess distribution.
      kernel_provider: a `Layer` instance equipped with an @property, which
        yields a `PositiveSemidefiniteKernel` instance. The latter is used to
        parameterize the constructed VariationalGaussianProcess distribution
        returned by calling the layer.
      event_shape: the shape of the output of the layer. This translates to a
        batch of underlying VariationalGaussianProcess distribtuions. For
        example, `event_shape = [3]` means we are modeling a batch of 3
        distributions over functions. We can think of this as a distrbution over
        3-dimensional vector-valued functions.
      inducing_index_points_initializer: a `tf.keras.initializer.Initializer`
        used to initialize the trainable `inducing_index_points` variables.
        Training VGP's is pretty sensitive to choice of initial inducing index
        point locations. A reasonable heuristic is to scatter them near the
        data, not too close to each other.
      unconstrained_observation_noise_variance_initializer: a
        `tf.keras.initializer.Initializer` used to initialize the unconstrained
        observation noise variable. The observation noise variance is computed
        from this variable via the `tf.nn.softplus` function.
      variational_inducing_observations_scale_initializer: a
        `tf.keras.initializer.Initializer` used to initialize the variational
        inducing observations scale.
      mean_fn: a callable that maps layer inputs to mean function values. Passed
        to the mean_fn parameter of VariationalGaussianProcess distribution. If
        omitted, defaults to a constant function with trainable variable value.
      jitter: a small term added to the diagonal of various kernel matrices for
        numerical stability.
      convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
        instance and returns a `tf.Tensor`-like object. For examples, see
        `class` docstring.
      name: name to give to this layer and the scope of ops and variables it
        contains.
    """
    convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

    super(VariationalGaussianProcess, self).__init__(
        make_distribution_fn=lambda x: VariationalGaussianProcess.new(  # pylint: disable=g-long-lambda
            x,
            kernel_provider=self._kernel_provider,
            event_shape=self._event_shape,
            inducing_index_points=self._inducing_index_points,
            variational_inducing_observations_loc=(
                self._variational_inducing_observations_loc),
            variational_inducing_observations_scale=(
                self._variational_inducing_observations_scale),
            mean_fn=self._mean_fn,
            observation_noise_variance=(
                self._unconstrained_observation_noise_variance),
            jitter=self._jitter),
        convert_to_tensor_fn=convert_to_tensor_fn,
        dtype=kernel_provider.dtype,
        name=name)

    tmp_kernel = kernel_provider.kernel
    self._dtype = tmp_kernel.dtype.as_numpy_dtype
    self._feature_ndims = tmp_kernel.feature_ndims
    self._num_inducing_points = num_inducing_points
    self._event_shape = tf.TensorShape(event_shape)
    self._mean_fn = mean_fn
    self._jitter = jitter
    self._inducing_index_points_initializer = inducing_index_points_initializer
    self._unconstrained_observation_noise_variance_initializer = (
        unconstrained_observation_noise_variance_initializer)
    self._variational_inducing_observations_scale_initializer = (
        variational_inducing_observations_scale_initializer)
    self._kernel_provider = kernel_provider

  def build(self, input_shape):
    input_feature_shape = input_shape[-self._feature_ndims:]

    inducing_index_points_shape = (
        self._event_shape.as_list() +
        [self._num_inducing_points] +
        input_feature_shape.as_list())

    if self._mean_fn is None:
      self.mean = self.add_weight(
          initializer=tf.initializers.constant([0.]),
          dtype=self._dtype,
          name='mean')
      self._mean_fn = lambda x: self.mean

    # This matches the default value of observation_noise_variance in the
    # VariationalGaussianProcess distribution itself. As that logic changes
    # (see b/136668249) we'll need to keep this in sync.
    self._unconstrained_observation_noise_variance = 0.
    if self._unconstrained_observation_noise_variance_initializer is not None:
      unconstrained_observation_noise_variance = self.add_weight(
          initializer=(
              self._unconstrained_observation_noise_variance_initializer),
          dtype=self._dtype,
          name='observation_noise_variance')
      self._unconstrained_observation_noise_variance = tfp_util.DeferredTensor(
          unconstrained_observation_noise_variance, tf.nn.softplus)

    self._inducing_index_points = self.add_weight(
        name='inducing_index_points',
        shape=inducing_index_points_shape,
        initializer=self._inducing_index_points_initializer,
        dtype=self._dtype)

    self._variational_inducing_observations_loc = self.add_weight(
        name='variational_inducing_observations_loc',
        shape=self._event_shape.as_list() + [self._num_inducing_points],
        initializer=tf.initializers.zeros(),
        dtype=self._dtype)

    if self._variational_inducing_observations_scale_initializer is None:
      eyes = (np.ones(self._event_shape.as_list() + [1, 1]) *
              np.eye(self._num_inducing_points, dtype=self._dtype))
      self._variational_inducing_observations_scale_initializer = (
          tf.initializers.constant(1e-5 * eyes))
    self._variational_inducing_observations_scale = self.add_weight(
        name='variational_inducing_observations_scale',
        shape=(self._event_shape.as_list() +
               [self._num_inducing_points, self._num_inducing_points]),
        initializer=self._variational_inducing_observations_scale_initializer)

  @staticmethod
  def new(x,
          kernel_provider,
          event_shape,
          inducing_index_points,
          mean_fn,
          variational_inducing_observations_loc,
          variational_inducing_observations_scale,
          observation_noise_variance,
          jitter=1e-6,
          name=None):
    vgp = variational_gaussian_process_lib.VariationalGaussianProcess(
        kernel=kernel_provider.kernel,
        index_points=x,
        inducing_index_points=inducing_index_points,
        variational_inducing_observations_loc=(
            variational_inducing_observations_loc),
        variational_inducing_observations_scale=(
            variational_inducing_observations_scale),
        mean_fn=mean_fn,
        observation_noise_variance=observation_noise_variance,
        jitter=jitter)
    ind = independent_lib.Independent(vgp, reinterpreted_batch_ndims=1)
    bij = transpose_lib.Transpose(rightmost_transposed_ndims=2)
    d = transformed_distribution_lib.TransformedDistribution(ind, bijector=bij)
    def _transposed_variational_loss(y, kl_weight=1.):
      loss = vgp.variational_loss(bij.forward(y), kl_weight=kl_weight)
      return loss
    d.variational_loss = _transposed_variational_loss
    d.surrogate_posterior_kl_divergence_prior = (
        vgp.surrogate_posterior_kl_divergence_prior)
    return d


# For deserialization.
tf.keras.utils.get_custom_objects().update({
    'DistributionLambda': DistributionLambda,
    'IndependentBernoulli': IndependentBernoulli,
    'IndependentLogistic': IndependentLogistic,
    'IndependentNormal': IndependentNormal,
    'IndependentPoisson': IndependentPoisson,
    'MixtureLogistic': MixtureLogistic,
    'MixtureNormal': MixtureNormal,
})

sample = tfd.Distribution.sample
mean = tfd.Distribution.mean
mode = tfd.Distribution.mode
stddev = tfd.Distribution.stddev
variance = tfd.Distribution.variance


def _serialize(convert_to_tensor_fn):
  return tf.keras.utils.serialize_keras_object(convert_to_tensor_fn)


def _deserialize(name, custom_objects=None):
  return tf.keras.utils.deserialize_keras_object(
      name,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='convert-to-tensor function')


def _get_convert_to_tensor_fn(identifier):
  """Return a convert-to-tensor func, given a name, config, callable, etc."""
  if identifier is None:
    return None

  if isinstance(identifier, six.string_types):
    identifier = str(identifier)
    return _deserialize(identifier)

  if isinstance(identifier, dict):
    return _deserialize(identifier)

  if isinstance(identifier, property):
    identifier = identifier.fget
  if callable(identifier):
    return identifier

  raise ValueError('Could not interpret '
                   'convert-to-tensor function identifier:', identifier)


def _reduce_tensor(tensor):
  val = tf.get_static_value(tensor)
  if val is None:
    raise ValueError('Cannot pickle Tensor -- '
                     'its value is not known statically: {}.'.format(tensor))
  return (tf.convert_to_tensor, (val,))


def _serialize_function(func):
  """Serializes a function (using CloudPickle)."""
  buffer = io.BytesIO()
  pickler = CloudPickler(buffer)

  # Serializing a DistributionLambda or other distribution layer may require
  # serializaing a lambda or function that closes over a constant, graph-mode
  # Tensor, but graph-mode Tensors do not support pickling.  We modify
  # `pickler.dispatch_table` so that a special reduction function will be used
  # for graph-mode Tensors, which will:
  #  - Correctly serialize constant graph-mode Tensors.
  #  - Raise an explanatory error message for non-constant graph-mode Tensors.
  if not hasattr(pickler, 'dispatch_table'):
    pickler.dispatch_table = {}
  pickler.dispatch_table[tf.Tensor] = _reduce_tensor

  pickler.dump(func)
  return codecs.encode(buffer.getvalue(), 'base64').decode('ascii')


def _deserialize_function(code):
  raw_code = codecs.decode(code.encode('ascii'), 'base64')
  return pickle.loads(raw_code)
