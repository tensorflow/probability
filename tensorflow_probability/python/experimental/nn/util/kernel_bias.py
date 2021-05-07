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
# ============================================================================
"""Utility functions for building layer kernels and biases."""


import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors.chain import Chain
from tensorflow_probability.python.bijectors.shift import Shift
from tensorflow_probability.python.bijectors.softplus import Softplus
from tensorflow_probability.python.distributions.categorical import Categorical
from tensorflow_probability.python.distributions.independent import Independent
from tensorflow_probability.python.distributions.joint_distribution_sequential import JointDistributionSequential
from tensorflow_probability.python.distributions.mixture_same_family import MixtureSameFamily
from tensorflow_probability.python.distributions.normal import Normal
from tensorflow_probability.python.distributions.sample import Sample
from tensorflow_probability.python.experimental.nn import initializers as nn_init_lib
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.util.deferred_tensor import TransformedVariable


__all__ = [
    'make_kernel_bias',
    'make_kernel_bias_posterior_mvn_diag',
    'make_kernel_bias_prior_spike_and_slab',
]


# make_kernel_bias* functions must all have the same call signature.


def make_kernel_bias(
    kernel_shape,
    bias_shape,
    kernel_initializer=None,
    bias_initializer=None,
    kernel_batch_ndims=0,
    bias_batch_ndims=0,
    dtype=tf.float32,
    kernel_name='kernel',
    bias_name='bias'):
  # pylint: disable=line-too-long
  """Creates kernel and bias as `tf.Variable`s.

  Args:
    kernel_shape: ...
    bias_shape: ...
    kernel_initializer: ...
      Default value: `None` (i.e., `tf.initializers.glorot_uniform()`).
    bias_initializer: ...
      Default value: `None` (i.e., `tf.initializers.zeros()`).
    kernel_batch_ndims: ...
      Default value: `0`.
    bias_batch_ndims: ...
      Default value: `0`.
    dtype: ...
      Default value: `tf.float32`.
    kernel_name: ...
      Default value: `"kernel"`.
    bias_name: ...
      Default value: `"bias"`.

  Returns:
    kernel: ...
    bias: ...

  #### Recomendations:

  ```python
  #   tf.nn.relu    ==> tf.initializers.he_*
  #   tf.nn.elu     ==> tf.initializers.he_*
  #   tf.nn.selu    ==> tf.initializers.lecun_*
  #   tf.nn.tanh    ==> tf.initializers.glorot_*
  #   tf.nn.sigmoid ==> tf.initializers.glorot_*
  #   tf.nn.softmax ==> tf.initializers.glorot_*
  #   None          ==> tf.initializers.glorot_*
  # https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404
  # https://stats.stackexchange.com/a/393012/1835

  def make_uniform(size):
    s = tf.math.rsqrt(size / 3.)
    return tfd.Uniform(low=-s, high=s)

  def make_normal(size):
    # Constant is: `scipy.stats.truncnorm.var(loc=0., scale=1., a=-2., b=2.)`.
    s = tf.math.rsqrt(size) / 0.87962566103423978
    return tfd.TruncatedNormal(loc=0, scale=s, low=-2., high=2.)

  # He.  https://arxiv.org/abs/1502.01852
  he_uniform = make_uniform(fan_in / 2.)
  he_normal  = make_normal (fan_in / 2.)

  # Glorot (aka Xavier). http://proceedings.mlr.press/v9/glorot10a.html
  glorot_uniform = make_uniform((fan_in + fan_out) / 2.)
  glorot_normal  = make_normal ((fan_in + fan_out) / 2.)
  ```

  """
  # pylint: enable=line-too-long
  if kernel_initializer is None:
    kernel_initializer = nn_init_lib.glorot_uniform()
  if bias_initializer is None:
    bias_initializer = tf.initializers.zeros()
  return (
      tf.Variable(_try_call_init_fn(kernel_initializer,
                                    kernel_shape,
                                    dtype,
                                    kernel_batch_ndims),
                  name=kernel_name),
      tf.Variable(_try_call_init_fn(bias_initializer,
                                    bias_shape,
                                    dtype,
                                    bias_batch_ndims),
                  name=bias_name),
  )


def make_kernel_bias_prior_spike_and_slab(
    kernel_shape,
    bias_shape,
    kernel_initializer=None,  # pylint: disable=unused-argument
    bias_initializer=None,  # pylint: disable=unused-argument
    kernel_batch_ndims=0,  # pylint: disable=unused-argument
    bias_batch_ndims=0,  # pylint: disable=unused-argument
    dtype=tf.float32,
    kernel_name='prior_kernel',
    bias_name='prior_bias'):
  """Create prior for Variational layers with kernel and bias.

  Note: Distribution scale is inversely related to regularization strength.
  Consider a "Normal" prior; bigger scale corresponds to less L2 regularization.
  I.e.,
  ```python
  scale    = (2. * l2weight)**-0.5
  l2weight = scale**-2. / 2.
  ```
  have a similar regularizing effect.

  The std. deviation of each of the component distributions returned by this
  function is approximately `1415` (or approximately `l2weight = 25e-6`). In
  other words this prior is extremely "weak".

  Args:
    kernel_shape: ...
    bias_shape: ...
    kernel_initializer: Ignored.
      Default value: `None` (i.e., `tf.initializers.glorot_uniform()`).
    bias_initializer: Ignored.
      Default value: `None` (i.e., `tf.initializers.zeros()`).
    kernel_batch_ndims: ...
      Default value: `0`.
    bias_batch_ndims: ...
      Default value: `0`.
    dtype: ...
      Default value: `tf.float32`.
    kernel_name: ...
      Default value: `"prior_kernel"`.
    bias_name: ...
      Default value: `"prior_bias"`.

  Returns:
    kernel_and_bias_distribution: ...
  """
  w = MixtureSameFamily(
      mixture_distribution=Categorical(probs=[0.5, 0.5]),
      components_distribution=Normal(
          loc=0.,
          scale=tf.constant([1., 2000.], dtype=dtype)))
  return JointDistributionSequential([
      Sample(w, kernel_shape, name=kernel_name),
      Sample(w, bias_shape, name=bias_name),
  ])


def make_kernel_bias_posterior_mvn_diag(
    kernel_shape,
    bias_shape,
    kernel_initializer=None,
    bias_initializer=None,
    kernel_batch_ndims=0,
    bias_batch_ndims=0,  # pylint: disable=unused-argument
    dtype=tf.float32,
    kernel_name='posterior_kernel',
    bias_name='posterior_bias'):
  """Create learnable posterior for Variational layers with kernel and bias.

  Args:
    kernel_shape: ...
    bias_shape: ...
    kernel_initializer: ...
      Default value: `None` (i.e., `tf.initializers.glorot_uniform()`).
    bias_initializer: ...
      Default value: `None` (i.e., `tf.initializers.zeros()`).
    kernel_batch_ndims: ...
      Default value: `0`.
    bias_batch_ndims: ...
      Default value: `0`.
    dtype: ...
      Default value: `tf.float32`.
    kernel_name: ...
      Default value: `"posterior_kernel"`.
    bias_name: ...
      Default value: `"posterior_bias"`.

  Returns:
    kernel_and_bias_distribution: ...
  """
  if kernel_initializer is None:
    kernel_initializer = nn_init_lib.glorot_uniform()
  if bias_initializer is None:
    bias_initializer = tf.initializers.zeros()
  make_loc = lambda init_fn, shape, batch_ndims, name: tf.Variable(  # pylint: disable=g-long-lambda
      _try_call_init_fn(init_fn, shape, dtype, batch_ndims),
      name=name + '_loc')
  # Setting the initial scale to a relatively small value causes the `loc` to
  # quickly move toward a lower loss value.
  make_scale = lambda shape, name: TransformedVariable(  # pylint: disable=g-long-lambda
      tf.fill(shape, value=tf.constant(1e-3, dtype=dtype)),
      Chain([Shift(1e-5), Softplus()]),
      name=name + '_scale')
  return JointDistributionSequential([
      Independent(
          Normal(loc=make_loc(kernel_initializer,
                              kernel_shape,
                              kernel_batch_ndims,
                              kernel_name),
                 scale=make_scale(kernel_shape, kernel_name)),
          reinterpreted_batch_ndims=ps.size(kernel_shape),
          name=kernel_name),
      Independent(
          Normal(loc=make_loc(bias_initializer,
                              bias_shape,
                              kernel_batch_ndims,
                              bias_name),
                 scale=make_scale(bias_shape, bias_name)),
          reinterpreted_batch_ndims=ps.size(bias_shape),
          name=bias_name),
  ])


def _try_call_init_fn(fn, *args):
  """Try to call function with first num_args else num_args - 1."""
  try:
    return fn(*args)
  except TypeError:
    return fn(*args[:-1])

