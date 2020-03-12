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
"""Utilities for initializing neural network layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static


__all__ = [
    'glorot_normal',
    'glorot_uniform',
    'he_normal',
    'he_uniform',
]


def glorot_normal(seed=None):
  """The Glorot normal initializer, aka Xavier normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with standard deviation (after truncation) given by
  `stddev = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number
  of input units in the weight tensor and `fan_out` is the number of
  output units in the weight tensor.

  Args:
    seed: A Python integer. Used to create random seeds.
      Default value: `None`.

  Returns:
    init_fn: A python `callable` which takes a shape `Tensor`, dtype and an
      optional scalar `int` number of batch dims and returns a randomly
      initialized `Tensor` with the specified shape and dtype.

  References:
    [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
    ([pdf](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf))
  """
  return lambda shape, dtype, batch_ndims=0: _initialize(  # pylint: disable=g-long-lambda
      shape, dtype, batch_ndims,
      scale=1., mode='fan_avg', distribution='truncated_normal', seed=seed)


def glorot_uniform(seed=None):
  """The Glorot uniform initializer, aka Xavier uniform initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(6 / (fan_in + fan_out))`
  where `fan_in` is the number of input units in the weight tensor
  and `fan_out` is the number of output units in the weight tensor.

  Args:
    seed: A Python integer. Used to create random seeds.
      Default value: `None`.

  Returns:
    init_fn: A python `callable` which takes a shape `Tensor`, dtype and an
      optional scalar `int` number of batch dims and returns a randomly
      initialized `Tensor` with the specified shape and dtype.

  References:
    [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
    ([pdf](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf))
  """
  return lambda shape, dtype, batch_ndims=0: _initialize(  # pylint: disable=g-long-lambda
      shape, dtype, batch_ndims,
      scale=1., mode='fan_avg', distribution='uniform', seed=seed)


def he_normal(seed=None):
  # pylint: disable=line-too-long
  """He normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with standard deviation (after truncation) given by
  `stddev = sqrt(2 / fan_in)` where `fan_in` is the number of
  input units in the weight tensor.

  Args:
    seed: A Python integer. Used to create random seeds.
      Default value: `None`.

  Returns:
    init_fn: A python `callable` which takes a shape `Tensor`, dtype and an
      optional scalar `int` number of batch dims and returns a randomly
      initialized `Tensor` with the specified shape and dtype.

  References:
    [He et al., 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)
    ([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))
  """
  # pylint: enable=line-too-long
  return lambda shape, dtype, batch_ndims=0: _initialize(  # pylint: disable=g-long-lambda
      shape, dtype, batch_ndims,
      scale=2., mode='fan_in', distribution='truncated_normal', seed=seed)


def he_uniform(seed=None):
  # pylint: disable=line-too-long
  """He uniform variance scaling initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(6 / fan_in)`
  where `fan_in` is the number of input units in the weight tensor.

  Args:
    seed: A Python integer. Used to create random seeds.
      Default value: `None`.

  Returns:
    init_fn: A python `callable` which takes a shape `Tensor`, dtype and an
      optional scalar `int` number of batch dims and returns a randomly
      initialized `Tensor` with the specified shape and dtype.

  References:
    [He et al., 2015]
    (https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)
    ([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))
  """
  # pylint: enable=line-too-long
  return lambda shape, dtype, batch_ndims=0: _initialize(  # pylint: disable=g-long-lambda
      shape, dtype, batch_ndims,
      scale=2., mode='fan_in', distribution='uniform', seed=seed)


def _initialize(shape, dtype, batch_ndims, scale, mode, distribution,
                seed=None):
  """Samples a random `Tensor` per specified args."""
  if not dtype_util.is_floating(dtype):
    raise TypeError('Argument `dtype` must be float type (saw: "{}").'.format(
        dtype))
  shape = prefer_static.reshape(shape, shape=[-1])  # Ensure shape is vector.
  fan_in, fan_out = _compute_fans_from_shape(shape, batch_ndims)
  fans = _summarize_fans(fan_in, fan_out, mode, dtype)
  scale = prefer_static.cast(scale, dtype)
  return _sample_distribution(shape, scale / fans, distribution, seed, dtype)


def _compute_fans_from_shape(shape, batch_ndims=0):
  """Extracts `fan_in, fan_out` from specified shape `Tensor`."""
  # Ensure shape is a vector of length >=2.
  num_pad = prefer_static.maximum(0, 2 - prefer_static.size(shape))
  shape = prefer_static.pad(
      shape, paddings=[[0, num_pad]], constant_values=1)
  (
      batch_shape,  # pylint: disable=unused-variable
      extra_shape,
      fan_in,
      fan_out,
  ) = prefer_static.split(shape, [batch_ndims, -1, 1, 1])
  # The following logic is primarily intended for convolutional layers which
  # have spatial semantics in addition to input/output channels.
  receptive_field_size = prefer_static.reduce_prod(extra_shape)
  fan_in = fan_in[0] * receptive_field_size
  fan_out = fan_out[0] * receptive_field_size
  return fan_in, fan_out


def _summarize_fans(fan_in, fan_out, mode, dtype):
  """Combines `fan_in`, `fan_out` per specified `mode`."""
  fan_in = prefer_static.cast(fan_in, dtype)
  fan_out = prefer_static.cast(fan_out, dtype)
  mode = str(mode).lower()
  if mode == 'fan_in':
    return fan_in
  elif mode == 'fan_out':
    return fan_out
  elif mode == 'fan_avg':
    return (fan_in + fan_out) / 2.
  raise ValueError('Unrecognized mode: "{}".'.format(mode))


def _sample_distribution(shape, var, distribution, seed, dtype):
  """Samples from specified distribution (by appropriately scaling `var` arg."""
  distribution = str(distribution).lower()
  if distribution == 'truncated_normal':
    # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
    stddev = prefer_static.sqrt(var) / 0.87962566103423978
    return tf.random.truncated_normal(shape, 0., stddev, dtype, seed=seed)
  elif distribution == 'uniform':
    limit = prefer_static.sqrt(3. * var)
    return tf.random.uniform(shape, -limit, limit, dtype, seed=seed)
  elif distribution == 'untruncated_normal':
    stddev = prefer_static.sqrt(var)
    return tf.random.normal(shape, 0., stddev, dtype, seed=seed)
  raise ValueError('Unrecognized distribution: "{}".'.format(distribution))
