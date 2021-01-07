# Copyright 2021 The TensorFlow Probability Authors.
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
"""The Unit Normal (Gaussian) distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import special_math


__all__ = [
    'UnitNormal',
]


class UnitNormal(distribution.Distribution):
  """The Unit Normal distribution.

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x) = exp(-0.5 x**2) / Z
  Z = (2 pi)**0.5
  ```

  where `Z` is the normalization constant.

  The Unit Normal distribution is a special case of the Normal distribution
  where `loc = 0` is the mean and `scale = 1` is the std. deviation. The
  implementation is sightly more computationally efficient than `tfd.Normal(0.,
  1.)`.

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Define a single scalar Unit Normal distribution.
  dist = tfd.UnitNormal()

  # Evaluate the cdf at 1, returning a scalar.
  dist.cdf(1.)
  # ==> 0.8413447

  # Define a batch of two scalar valued Unit Normals.
  dist = tfd.UnitNormal(batch_shape=[2])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a shape `[2]` tensor.
  dist.prob([0., 1.5])
  # ==> array([0.3989423 , 0.12951759], dtype=float32)

  # Get 3 samples, returning a shape `[3, 2]` tensor.
  dist.sample(3)
  # ==>
  # array([[ 0.38014176, -0.2981141 ],
  #        [-0.5843418 , -0.02090771],
  #        [ 1.0401396 ,  0.02187739]], dtype=float32)
  ```

  Arguments are broadcast when possible:

  ```python
  # Define a batch of shape `[3, 4]` Unit Normals
  dist = tfd.UnitNormal(batch_shape=[3, 4])

  # Evaluate the pdf of all distributions on the same point, 3.,
  # returning a shape `[3, 4]` tensor.
  dist.prob(3.)
  # ==>
  # array([[0.00443185, 0.00443185, 0.00443185, 0.00443185],
  #        [0.00443185, 0.00443185, 0.00443185, 0.00443185],
  #        [0.00443185, 0.00443185, 0.00443185, 0.00443185]], dtype=float32)

  # Evaluate the pdf of all distributions at different points using 2 samples
  # from each distribution, returning a shape `[2, 3, 4]` tensor:
  dist.prob(dist.sample(2))
  # ==>
  # array([[[0.3220002 , 0.37833017, 0.1470757 , 0.32863948],
  #         [0.39649498, 0.38896585, 0.1496945 , 0.21469072],
  #         [0.35379156, 0.37573174, 0.3285865 , 0.39886284]],
  #
  #        [[0.14576726, 0.35835844, 0.19727986, 0.13787058],
  #         [0.3731404 , 0.02210679, 0.01263283, 0.21340393],
  #         [0.28763762, 0.13159916, 0.33685088, 0.37146214]]], dtype=float32)
  ```
  """

  def __init__(
      self,
      batch_shape=(),
      dtype=tf.float32,
      validate_args=False,
      allow_nan_stats=True,
      name="UnitNormal",
  ):
    """Construct Unit Normal distributions.

    Since this distribution has no parameters, `batch_shape` and `dtype` cannot
    be inferred so they must be supplied by the user.

    Args:
      batch_shape: shape: A 1-D integer Tensor or Python array. The batch shape
        of the distribution. Default is `()` which results in a scalar
        distribution.
      dtype: The `DType` of `Tensor`s handled by this `Distribution`. Default is
        `tf.float32`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or more
        of the statistic's batch members are undefined.  name: Python `str` name
        prefixed to Ops created by this class.
    """

    parameters = dict(locals())
    with tf.name_scope(name) as name:
      self._shape = batch_shape
      super(UnitNormal, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name,
      )

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict()

  @property
  def loc(self):
    """Distribution parameter for the mean."""
    return tf.zeros(shape=self.batch_shape, dtype=self.dtype, name="loc")

  @property
  def scale(self):
    """Distribution parameter for standard deviation."""
    return tf.ones(shape=self.batch_shape, dtype=self.dtype, name="scale")

  def _batch_shape_tensor(self):
    return self._shape

  def _batch_shape(self):
    return self._shape

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    shape = ps.concat([[n], self.batch_shape], axis=0)
    return samplers.normal(shape=shape, dtype=self.dtype, seed=seed)

  def _log_prob(self, x):
    log_unnormalized = -0.5 * tf.math.square(x)
    log_normalization = tf.constant(0.5 * np.log(2. * np.pi), dtype=self.dtype)
    log_prob = log_unnormalized - log_normalization
    return self._broadcast_with_batch_shape(log_prob)

  def _log_cdf(self, x):
    log_cdf = special_math.log_ndtr(x)
    return self._broadcast_with_batch_shape(log_cdf)

  def _cdf(self, x):
    cdf = special_math.ndtr(x)
    return self._broadcast_with_batch_shape(cdf)

  def _log_survival_function(self, x):
    log_survival = special_math.log_ndtr(-x)
    return self._broadcast_with_batch_shape(log_survival)

  def _survival_function(self, x):
    survival = special_math.ndtr(-x)
    return self._broadcast_with_batch_shape(survival)

  def _entropy(self):
    entropy = 0.5 + 0.5 * np.log(2. * np.pi)
    return tf.constant(entropy, dtype=self.dtype, shape=self.batch_shape)

  def _mean(self):
    return self.loc

  def _quantile(self, p):
    quantile = tf.math.ndtri(p)
    return self._broadcast_with_batch_shape(quantile)

  def _stddev(self):
    return self.scale

  _mode = _mean

  def _default_event_space_bijector(self):
    return identity_bijector.Identity(validate_args=self.validate_args)

  def _broadcast_with_batch_shape(self, x):
    shape = ps.broadcast_shape(ps.shape(x), self.batch_shape)
    return ps.broadcast_to(x, shape)


@kullback_leibler.RegisterKL(UnitNormal, UnitNormal)
def _kl_normal_normal(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b UnitNormal.

  Args:
    a: instance of a UnitNormal distribution object.
    b: instance of a UnitNormal distribution object.
    name: Name to use for created operations.
      Default value: `None` (i.e. `'kl_unit_normal_unit_normal'`).

  Returns:
    kl_div: Batchwise KL(a || b)
  """
  with tf.name_scope(name or "kl_unit_normal_unit_normal"):
    dtype = dtype_util.common_dtype([a, b])
    batch_shape = ps.broadcast_shape(a.batch_shape, b.batch_shape)
    return tf.zeros(batch_shape, dtype)
