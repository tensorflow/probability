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
"""Batch Norm bijector."""


# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import tf_keras


__all__ = [
    'BatchNormalization',
]


def _undo_batch_normalization(x,
                              mean,
                              variance,
                              offset,
                              scale,
                              variance_epsilon,
                              name=None):
  r"""Inverse of tf.nn.batch_normalization.

  Args:
    x: Input `Tensor` of arbitrary dimensionality.
    mean: A mean `Tensor`.
    variance: A variance `Tensor`.
    offset: An offset `Tensor`, often denoted `beta` in equations, or
      None. If present, will be added to the normalized tensor.
    scale: A scale `Tensor`, often denoted `gamma` in equations, or
      `None`. If present, the scale is applied to the normalized tensor.
    variance_epsilon: A small `float` added to the minibatch `variance` to
      prevent dividing by zero.
    name: A name for this operation (optional).

  Returns:
    batch_unnormalized: The de-normalized, de-scaled, de-offset `Tensor`.
  """
  with tf.name_scope(name or 'undo_batch_normalization'):
    # inv = tf.rsqrt(variance + variance_epsilon)
    # if scale is not None:
    #   inv *= scale
    # return x * inv + (
    #     offset - mean * inv if offset is not None else -mean * inv)
    rescale = tf.sqrt(variance + variance_epsilon)
    if scale is not None:
      rescale = rescale / scale
    batch_unnormalized = x * rescale + (
        mean - offset * rescale if offset is not None else mean)
    return batch_unnormalized


class BatchNormalization(bijector.Bijector):
  """Compute `Y = g(X) s.t. X = g^-1(Y) = (Y - mean(Y)) / std(Y)`.

  Applies Batch Normalization [(Ioffe and Szegedy, 2015)][1] to samples from a
  data distribution. This can be used to stabilize training of normalizing
  flows ([Papamakarios et al., 2016][3]; [Dinh et al., 2017][2])

  When training Deep Neural Networks (DNNs), it is common practice to
  normalize or whiten features by shifting them to have zero mean and
  scaling them to have unit variance.

  The `inverse()` method of the `BatchNormalization` bijector, which is used in
  the log-likelihood computation of data samples, implements the normalization
  procedure (shift-and-scale) using the mean and standard deviation of the
  current minibatch.

  Conversely, the `forward()` method of the bijector de-normalizes samples (e.g.
  `X*std(Y) + mean(Y)` with the running-average mean and standard deviation
  computed at training-time. De-normalization is useful for sampling.

  ```python
  distribution = tfd.TransformedDistribution(
      distribution=tfd.Normal(loc=[0.0], scale=[1.0]),
      bijector=tfb.BatchNormalization())

  y = tfd.Normal(loc=[1.0], scale_diag=[2.0]).sample(100)  # ~ N(1, 2)
  x = distribution.bijector.inverse(y)  # ~ N(0, 1)
  y2 = distribution.sample(100)  # ~ N(1, 2)
  ```

  During training time, `BatchNormalization.inverse` and
  `BatchNormalization.forward` are not guaranteed to be inverses of each other
  because `inverse(y)` uses statistics of the current minibatch, while
  `forward(x)` uses running-average statistics accumulated from training. In
  other words, `BatchNormalization.inverse(BatchNormalization.forward(...))` and
  `BatchNormalization.forward(BatchNormalization.inverse(...))` will be
  identical when `training=False` but may be different when `training=True`.

  #### References

  [1]: Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating
       Deep Network Training by Reducing Internal Covariate Shift. In
       _International Conference on Machine Learning_, 2015.
       https://arxiv.org/abs/1502.03167

  [2]: Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density Estimation
       using Real NVP. In _International Conference on Learning
       Representations_, 2017. https://arxiv.org/abs/1605.08803

  [3]: George Papamakarios, Theo Pavlakou, and Iain Murray. Masked
       Autoregressive Flow for Density Estimation. In _Neural Information
       Processing Systems_, 2017. https://arxiv.org/abs/1705.07057
  """

  def __init__(self,
               batchnorm_layer=None,
               training=True,
               validate_args=False,
               name='batch_normalization'):
    """Instantiates the `BatchNormalization` bijector.

    Args:
      batchnorm_layer: `tf.layers.BatchNormalization` layer object. If `None`,
        defaults to a `tf_keras.layers.BatchNormalization` with
        `gamma_constraint=tf.nn.relu(x) + 1e-6)`.
        This ensures positivity of the scale variable.

      training: If True, updates running-average statistics during call to
        `inverse()`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    Raises:
      ValueError: If bn_layer is not an instance of
        `tf.layers.BatchNormalization`, or if it is specified with `renorm=True`
        or a virtual batch size.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      # Scale must be positive.
      g_constraint = lambda x: tf.nn.relu(x) + 1e-6
      self.batchnorm = batchnorm_layer or tf_keras.layers.BatchNormalization(
          gamma_constraint=g_constraint)
      self._validate_bn_layer(self.batchnorm)
      self._training = training
      if isinstance(self.batchnorm.axis, int):
        forward_min_event_ndims = 1
      else:
        forward_min_event_ndims = len(self.batchnorm.axis)
      super(BatchNormalization, self).__init__(
          forward_min_event_ndims=forward_min_event_ndims,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict()

  def _validate_bn_layer(self, layer):
    """Check for valid BatchNormalization layer.

    Args:
      layer: Instance of `tf.layers.BatchNormalization`.
    Raises:
      ValueError: If batchnorm_layer argument is not an instance of
      `tf.layers.BatchNormalization`, or if `batchnorm_layer.renorm=True` or
      if `batchnorm_layer.virtual_batch_size` is specified.
    """
    if (not isinstance(layer, tf_keras.layers.BatchNormalization) and
        not isinstance(layer, tf_keras.tf1_layers.BatchNormalization)):
      raise ValueError(
          'batchnorm_layer must be an instance of '
          '`tf_keras.layers.BatchNormalization` or '
          '`tf.compat.v1.layers.BatchNormalization`. Got {}'.format(
              type(layer)))
    if layer.renorm:
      raise ValueError(
          '`BatchNormalization` Bijector does not support renormalization, '
          'but `batchnorm_layer.renorm` is `True`.')
    if layer.virtual_batch_size:
      raise ValueError(
          '`BatchNormlization` Bijector does not support virtual batch sizes, '
          'but `batchnorm_layer.virtual_batch_size` is `True`.')

  def _get_broadcast_fn(self, x):
    ndims = len(x.shape)
    reduction_axes = [i for i in range(ndims) if i not in self.batchnorm.axis]
    # Broadcasting only necessary for single-axis batch norm where the axis is
    # not the last dimension
    broadcast_shape = [1] * ndims
    broadcast_shape[self.batchnorm.axis[0]] = x.shape[self.batchnorm.axis[0]]
    def _broadcast(v):
      if (v is not None and
          len(v.shape) != ndims and
          reduction_axes != list(range(ndims - 1))):
        return tf.reshape(v, broadcast_shape)
      return v
    return _broadcast

  def _normalize(self, y):
    return self.batchnorm(y, training=self._training)

  def _de_normalize(self, x):
    # Uses the saved statistics.
    if not self.batchnorm.built:
      self.batchnorm.build(x.shape)
    broadcast_fn = self._get_broadcast_fn(x)
    mean = broadcast_fn(self.batchnorm.moving_mean)
    variance = broadcast_fn(self.batchnorm.moving_variance)
    beta = broadcast_fn(self.batchnorm.beta) if self.batchnorm.center else None
    gamma = broadcast_fn(self.batchnorm.gamma) if self.batchnorm.scale else None
    return _undo_batch_normalization(
        x, mean, variance, beta, gamma, self.batchnorm.epsilon)

  def _forward(self, x):
    return self._de_normalize(x)

  def _inverse(self, y):
    return self._normalize(y)

  def _forward_log_det_jacobian(self, x):
    # Uses saved statistics to compute volume distortion.
    return -self._inverse_log_det_jacobian(x, use_saved_statistics=True)

  def _inverse_log_det_jacobian(self, y, use_saved_statistics=False):
    if not self.batchnorm.built:
      # Create variables.
      self.batchnorm.build(y.shape)

    event_dims = self.batchnorm.axis
    reduction_axes = [i for i in range(len(y.shape)) if i not in event_dims]

    # At training-time, ildj is computed from the mean and log-variance across
    # the current minibatch.
    # We use multiplication instead of tf.where() to get easier broadcasting.
    log_variance = tf.math.log(
        tf.where(
            tf.logical_or(use_saved_statistics, tf.logical_not(self._training)),
            self.batchnorm.moving_variance,
            tf.nn.moments(x=y, axes=reduction_axes, keepdims=True)[1]) +
        self.batchnorm.epsilon)

    # TODO(b/137216713): determine whether it's unsafe for the reduce_sums below
    # to happen across all axes.
    # `gamma` and `log Var(y)` reductions over event_dims.
    # Log(total change in area from gamma term).
    log_total_gamma = tf.reduce_sum(tf.math.log(self.batchnorm.gamma))

    # Log(total change in area from log-variance term).
    log_total_variance = tf.reduce_sum(log_variance)
    # The ildj is scalar, as it does not depend on the values of x and are
    # constant across minibatch elements.
    return log_total_gamma - 0.5 * log_total_variance
