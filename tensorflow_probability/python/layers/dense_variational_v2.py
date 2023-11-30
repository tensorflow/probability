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
"""DenseVariational layer."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import kullback_leibler

from tensorflow_probability.python.internal import tf_keras


class DenseVariational(tf_keras.layers.Layer):
  """Dense layer with random `kernel` and `bias`.

  This layer uses variational inference to fit a "surrogate" posterior to the
  distribution over both the `kernel` matrix and the `bias` terms which are
  otherwise used in a manner similar to `tf_keras.layers.Dense`.

  This layer fits the "weights posterior" according to the following generative
  process:

  ```none
  [K, b] ~ Prior()
  M = matmul(X, K) + b
  Y ~ Likelihood(M)
  ```

  """

  def __init__(self,
               units,
               make_posterior_fn,
               make_prior_fn,
               kl_weight=None,
               kl_use_exact=False,
               activation=None,
               use_bias=True,
               activity_regularizer=None,
               **kwargs):
    """Creates the `DenseVariational` layer.

    Args:
      units: Positive integer, dimensionality of the output space.
      make_posterior_fn: Python callable taking `tf.size(kernel)`,
        `tf.size(bias)`, `dtype` and returns another callable which takes an
        input and produces a `tfd.Distribution` instance.
      make_prior_fn: Python callable taking `tf.size(kernel)`, `tf.size(bias)`,
        `dtype` and returns another callable which takes an input and produces a
        `tfd.Distribution` instance.
      kl_weight: Amount by which to scale the KL divergence loss between prior
        and posterior.
      kl_use_exact: Python `bool` indicating that the analytical KL divergence
        should be used rather than a Monte Carlo approximation.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")..
      **kwargs: Extra arguments forwarded to `tf_keras.layers.Layer`.
    """
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)
    super(DenseVariational, self).__init__(
        activity_regularizer=tf_keras.regularizers.get(activity_regularizer),
        **kwargs)
    self.units = int(units)

    self._make_posterior_fn = make_posterior_fn
    self._make_prior_fn = make_prior_fn
    self._kl_divergence_fn = _make_kl_divergence_penalty(
        kl_use_exact, weight=kl_weight)

    self.activation = tf_keras.activations.get(activation)
    self.use_bias = use_bias
    self.supports_masking = False
    self.input_spec = tf_keras.layers.InputSpec(min_ndim=2)

  def build(self, input_shape):
    dtype = tf.as_dtype(self.dtype or tf_keras.backend.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))
    input_shape = tf.TensorShape(input_shape)
    last_dim = tf.compat.dimension_value(input_shape[-1])
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `DenseVariational` '
                       'should be defined. Found `None`.')
    self.input_spec = tf_keras.layers.InputSpec(
        min_ndim=2, axes={-1: last_dim})

    with tf.name_scope('posterior'):
      self._posterior = self._make_posterior_fn(
          last_dim * self.units,
          self.units if self.use_bias else 0,
          dtype)
    with tf.name_scope('prior'):
      self._prior = self._make_prior_fn(
          last_dim * self.units,
          self.units if self.use_bias else 0,
          dtype)

    self.built = True

  def call(self, inputs):
    dtype = tf.as_dtype(self.dtype or tf_keras.backend.floatx())
    inputs = tf.cast(inputs, dtype, name='inputs')

    q = self._posterior(inputs)
    r = self._prior(inputs)
    self.add_loss(self._kl_divergence_fn(q, r))

    w = tf.convert_to_tensor(value=q)
    prev_units = self.input_spec.axes[-1]
    if self.use_bias:
      split_sizes = [prev_units * self.units, self.units]
      kernel, bias = tf.split(w, split_sizes, axis=-1)
    else:
      kernel, bias = w, None

    kernel = tf.reshape(kernel, shape=tf.concat([
        tf.shape(kernel)[:-1],
        [prev_units, self.units],
    ], axis=0))
    outputs = tf.matmul(inputs, kernel)

    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, bias)

    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint: disable=not-callable

    return outputs

  def compute_output_shape(self, input_shape):
    """Computes the output shape of the layer.

    Args:
      input_shape: `TensorShape` or `list` of `TensorShape`
        (only last dim is used)
    Returns:
      The output shape.
    Raises:
        ValueError: If the innermost dimension of `input_shape` is not defined.
    """
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if input_shape[-1] is None:
      raise ValueError(
          f'The innermost dimension of input_shape must be defined, but saw: {input_shape}'
      )
    return input_shape[:-1].concatenate(self.units)


def _make_kl_divergence_penalty(
    use_exact_kl=False,
    test_points_reduce_axis=(),  # `None` == "all"; () == "none".
    test_points_fn=tf.convert_to_tensor,
    weight=None):
  """Creates a callable computing `KL[a,b]` from `a`, a `tfd.Distribution`."""

  if use_exact_kl:
    kl_divergence_fn = kullback_leibler.kl_divergence
  else:
    def kl_divergence_fn(distribution_a, distribution_b):
      z = test_points_fn(distribution_a)
      return tf.reduce_mean(
          distribution_a.log_prob(z) - distribution_b.log_prob(z),
          axis=test_points_reduce_axis)

  # Closure over: kl_divergence_fn, weight.
  def _fn(distribution_a, distribution_b):
    """Closure that computes KLDiv as a function of `a` as in `KL[a, b]`."""
    with tf.name_scope('kldivergence_loss'):
      kl = kl_divergence_fn(distribution_a, distribution_b)
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
