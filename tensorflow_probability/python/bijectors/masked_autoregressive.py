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
"""MaskedAutoregressiveFlow bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import six
import tensorflow as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math.numeric import clip_by_value_preserve_gradient


__all__ = [
    "AutoregressiveLayer",
    "MaskedAutoregressiveFlow",
    "masked_autoregressive_default_template",
    "masked_dense",
]


class MaskedAutoregressiveFlow(bijector.Bijector):
  """Affine MaskedAutoregressiveFlow bijector.

  The affine autoregressive flow [(Papamakarios et al., 2016)][3] provides a
  relatively simple framework for user-specified (deep) architectures to learn a
  distribution over continuous events. Regarding terminology,

    "Autoregressive models decompose the joint density as a product of
    conditionals, and model each conditional in turn. Normalizing flows
    transform a base density (e.g. a standard Gaussian) into the target density
    by an invertible transformation with tractable Jacobian."
    [(Papamakarios et al., 2016)][3]

  In other words, the "autoregressive property" is equivalent to the
  decomposition, `p(x) = prod{ p(x[perm[i]] | x[perm[0:i]]) : i=0, ..., d }`
  where `perm` is some permutation of `{0, ..., d}`. In the simple case where
  the permutation is identity this reduces to:
  `p(x) = prod{ p(x[i] | x[0:i]) : i=0, ..., d }`. The provided
  `shift_and_log_scale_fn`, `masked_autoregressive_default_template`, achieves
  this property by zeroing out weights in its `masked_dense` layers.

  In TensorFlow Probability, "normalizing flows" are implemented as
  `tfp.bijectors.Bijector`s. The `forward` "autoregression" is implemented
  using a `tf.while_loop` and a deep neural network (DNN) with masked weights
  such that the autoregressive property is automatically met in the `inverse`.

  A `TransformedDistribution` using `MaskedAutoregressiveFlow(...)` uses the
  (expensive) forward-mode calculation to draw samples and the (cheap)
  reverse-mode calculation to compute log-probabilities. Conversely, a
  `TransformedDistribution` using `Invert(MaskedAutoregressiveFlow(...))` uses
  the (expensive) forward-mode calculation to compute log-probabilities and the
  (cheap) reverse-mode calculation to compute samples.  See "Example Use"
  [below] for more details.

  Given a `shift_and_log_scale_fn`, the forward and inverse transformations are
  (a sequence of) affine transformations. A "valid" `shift_and_log_scale_fn`
  must compute each `shift` (aka `loc` or "mu" in [Germain et al. (2015)][1])
  and `log(scale)` (aka "alpha" in [Germain et al. (2015)][1]) such that each
  are broadcastable with the arguments to `forward` and `inverse`, i.e., such
  that the calculations in `forward`, `inverse` [below] are possible.

  For convenience, `masked_autoregressive_default_template` is offered as a
  possible `shift_and_log_scale_fn` function. It implements the MADE
  architecture [(Germain et al., 2015)][1]. MADE is a feed-forward network that
  computes a `shift` and `log(scale)` using `masked_dense` layers in a deep
  neural network. Weights are masked to ensure the autoregressive property. It
  is possible that this architecture is suboptimal for your task. To build
  alternative networks, either change the arguments to
  `masked_autoregressive_default_template`, use the `masked_dense` function to
  roll-out your own, or use some other architecture, e.g., using `tf.layers`.

  Warning: no attempt is made to validate that the `shift_and_log_scale_fn`
  enforces the "autoregressive property".

  Assuming `shift_and_log_scale_fn` has valid shape and autoregressive
  semantics, the forward transformation is

  ```python
  def forward(x):
    y = zeros_like(x)
    event_size = x.shape[-event_dims:].num_elements()
    for _ in range(event_size):
      shift, log_scale = shift_and_log_scale_fn(y)
      y = x * tf.exp(log_scale) + shift
    return y
  ```

  and the inverse transformation is

  ```python
  def inverse(y):
    shift, log_scale = shift_and_log_scale_fn(y)
    return (y - shift) / tf.exp(log_scale)
  ```

  Notice that the `inverse` does not need a for-loop. This is because in the
  forward pass each calculation of `shift` and `log_scale` is based on the `y`
  calculated so far (not `x`). In the `inverse`, the `y` is fully known, thus is
  equivalent to the scaling used in `forward` after `event_size` passes, i.e.,
  the "last" `y` used to compute `shift`, `log_scale`. (Roughly speaking, this
  also proves the transform is bijective.)

  #### Examples

  ```python
  tfd = tfp.distributions
  tfb = tfp.bijectors

  dims = 5

  # A common choice for a normalizing flow is to use a Gaussian for the base
  # distribution. (However, any continuous distribution would work.) E.g.,
  maf = tfd.TransformedDistribution(
      distribution=tfd.Normal(loc=0., scale=1.),
      bijector=tfb.MaskedAutoregressiveFlow(
          shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
              hidden_layers=[512, 512])),
      event_shape=[dims])

  x = maf.sample()  # Expensive; uses `tf.while_loop`, no Bijector caching.
  maf.log_prob(x)   # Almost free; uses Bijector caching.
  maf.log_prob(0.)  # Cheap; no `tf.while_loop` despite no Bijector caching.

  # [Papamakarios et al. (2016)][3] also describe an Inverse Autoregressive
  # Flow [(Kingma et al., 2016)][2]:
  iaf = tfd.TransformedDistribution(
      distribution=tfd.Normal(loc=0., scale=1.),
      bijector=tfb.Invert(tfb.MaskedAutoregressiveFlow(
          shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
              hidden_layers=[512, 512]))),
      event_shape=[dims])

  x = iaf.sample()  # Cheap; no `tf.while_loop` despite no Bijector caching.
  iaf.log_prob(x)   # Almost free; uses Bijector caching.
  iaf.log_prob(0.)  # Expensive; uses `tf.while_loop`, no Bijector caching.

  # In many (if not most) cases the default `shift_and_log_scale_fn` will be a
  # poor choice. Here's an example of using a "shift only" version and with a
  # different number/depth of hidden layers.
  shift_only = True
  maf_no_scale_hidden2 = tfd.TransformedDistribution(
      distribution=tfd.Normal(loc=0., scale=1.),
      bijector=tfb.MaskedAutoregressiveFlow(
          tfb.masked_autoregressive_default_template(
              hidden_layers=[32],
              shift_only=shift_only),
          is_constant_jacobian=shift_only),
      event_shape=[dims])
  ```

  #### References

  [1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
       Masked Autoencoder for Distribution Estimation. In _International
       Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509

  [2]: Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya
       Sutskever, and Max Welling. Improving Variational Inference with Inverse
       Autoregressive Flow. In _Neural Information Processing Systems_, 2016.
       https://arxiv.org/abs/1606.04934

  [3]: George Papamakarios, Theo Pavlakou, and Iain Murray. Masked
       Autoregressive Flow for Density Estimation. In _Neural Information
       Processing Systems_, 2017. https://arxiv.org/abs/1705.07057
  """

  def __init__(self,
               shift_and_log_scale_fn,
               is_constant_jacobian=False,
               validate_args=False,
               unroll_loop=False,
               event_ndims=1,
               name=None):
    """Creates the MaskedAutoregressiveFlow bijector.

    Args:
      shift_and_log_scale_fn: Python `callable` which computes `shift` and
        `log_scale` from both the forward domain (`x`) and the inverse domain
        (`y`). Calculation must respect the "autoregressive property" (see class
        docstring). Suggested default
        `masked_autoregressive_default_template(hidden_layers=...)`.
        Typically the function contains `tf.Variables` and is wrapped using
        `tf.make_template`. Returning `None` for either (both) `shift`,
        `log_scale` is equivalent to (but more efficient than) returning zero.
      is_constant_jacobian: Python `bool`. Default: `False`. When `True` the
        implementation assumes `log_scale` does not depend on the forward domain
        (`x`) or inverse domain (`y`) values. (No validation is made;
        `is_constant_jacobian=False` is always safe but possibly computationally
        inefficient.)
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      unroll_loop: Python `bool` indicating whether the `tf.while_loop` in
        `_forward` should be replaced with a static for loop. Requires that
        the final dimension of `x` be known at graph construction time. Defaults
        to `False`.
      event_ndims: Python `integer`, the intrinsic dimensionality of this
        bijector. 1 corresponds to a simple vector autoregressive bijector as
        implemented by the `masked_autoregressive_default_template`, 2 might be
        useful for a 2D convolutional `shift_and_log_scale_fn` and so on.
      name: Python `str`, name given to ops managed by this object.
    """
    name = name or "masked_autoregressive_flow"
    self._shift_and_log_scale_fn = shift_and_log_scale_fn
    self._unroll_loop = unroll_loop
    self._event_ndims = event_ndims
    super(MaskedAutoregressiveFlow, self).__init__(
        forward_min_event_ndims=self._event_ndims,
        is_constant_jacobian=is_constant_jacobian,
        validate_args=validate_args,
        name=name)

  def _forward(self, x):
    static_event_size = tensorshape_util.num_elements(
        tensorshape_util.with_rank_at_least(
            x.shape, self._event_ndims)[-self._event_ndims:])

    if self._unroll_loop:
      if not static_event_size:
        raise ValueError(
            "The final {} dimensions of `x` must be known at graph "
            "construction time if `unroll_loop=True`. `x.shape: {!r}`".format(
                self._event_ndims, x.shape))
      y = tf.zeros_like(x, name="y0")

      for _ in range(static_event_size):
        shift, log_scale = self._shift_and_log_scale_fn(y)
        # next_y = scale * x + shift
        next_y = x
        if log_scale is not None:
          next_y *= tf.exp(log_scale)
        if shift is not None:
          next_y += shift
        y = next_y
      return y

    event_size = tf.reduce_prod(
        input_tensor=tf.shape(input=x)[-self._event_ndims:])
    y0 = tf.zeros_like(x, name="y0")
    # call the template once to ensure creation
    _ = self._shift_and_log_scale_fn(y0)
    def _loop_body(index, y0):
      """While-loop body for autoregression calculation."""
      # Set caching device to avoid re-getting the tf.Variable for every while
      # loop iteration.
      with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()) as vs:
        if vs.caching_device is None and not tf.executing_eagerly():
          vs.set_caching_device(lambda op: op.device)
        shift, log_scale = self._shift_and_log_scale_fn(y0)
      y = x
      if log_scale is not None:
        y *= tf.exp(log_scale)
      if shift is not None:
        y += shift
      return index + 1, y
    # If the event size is available at graph construction time, we can inform
    # the graph compiler of the maximum number of steps. If not,
    # static_event_size will be None, and the maximum_iterations argument will
    # have no effect.
    _, y = tf.while_loop(
        cond=lambda index, _: index < event_size,
        body=_loop_body,
        loop_vars=(0, y0),
        maximum_iterations=static_event_size)
    return y

  def _inverse(self, y):
    shift, log_scale = self._shift_and_log_scale_fn(y)
    x = y
    if shift is not None:
      x -= shift
    if log_scale is not None:
      x *= tf.exp(-log_scale)
    return x

  def _inverse_log_det_jacobian(self, y):
    _, log_scale = self._shift_and_log_scale_fn(y)
    if log_scale is None:
      return tf.constant(0., dtype=y.dtype, name="ildj")
    return -tf.reduce_sum(
        input_tensor=log_scale, axis=tf.range(-self._event_ndims, 0))


MASK_INCLUSIVE = "inclusive"
MASK_EXCLUSIVE = "exclusive"


def _gen_slices(num_blocks, n_in, n_out, mask_type=MASK_EXCLUSIVE):
  """Generate the slices for building an autoregressive mask."""
  # TODO(b/67594795): Better support of dynamic shape.
  slices = []
  col = 0
  d_in = n_in // num_blocks
  d_out = n_out // num_blocks
  row = d_out if mask_type == MASK_EXCLUSIVE else 0
  for _ in range(num_blocks):
    row_slice = slice(row, None)
    col_slice = slice(col, col + d_in)
    slices.append([row_slice, col_slice])
    col += d_in
    row += d_out
  return slices


def _gen_mask(num_blocks,
              n_in,
              n_out,
              mask_type=MASK_EXCLUSIVE,
              dtype=tf.float32):
  """Generate the mask for building an autoregressive dense layer."""
  # TODO(b/67594795): Better support of dynamic shape.
  mask = np.zeros([n_out, n_in], dtype=dtype.as_numpy_dtype())
  slices = _gen_slices(num_blocks, n_in, n_out, mask_type=mask_type)
  for [row_slice, col_slice] in slices:
    mask[row_slice, col_slice] = 1
  return mask


def masked_dense(inputs,
                 units,
                 num_blocks=None,
                 exclusive=False,
                 kernel_initializer=None,
                 reuse=None,
                 name=None,
                 *args,  # pylint: disable=keyword-arg-before-vararg
                 **kwargs):
  """A autoregressively masked dense layer. Analogous to `tf.layers.dense`.

  See [Germain et al. (2015)][1] for detailed explanation.

  Arguments:
    inputs: Tensor input.
    units: Python `int` scalar representing the dimensionality of the output
      space.
    num_blocks: Python `int` scalar representing the number of blocks for the
      MADE masks.
    exclusive: Python `bool` scalar representing whether to zero the diagonal of
      the mask, used for the first layer of a MADE.
    kernel_initializer: Initializer function for the weight matrix.
      If `None` (default), weights are initialized using the
      `tf.glorot_random_initializer`.
    reuse: Python `bool` scalar representing whether to reuse the weights of a
      previous layer by the same name.
    name: Python `str` used to describe ops managed by this function.
    *args: `tf.layers.dense` arguments.
    **kwargs: `tf.layers.dense` keyword arguments.

  Returns:
    Output tensor.

  Raises:
    NotImplementedError: if rightmost dimension of `inputs` is unknown prior to
      graph execution.

  #### References

  [1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
       Masked Autoencoder for Distribution Estimation. In _International
       Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509
  """
  # TODO(b/67594795): Better support of dynamic shape.
  input_depth = tf.compat.dimension_value(
      tensorshape_util.with_rank_at_least(inputs.shape, 1)[-1])
  if input_depth is None:
    raise NotImplementedError(
        "Rightmost dimension must be known prior to graph execution.")

  mask = _gen_mask(num_blocks, input_depth, units,
                   MASK_EXCLUSIVE if exclusive else MASK_INCLUSIVE).T

  if kernel_initializer is None:
    kernel_initializer = tf.compat.v1.glorot_normal_initializer()

  def masked_initializer(shape, dtype=None, partition_info=None):
    return mask * kernel_initializer(shape, dtype, partition_info)

  with tf.compat.v2.name_scope(name or "masked_dense"):
    layer = tf.compat.v1.layers.Dense(
        units,
        kernel_initializer=masked_initializer,
        kernel_constraint=lambda x: mask * x,
        name=name,
        dtype=dtype_util.base_dtype(inputs.dtype),
        _scope=name,
        _reuse=reuse,
        *args,  # pylint: disable=keyword-arg-before-vararg
        **kwargs)
    return layer.apply(inputs)


def masked_autoregressive_default_template(hidden_layers,
                                           shift_only=False,
                                           activation=tf.nn.relu,
                                           log_scale_min_clip=-5.,
                                           log_scale_max_clip=3.,
                                           log_scale_clip_gradient=False,
                                           name=None,
                                           *args,  # pylint: disable=keyword-arg-before-vararg
                                           **kwargs):
  """Build the Masked Autoregressive Density Estimator (Germain et al., 2015).

  This will be wrapped in a make_template to ensure the variables are only
  created once. It takes the input and returns the `loc` ("mu" in [Germain et
  al. (2015)][1]) and `log_scale` ("alpha" in [Germain et al. (2015)][1]) from
  the MADE network.

  Warning: This function uses `masked_dense` to create randomly initialized
  `tf.Variables`. It is presumed that these will be fit, just as you would any
  other neural architecture which uses `tf.layers.dense`.

  #### About Hidden Layers

  Each element of `hidden_layers` should be greater than the `input_depth`
  (i.e., `input_depth = tf.shape(input)[-1]` where `input` is the input to the
  neural network). This is necessary to ensure the autoregressivity property.

  #### About Clipping

  This function also optionally clips the `log_scale` (but possibly not its
  gradient). This is useful because if `log_scale` is too small/large it might
  underflow/overflow making it impossible for the `MaskedAutoregressiveFlow`
  bijector to implement a bijection. Additionally, the `log_scale_clip_gradient`
  `bool` indicates whether the gradient should also be clipped. The default does
  not clip the gradient; this is useful because it still provides gradient
  information (for fitting) yet solves the numerical stability problem. I.e.,
  `log_scale_clip_gradient = False` means
  `grad[exp(clip(x))] = grad[x] exp(clip(x))` rather than the usual
  `grad[clip(x)] exp(clip(x))`.

  Args:
    hidden_layers: Python `list`-like of non-negative integer, scalars
      indicating the number of units in each hidden layer. Default: `[512, 512].
    shift_only: Python `bool` indicating if only the `shift` term shall be
      computed. Default: `False`.
    activation: Activation function (callable). Explicitly setting to `None`
      implies a linear activation.
    log_scale_min_clip: `float`-like scalar `Tensor`, or a `Tensor` with the
      same shape as `log_scale`. The minimum value to clip by. Default: -5.
    log_scale_max_clip: `float`-like scalar `Tensor`, or a `Tensor` with the
      same shape as `log_scale`. The maximum value to clip by. Default: 3.
    log_scale_clip_gradient: Python `bool` indicating that the gradient of
      `tf.clip_by_value` should be preserved. Default: `False`.
    name: A name for ops managed by this function. Default:
      "masked_autoregressive_default_template".
    *args: `tf.layers.dense` arguments.
    **kwargs: `tf.layers.dense` keyword arguments.

  Returns:
    shift: `Float`-like `Tensor` of shift terms (the "mu" in
      [Germain et al.  (2015)][1]).
    log_scale: `Float`-like `Tensor` of log(scale) terms (the "alpha" in
      [Germain et al. (2015)][1]).

  Raises:
    NotImplementedError: if rightmost dimension of `inputs` is unknown prior to
      graph execution.

  #### References

  [1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
       Masked Autoencoder for Distribution Estimation. In _International
       Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509
  """
  name = name or "masked_autoregressive_default_template"
  with tf.compat.v2.name_scope(name):

    def _fn(x):
      """MADE parameterized via `masked_autoregressive_default_template`."""
      # TODO(b/67594795): Better support of dynamic shape.
      input_depth = tf.compat.dimension_value(
          tensorshape_util.with_rank_at_least(x.shape, 1)[-1])
      if input_depth is None:
        raise NotImplementedError(
            "Rightmost dimension must be known prior to graph execution.")
      input_shape = (
          np.int32(tensorshape_util.as_list(x.shape))
          if tensorshape_util.is_fully_defined(x.shape) else tf.shape(input=x))
      if tensorshape_util.rank(x.shape) == 1:
        x = x[tf.newaxis, ...]
      for i, units in enumerate(hidden_layers):
        x = masked_dense(
            inputs=x,
            units=units,
            num_blocks=input_depth,
            exclusive=True if i == 0 else False,
            activation=activation,
            *args,  # pylint: disable=keyword-arg-before-vararg
            **kwargs)
      x = masked_dense(
          inputs=x,
          units=(1 if shift_only else 2) * input_depth,
          num_blocks=input_depth,
          activation=None,
          *args,  # pylint: disable=keyword-arg-before-vararg
          **kwargs)
      if shift_only:
        x = tf.reshape(x, shape=input_shape)
        return x, None
      x = tf.reshape(x, shape=tf.concat([input_shape, [2]], axis=0))
      shift, log_scale = tf.unstack(x, num=2, axis=-1)
      which_clip = (
          tf.clip_by_value
          if log_scale_clip_gradient else clip_by_value_preserve_gradient)
      log_scale = which_clip(log_scale, log_scale_min_clip, log_scale_max_clip)
      return shift, log_scale

    return tf.compat.v1.make_template(name, _fn)


class AutoregressiveLayer(tf.keras.layers.Layer):
  r"""Masked Autoencoder for Distribution Estimation [Germain et al. (2015)][1].

  A `AutoregressiveLayer` takes as input a Tensor of shape `[..., event_size]`
  and returns a Tensor of shape `[..., event_size, params]`.

  The output satisfies the autoregressive property.  That is, the layer is
  configured with some permutation `ord` of `{0, ..., event_size-1}` (i.e., an
  ordering of the input dimensions), and the output `output[batch_idx, i, ...]`
  for input dimension `i` depends only on inputs `x[batch_idx, j]` where
  `ord(j) < ord(i)`.  The autoregressive property allows us to use
  `output[batch_idx, i]` to parameterize conditional distributions:
    `p(x[batch_idx, i] | x[batch_idx, ] for ord(j) < ord(i))`
  which give us a tractable distribution over input `x[batch_idx]`:
    `p(x[batch_idx]) = prod_i p(x[batch_idx, ord(i)] | x[batch_idx, ord(0:i)])`

  For example, when `params` is 2, the output of the layer can parameterize
  the location and log-scale of an autoregressive Gaussian distribution.

  #### Example

  ```python
  # Generate data -- as in Figure 1 in [Papamakarios et al. (2017)][2]).
  n = 2000
  x2 = np.random.randn(n).astype(dtype=np.float32) * 2.
  x1 = np.random.randn(n).astype(dtype=np.float32) + (x2 * x2 / 4.)
  data = np.stack([x1, x2], axis=-1)

  # Density estimation with MADE.
  made = tfb.AutoregressiveLayer(params=2, hidden_units=[10, 10])

  distribution = tfd.TransformedDistribution(
      distribution=tfd.Normal(loc=0., scale=1.),
      bijector=tfb.MaskedAutoregressiveFlow(
          lambda x: tf.unstack(made(x), num=2, axis=-1)),
      event_shape=[2])

  # Construct and fit model.
  x_ = tfkl.Input(shape=(2,), dtype=tf.float32)
  log_prob_ = distribution.log_prob(x_)
  model = tfk.Model(x_, log_prob_)

  model.compile(optimizer=tf.compat.v2.optimizers.Adam(),
                loss=lambda _, log_prob: -log_prob)

  batch_size = 25
  model.fit(x=data,
            y=np.zeros((n, 0), dtype=np.float32),
            batch_size=batch_size,
            epochs=1,
            steps_per_epoch=1,  # Usually `n // batch_size`.
            shuffle=True,
            verbose=True)

  # Use the fitted distribution.
  distribution.sample((3, 1))
  distribution.log_prob(np.ones((3, 2), dtype=np.float32))
  ```

  #### Examples: Handling Rank-2+ Tensors

  `AutoregressiveLayer` can be used as a building block to achieve different
  autoregressive structures over rank-2+ tensors.  For example, suppose we want
  to build an autoregressive distribution over images with dimension `[weight,
  height, channels]` with `channels = 3`:

   1. We can parameterize a "fully autoregressive" distribution, with
      cross-channel and within-pixel autoregressivity:
      ```
          r0    g0   b0     r0    g0   b0       r0   g0    b0
          ^   ^      ^         ^   ^   ^         ^      ^   ^
          |  /  ____/           \  |  /           \____  \  |
          | /__/                 \ | /                 \__\ |
          r1    g1   b1     r1 <- g1   b1       r1   g1 <- b1
                                               ^          |
                                                \_________/
      ```

      as:
      ```python
      # Generate random images for training data.
      images = np.random.uniform(size=(100, 8, 8, 3)).astype(np.float32)
      n, width, height, channels = images.shape

      # Reshape images to achieve desired autoregressivity.
      event_shape = [height * width * channels]
      reshaped_images = tf.reshape(images, [n, event_shape])

      # Density estimatino with MADE.
      made = tfb.AutoregressiveLayer(params=2, event_shape=event_shape,
                                     hidden_units=[20, 20], activation="relu")
      distribution = tfd.TransformedDistribution(
          distribution=tfd.Normal(loc=0., scale=1.),
          bijector=tfb.MaskedAutoregressiveFlow(
              lambda x: tf.unstack(made(x), num=2, axis=-1)),
          event_shape=event_shape)

      # Construct and fit model.
      x_ = tfkl.Input(shape=event_shape, dtype=tf.float32)
      log_prob_ = distribution.log_prob(x_)
      model = tfk.Model(x_, log_prob_)

      model.compile(optimizer=tf.compat.v2.optimizers.Adam(),
                    loss=lambda _, log_prob: -log_prob)

      batch_size = 10
      model.fit(x=data,
                y=np.zeros((n, 0), dtype=np.float32),
                batch_size=batch_size,
                epochs=10,
                steps_per_epoch=n // batch_size,
                shuffle=True,
                verbose=True)

      # Use the fitted distribution.
      distribution.sample((3, 1))
      distribution.log_prob(np.ones((5, 8, 8, 3), dtype=np.float32))
      ```

   2. We can parameterize a distribution with neither cross-channel nor
      within-pixel autoregressivity:
      ```
          r0    g0   b0
          ^     ^    ^
          |     |    |
          |     |    |
          r1    g1   b1
      ```

      as:
      ```python
      # Generate fake images.
      images = np.random.choice([0, 1], size=(100, 8, 8, 3))
      n, width, height, channels = images.shape

      # Reshape images to achieve desired autoregressivity.
      reshaped_images = np.transpose(
          np.reshape(images, [n, width * height, channels]),
          axes=[0, 2, 1])

      made = tfb.AutoregressiveLayer(params=1, event_shape=[width * height],
                                     hidden_units=[20, 20], activation="relu")

      # Density estimation with MADE.
      #
      # NOTE: Parameterize an autoregressive distribution over an event_shape of
      # [channels, width * height], with univariate Bernoulli conditional
      # distributions.
      distribution = tfd.Autoregressive(
          lambda x: tfd.Independent(
              tfd.Bernoulli(logits=tf.unstack(made(x), axis=-1)[0],
                            dtype=tf.float32),
              reinterpreted_batch_ndims=2),
          sample0=tf.zeros([channels, width * height], dtype=tf.float32))

      # Construct and fit model.
      x_ = tfkl.Input(shape=(channels, width * height), dtype=tf.float32)
      log_prob_ = distribution.log_prob(x_)
      model = tfk.Model(x_, log_prob_)

      model.compile(optimizer=tf.compat.v2.optimizers.Adam(),
                    loss=lambda _, log_prob: -log_prob)

      batch_size = 10
      model.fit(x=reshaped_images,
                y=np.zeros((n, 0), dtype=np.float32),
                batch_size=batch_size,
                epochs=10,
                steps_per_epoch=n // batch_size,
                shuffle=True,
                verbose=True)

      distribution.sample(7)
      distribution.log_prob(np.ones((4, 8, 8, 3), dtype=np.float32))
      ```

      Note that one set of weights is shared for the mapping for each channel
      from image to distribution parameters -- i.e., the mapping
      `layer(reshaped_images[..., channel, :])`, where `channel` is 0, 1, or 2.

      To use separate weights for each channel, we could construct an
      `AutoregressiveLayer` and `TransformedDistribution` for each channel, and
      combine them with a `tfd.Blockwise` distribution.

  #### References

  [1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
       Masked Autoencoder for Distribution Estimation. In _International
       Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509

  [2]: George Papamakarios, Theo Pavlakou, Iain Murray, Masked Autoregressive
       Flow for Density Estimation.  In _Neural Information Processing Systems_,
       2017. https://arxiv.org/abs/1705.07057
  """

  def __init__(self,
               params,
               event_shape=None,
               hidden_units=None,
               input_order="left-to-right",
               hidden_degrees="equal",
               activation=None,
               use_bias=True,
               kernel_initializer="glorot_uniform",
               validate_args=False,
               **kwargs):
    """Constructs the MADE layer.

    Arguments:
      params: Python integer specifying the number of parameters to output
        per input.
      event_shape: Python `list`-like of positive integers (or a single int),
        specifying the shape of the input to this layer, which is also the
        event_shape of the distribution parameterized by this layer.  Currently
        only rank-1 shapes are supported.  That is, event_shape must be a single
        integer.  If not specified, the event shape is inferred when this layer
        is first called or built.
      hidden_units: Python `list`-like of non-negative integers, specifying
        the number of units in each hidden layer.
      input_order: Order of degrees to the input units: 'random',
        'left-to-right', 'right-to-left', or an array of an explicit order. For
        example, 'left-to-right' builds an autoregressive model:
        `p(x) = p(x1) p(x2 | x1) ... p(xD | x<D)`.  Default: 'left-to-right'.
      hidden_degrees: Method for assigning degrees to the hidden units:
        'equal', 'random'.  If 'equal', hidden units in each layer are allocated
        equally (up to a remainder term) to each degree.  Default: 'equal'.
      activation: An activation function.  See `tf.keras.layers.Dense`. Default:
        `None`.
      use_bias: Whether or not the dense layers constructed in this layer
        should have a bias term.  See `tf.keras.layers.Dense`.  Default: `True`.
      kernel_initializer: Initializer for the kernel weights matrix.  Default:
        'glorot_uniform'.
      validate_args: Python `bool`, default `False`. When `True`, layer
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      **kwargs: Addition keyword arguments passed to the `tf.keras.layers.Dense`
        constructed by this layer.
    """
    super(AutoregressiveLayer, self).__init__(**kwargs)

    self._params = params
    self._event_shape = _list(event_shape) if event_shape is not None else None
    self._hidden_units = hidden_units if hidden_units is not None else []
    self._input_order_param = input_order
    self._hidden_degrees = hidden_degrees
    self._activation = activation
    self._use_bias = use_bias
    self._kernel_initializer = kernel_initializer
    self._validate_args = validate_args
    self._kwargs = kwargs

    if self._event_shape is not None:
      self._event_size = self._event_shape[-1]
      self._event_ndims = len(self._event_shape)

      if self._event_ndims != 1:
        raise ValueError("Parameter `event_shape` must describe a rank-1 shape."
                         " `event_shape: {!r}`".format(event_shape))

    # To be built in `build`.
    self._input_order = None
    self._masks = None
    self._network = None

  def build(self, input_shape):
    """See tfkl.Layer.build."""
    if self._event_shape is None:
      # `event_shape` wasn't specied at __init__, so infer from `input_shape`.
      self._event_shape = [tf.compat.dimension_value(input_shape[-1])]
      self._event_size = self._event_shape[-1]
      self._event_ndims = len(self._event_shape)
      # Should we throw if input_shape has rank > 2?

    if input_shape[-1] != self._event_shape[-1]:
      raise ValueError("Invalid final dimension of `input_shape`. "
                       "Expected `{!r}`, but got `{!r}`".format(
                           self._event_shape[-1], input_shape[-1]))

    # Construct the masks.
    self._input_order = _create_input_order(
        self._event_size, self._input_order_param)
    self._masks = _create_masks(_create_degrees(
        input_size=self._event_size,
        hidden_units=self._hidden_units,
        input_order=self._input_order,
        hidden_degrees=self._hidden_degrees))

    # In the final layer, we will produce `self._params` outputs for each of the
    # `self._event_size` inputs to `AutoregressiveLayer`.  But `masks[-1]` has
    # shape `[self._hidden_units[-1], self._event_size]`.  Thus, we need to
    # expand the mask to `[hidden_units[-1], event_size * self._params]` such
    # that all units for the same input are masked identically.  In particular,
    # we tile the mask so the j-th element of `tf.unstack(output, axis=-1)` is a
    # tensor of the j-th parameter/unit for each input.
    #
    # NOTE: Other orderings of the output could be faster -- should benchmark.
    self._masks[-1] = np.reshape(
        np.tile(self._masks[-1][..., tf.newaxis], [1, 1, self._params]),
        [self._masks[-1].shape[0], self._event_size * self._params])

    self._network = tf.keras.Sequential([
        # Starting this model with an `InputLayer` ensures that Keras will build
        # and propagate our `dtype` to each layer we add.
        tf.keras.layers.InputLayer((self._event_size,), dtype=self.dtype)
    ])

    # Input-to-hidden, hidden-to-hidden, and hidden-to-output layers:
    #  [..., self._event_size] -> [..., self._hidden_units[0]].
    #  [..., self._hidden_units[k-1]] -> [..., self._hidden_units[k]].
    #  [..., self._hidden_units[-1]] -> [..., event_size * self._params].
    layer_output_sizes = self._hidden_units + [self._event_size * self._params]
    for k in range(len(self._masks)):
      self._network.add(tf.keras.layers.Dense(
          layer_output_sizes[k],
          kernel_initializer=_make_masked_initializer(
              self._masks[k], self._kernel_initializer),
          kernel_constraint=_make_masked_constraint(self._masks[k]),
          activation=self._activation if k + 1 < len(self._masks) else None,
          use_bias=self._use_bias,
          **self._kwargs))

    # Record that the layer has been built.
    super(AutoregressiveLayer, self).build(input_shape)

  def call(self, x):
    """See tfkl.Layer.call."""
    with tf.compat.v2.name_scope(self.name or "AutoregressiveLayer_call"):
      x = tf.convert_to_tensor(value=x, dtype=self.dtype, name="x")
      input_shape = tf.shape(input=x)
      # TODO(b/67594795): Better support for dynamic shapes.
      if tensorshape_util.rank(x.shape) == 1:
        x = x[tf.newaxis, ...]
      return tf.reshape(self._network(x),
                        tf.concat([input_shape, [self._params]], axis=0))

  def compute_output_shape(self, input_shape):
    """See tfkl.Layer.compute_output_shape."""
    return input_shape + (self._params,)

  @property
  def event_shape(self):
    return self._event_shape

  @property
  def params(self):
    return self._params


def _list(xs):
  """Convert the given argument to a list."""
  try:
    return list(xs)
  except TypeError:
    return [xs]


def _create_input_order(input_size, input_order="left-to-right"):
  """Returns a degree vectors for the input."""
  if isinstance(input_order, six.string_types):
    if input_order == "left-to-right":
      return np.arange(start=1, stop=input_size + 1)
    elif input_order == "right-to-left":
      return np.arange(start=input_size, stop=0, step=-1)
    elif input_order == "random":
      ret = np.arange(start=1, stop=input_size + 1)
      np.random.shuffle(ret)
      return ret
  elif np.all(np.sort(input_order) == np.arange(1, input_size + 1)):
    return np.array(input_order)

  raise ValueError("Invalid input order: '{}'.".format(input_order))


def _create_degrees(input_size,
                    hidden_units=None,
                    input_order="left-to-right",
                    hidden_degrees="equal"):
  """Returns a list of degree vectors, one for each input and hidden layer.

  A unit with degree d can only receive input from units with degree < d. Output
  units always have the same degree as their associated input unit.

  Args:
    input_size: Number of inputs.
    hidden_units: list with the number of hidden units per layer. It does not
      include the output layer. Each hidden unit size must be at least the size
      of length (otherwise autoregressivity is not possible).
    input_order: Order of degrees to the input units: 'random', 'left-to-right',
      'right-to-left', or an array of an explicit order. For example,
      'left-to-right' builds an autoregressive model
      p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
    hidden_degrees: Method for assigning degrees to the hidden units:
      'equal', 'random'.  If 'equal', hidden units in each layer are allocated
      equally (up to a remainder term) to each degree.  Default: 'equal'.

  Raises:
    ValueError: invalid input order.
    ValueError: invalid hidden degrees.
  """
  input_order = _create_input_order(input_size, input_order)
  degrees = [input_order]

  if hidden_units is None:
    hidden_units = []

  for units in hidden_units:
    if isinstance(hidden_degrees, six.string_types):
      if hidden_degrees == "random":
        # samples from: [low, high)
        degrees.append(
            np.random.randint(low=min(np.min(degrees[-1]), input_size - 1),
                              high=input_size,
                              size=units))
      elif hidden_degrees == "equal":
        min_degree = min(np.min(degrees[-1]), input_size - 1)
        degrees.append(np.maximum(
            min_degree,
            # Evenly divide the range `[1, input_size - 1]` in to `units + 1`
            # segments, and pick the boundaries between the segments as degrees.
            np.ceil(np.arange(1, units + 1)
                    * (input_size - 1) / float(units + 1)).astype(np.int32)))
    else:
      raise ValueError('Invalid hidden order: "{}".'.format(hidden_degrees))

  return degrees


def _create_masks(degrees):
  """Returns a list of binary mask matrices enforcing autoregressivity."""
  return [
      # Create input->hidden and hidden->hidden masks.
      inp[:, np.newaxis] <= out
      for inp, out in zip(degrees[:-1], degrees[1:])
  ] + [
      # Create hidden->output mask.
      degrees[-1][:, np.newaxis] < degrees[0]
  ]


def _make_masked_initializer(mask, initializer):
  """Returns a masked version of the given initializer."""
  initializer = tf.keras.initializers.get(initializer)
  def masked_initializer(shape, dtype=None, partition_info=None):
    # If no `partition_info` is given, then don't pass it to `initializer`, as
    # `initializer` may be a `tf.compat.v2.initializers.Initializer` (which
    # don't accept a `partition_info` argument).
    if partition_info is None:
      x = initializer(shape, dtype)
    else:
      x = initializer(shape, dtype, partition_info)
    return tf.cast(mask, x.dtype) * x
  return masked_initializer


def _make_masked_constraint(mask):
  def masked_constraint(x):
    x = tf.convert_to_tensor(value=x, dtype_hint=tf.float32, name="x")
    return tf.cast(mask, x.dtype) * x
  return masked_constraint
