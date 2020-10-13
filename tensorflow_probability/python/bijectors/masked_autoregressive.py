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
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.bijectors import chain
from tensorflow_probability.python.bijectors import scale as scale_lib
from tensorflow_probability.python.bijectors import shift as shift_lib
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math.numeric import clip_by_value_preserve_gradient

from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'AutoregressiveNetwork',
    'MaskedAutoregressiveFlow',
    'masked_autoregressive_default_template',
    'masked_dense',
]


class MaskedAutoregressiveFlow(bijector_lib.Bijector):
  """Affine MaskedAutoregressiveFlow bijector.

  The affine autoregressive flow [(Papamakarios et al., 2016)][3] provides a
  relatively simple framework for user-specified (deep) architectures to learn a
  distribution over continuous events.  Regarding terminology,

    'Autoregressive models decompose the joint density as a product of
    conditionals, and model each conditional in turn.  Normalizing flows
    transform a base density (e.g. a standard Gaussian) into the target density
    by an invertible transformation with tractable Jacobian.'
    [(Papamakarios et al., 2016)][3]

  In other words, the 'autoregressive property' is equivalent to the
  decomposition, `p(x) = prod{ p(x[perm[i]] | x[perm[0:i]]) : i=0, ..., d }`
  where `perm` is some permutation of `{0, ..., d}`.  In the simple case where
  the permutation is identity this reduces to:
  `p(x) = prod{ p(x[i] | x[0:i]) : i=0, ..., d }`.

  In TensorFlow Probability, 'normalizing flows' are implemented as
  `tfp.bijectors.Bijector`s.  The `forward` 'autoregression' is implemented
  using a `tf.while_loop` and a deep neural network (DNN) with masked weights
  such that the autoregressive property is automatically met in the `inverse`.

  A `TransformedDistribution` using `MaskedAutoregressiveFlow(...)` uses the
  (expensive) forward-mode calculation to draw samples and the (cheap)
  reverse-mode calculation to compute log-probabilities.  Conversely, a
  `TransformedDistribution` using `Invert(MaskedAutoregressiveFlow(...))` uses
  the (expensive) forward-mode calculation to compute log-probabilities and the
  (cheap) reverse-mode calculation to compute samples.  See 'Example Use'
  [below] for more details.

  Given a `shift_and_log_scale_fn`, the forward and inverse transformations are
  (a sequence of) affine transformations.  A 'valid' `shift_and_log_scale_fn`
  must compute each `shift` (aka `loc` or 'mu' in [Germain et al. (2015)][1])
  and `log(scale)` (aka 'alpha' in [Germain et al. (2015)][1]) such that each
  are broadcastable with the arguments to `forward` and `inverse`, i.e., such
  that the calculations in `forward`, `inverse` [below] are possible.

  For convenience, `tfp.bijectors.AutoregressiveNetwork` is offered as a
  possible `shift_and_log_scale_fn` function.  It implements the MADE
  architecture [(Germain et al., 2015)][1].  MADE is a feed-forward network that
  computes a `shift` and `log(scale)` using masked dense layers in a deep
  neural network. Weights are masked to ensure the autoregressive property. It
  is possible that this architecture is suboptimal for your task. To build
  alternative networks, either change the arguments to
  `tfp.bijectors.AutoregressiveNetwork` or use some other architecture, e.g.,
  using `tf.keras.layers`.

  Warning: no attempt is made to validate that the `shift_and_log_scale_fn`
  enforces the 'autoregressive property'.

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

  Notice that the `inverse` does not need a for-loop.  This is because in the
  forward pass each calculation of `shift` and `log_scale` is based on the `y`
  calculated so far (not `x`).  In the `inverse`, the `y` is fully known, thus
  is equivalent to the scaling used in `forward` after `event_size` passes,
  i.e., the 'last' `y` used to compute `shift`, `log_scale`.  (Roughly speaking,
  this also proves the transform is bijective.)

  The `bijector_fn` argument allows specifying a more general coupling relation,
  such as the LSTM-inspired activation from [4], or Neural Spline Flow [5].  It
  must logically operate on each element of the input individually, and still
  obey the 'autoregressive property' described above.  The forward
  transformation is

  ```python
  def forward(x):
    y = zeros_like(x)
    event_size = x.shape[-event_dims:].num_elements()
    for _ in range(event_size):
      bijector = bijector_fn(y)
      y = bijector.forward(x)
    return y
  ```

  and inverse transformation is

  ```python
  def inverse(y):
      bijector = bijector_fn(y)
      return bijector.inverse(y)
  ```

  #### Examples

  ```python
  tfd = tfp.distributions
  tfb = tfp.bijectors

  dims = 2

  # A common choice for a normalizing flow is to use a Gaussian for the base
  # distribution.  (However, any continuous distribution would work.) Here, we
  # use `tfd.Sample` to create a joint Gaussian distribution with diagonal
  # covariance for the base distribution (note that in the Gaussian case,
  # `tfd.MultivariateNormalDiag` could also be used.)
  maf = tfd.TransformedDistribution(
      distribution=tfd.Sample(
          tfd.Normal(loc=0., scale=1.), sample_shape=[dims]),
      bijector=tfb.MaskedAutoregressiveFlow(
          shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
              params=2, hidden_units=[512, 512])))

  x = maf.sample()  # Expensive; uses `tf.while_loop`, no Bijector caching.
  maf.log_prob(x)   # Almost free; uses Bijector caching.
  # Cheap; no `tf.while_loop` despite no Bijector caching.
  maf.log_prob(tf.zeros(dims))

  # [Papamakarios et al. (2016)][3] also describe an Inverse Autoregressive
  # Flow [(Kingma et al., 2016)][2]:
  iaf = tfd.TransformedDistribution(
      distribution=tfd.Sample(
          tfd.Normal(loc=0., scale=1.), sample_shape=[dims]),
      bijector=tfb.Invert(tfb.MaskedAutoregressiveFlow(
          shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
              params=2, hidden_units=[512, 512]))))

  x = iaf.sample()  # Cheap; no `tf.while_loop` despite no Bijector caching.
  iaf.log_prob(x)   # Almost free; uses Bijector caching.
  # Expensive; uses `tf.while_loop`, no Bijector caching.
  iaf.log_prob(tf.zeros(dims))

  # In many (if not most) cases the default `shift_and_log_scale_fn` will be a
  # poor choice.  Here's an example of using a 'shift only' version and with a
  # different number/depth of hidden layers.
  made = tfb.AutoregressiveNetwork(params=1, hidden_units=[32])
  maf_no_scale_hidden2 = tfd.TransformedDistribution(
      distribution=tfd.Sample(
          tfd.Normal(loc=0., scale=1.), sample_shape=[dims]),
      bijector=tfb.MaskedAutoregressiveFlow(
          lambda y: (made(y)[..., 0], None),
          is_constant_jacobian=True))
  maf_no_scale_hidden2._made = made  # Ensure maf_no_scale_hidden2.trainable
  # NOTE: The last line ensures that maf_no_scale_hidden2.trainable_variables
  # will include all variables from `made`.
  ```

  #### Variable Tracking

  NOTE: Like all subclasses of `tfb.Bijector`, `tfb.MaskedAutoregressiveFlow`
  subclasses `tf.Module` for variable tracking.

  A `tfb.MaskedAutoregressiveFlow` instance saves a reference to the values
  passed as `shift_and_log_scale_fn` and `bijector_fn` to its constructor.
  Thus, for most values passed as `shift_and_log_scale_fn` or `bijector_fn`,
  variables referenced by those values will be found and tracked by the
  `tfb.MaskedAutoregressiveFlow` instance.  Please see the `tf.Module`
  documentation for further details.

  However, if the value passed to `shift_and_log_scale_fn` or `bijector_fn` is a
  Python function, then `tfb.MaskedAutoregressiveFlow` cannot automatically
  track variables used inside `shift_and_log_scale_fn` or `bijector_fn`.  To get
  `tfb.MaskedAutoregressiveFlow` to track such variables, either:

   1. Replace the Python function with a `tf.Module`, `tf.keras.Layer`,
      or other callable object through which `tf.Module` can find variables.

   2. Or, add a reference to the variables to the `tfb.MaskedAutoregressiveFlow`
      instance by setting an attribute -- for example:
      ````
      made1 = tfb.AutoregressiveNetwork(params=1, hidden_units=[10, 10])
      made2 = tfb.AutoregressiveNetwork(params=1, hidden_units=[10, 10])
      maf = tfb.MaskedAutoregressiveFlow(lambda y: (made1(y), made2(y) + 1.))
      maf._made_variables = made1.variables + made2.variables
      ````

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

  [4]: Diederik P Kingma, Tim Salimans, Max Welling. Improving Variational
       Inference with Inverse Autoregressive Flow. In _Neural Information
       Processing Systems_, 2016. https://arxiv.org/abs/1606.04934

  [5]: Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios. Neural
       Spline Flows, 2019. http://arxiv.org/abs/1906.04032
  """

  def __init__(self,
               shift_and_log_scale_fn=None,
               bijector_fn=None,
               is_constant_jacobian=False,
               validate_args=False,
               unroll_loop=False,
               event_ndims=1,
               name=None):
    """Creates the MaskedAutoregressiveFlow bijector.

    Args:
      shift_and_log_scale_fn: Python `callable` which computes `shift` and
        `log_scale` from the inverse domain (`y`). Calculation must respect the
        'autoregressive property' (see class docstring). Suggested default
        `tfb.AutoregressiveNetwork(params=2, hidden_layers=...)`.
        Typically the function contains `tf.Variables`. Returning `None` for
        either (both) `shift`, `log_scale` is equivalent to (but more efficient
        than) returning zero. If `shift_and_log_scale_fn` returns a single
        `Tensor`, the returned value will be unstacked to get the `shift` and
        `log_scale`: `tf.unstack(shift_and_log_scale_fn(y), num=2, axis=-1)`.
      bijector_fn: Python `callable` which returns a `tfb.Bijector` which
        transforms event tensor with the signature
        `(input, **condition_kwargs) -> bijector`. The bijector must operate on
        scalar events and must not alter the rank of its input. The
        `bijector_fn` will be called with `Tensors` from the inverse domain
        (`y`). Calculation must respect the 'autoregressive property' (see
        class docstring).
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
        implemented by the `tfp.bijectors.AutoregressiveNetwork`, 2 might be
        useful for a 2D convolutional `shift_and_log_scale_fn` and so on.
      name: Python `str`, name given to ops managed by this object.

    Raises:
      ValueError: If both or none of `shift_and_log_scale_fn` and `bijector_fn`
          are specified.
    """
    parameters = dict(locals())
    name = name or 'masked_autoregressive_flow'
    with tf.name_scope(name) as name:
      self._unroll_loop = unroll_loop
      self._event_ndims = event_ndims
      if bool(shift_and_log_scale_fn) == bool(bijector_fn):
        raise ValueError('Exactly one of `shift_and_log_scale_fn` and '
                         '`bijector_fn` should be specified.')
      if shift_and_log_scale_fn:
        def _bijector_fn(x, **condition_kwargs):
          params = shift_and_log_scale_fn(x, **condition_kwargs)
          if tf.is_tensor(params):
            shift, log_scale = tf.unstack(params, num=2, axis=-1)
          else:
            shift, log_scale = params

          bijectors = []
          if shift is not None:
            bijectors.append(shift_lib.Shift(shift))
          if log_scale is not None:
            bijectors.append(scale_lib.Scale(log_scale=log_scale))
          return chain.Chain(bijectors)

        bijector_fn = _bijector_fn

      if validate_args:
        bijector_fn = _validate_bijector_fn(bijector_fn)
      # Still do this assignment for variable tracking.
      self._shift_and_log_scale_fn = shift_and_log_scale_fn
      self._bijector_fn = bijector_fn
      super().__init__(
          forward_min_event_ndims=self._event_ndims,
          is_constant_jacobian=is_constant_jacobian,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  def _forward(self, x, **kwargs):
    static_event_size = tensorshape_util.num_elements(
        tensorshape_util.with_rank_at_least(
            x.shape, self._event_ndims)[-self._event_ndims:])

    if self._unroll_loop:
      if not static_event_size:
        raise ValueError(
            'The final {} dimensions of `x` must be known at graph '
            'construction time if `unroll_loop=True`. `x.shape: {!r}`'.format(
                self._event_ndims, x.shape))
      y = tf.zeros_like(x, name='y0')

      for _ in range(static_event_size):
        y = self._bijector_fn(y, **kwargs).forward(x)
      return y

    event_size = ps.reduce_prod(ps.shape(x)[-self._event_ndims:])
    y0 = tf.zeros_like(x, name='y0')
    # call the template once to ensure creation
    if not tf.executing_eagerly():
      _ = self._bijector_fn(y0, **kwargs).forward(y0)
    def _loop_body(y0):
      """While-loop body for autoregression calculation."""
      # Set caching device to avoid re-getting the tf.Variable for every while
      # loop iteration.
      with tf1.variable_scope(tf1.get_variable_scope()) as vs:
        if vs.caching_device is None and not tf.executing_eagerly():
          vs.set_caching_device(lambda op: op.device)
        bijector = self._bijector_fn(y0, **kwargs)
      y = bijector.forward(x)
      return (y,)
    (y,) = tf.while_loop(
        cond=lambda _: True,
        body=_loop_body,
        loop_vars=(y0,),
        maximum_iterations=event_size)
    return y

  def _inverse(self, y, **kwargs):
    bijector = self._bijector_fn(y, **kwargs)
    return bijector.inverse(y)

  def _inverse_log_det_jacobian(self, y, **kwargs):
    return self._bijector_fn(y, **kwargs).inverse_log_det_jacobian(
        y, event_ndims=self._event_ndims)


MASK_INCLUSIVE = 'inclusive'
MASK_EXCLUSIVE = 'exclusive'


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
  mask = np.zeros([n_out, n_in], dtype=dtype_util.as_numpy_dtype(dtype))
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
        'Rightmost dimension must be known prior to graph execution.')

  mask = _gen_mask(num_blocks, input_depth, units,
                   MASK_EXCLUSIVE if exclusive else MASK_INCLUSIVE).T

  if kernel_initializer is None:
    kernel_initializer = tf1.glorot_normal_initializer()

  def masked_initializer(shape, dtype=None, partition_info=None):
    return mask * kernel_initializer(shape, dtype, partition_info)

  with tf.name_scope(name or 'masked_dense'):
    layer = tf1.layers.Dense(
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


@deprecation.deprecated(
    '2020-02-01',
    '`masked_autoregressive_default_template` is deprecated; '
    'use `tfp.bijectors.AutoregressiveNetwork`.  '
    ' Also, please note the section "Variable Tracking" in the documentation '
    'for `tfp.bijectors.MaskedAutoregressiveFlow`.',
    warn_once=True)
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
  created once. It takes the input and returns the `loc` ('mu' in [Germain et
  al. (2015)][1]) and `log_scale` ('alpha' in [Germain et al. (2015)][1]) from
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
      'masked_autoregressive_default_template'.
    *args: `tf.layers.dense` arguments.
    **kwargs: `tf.layers.dense` keyword arguments.

  Returns:
    shift: `Float`-like `Tensor` of shift terms (the 'mu' in
      [Germain et al.  (2015)][1]).
    log_scale: `Float`-like `Tensor` of log(scale) terms (the 'alpha' in
      [Germain et al. (2015)][1]).

  Raises:
    NotImplementedError: if rightmost dimension of `inputs` is unknown prior to
      graph execution.

  #### References

  [1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
       Masked Autoencoder for Distribution Estimation. In _International
       Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509
  """
  name = name or 'masked_autoregressive_default_template'
  with tf.name_scope(name):

    def _fn(x):
      """MADE parameterized via `masked_autoregressive_default_template`."""
      # TODO(b/67594795): Better support of dynamic shape.
      input_depth = tf.compat.dimension_value(
          tensorshape_util.with_rank_at_least(x.shape, 1)[-1])
      if input_depth is None:
        raise NotImplementedError(
            'Rightmost dimension must be known prior to graph execution.')
      input_shape = (
          np.int32(tensorshape_util.as_list(x.shape))
          if tensorshape_util.is_fully_defined(x.shape) else tf.shape(x))
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

    return tf1.make_template(name, _fn)


class AutoregressiveNetwork(tf.keras.layers.Layer):
  r"""Masked Autoencoder for Distribution Estimation [Germain et al. (2015)][1].

  A `AutoregressiveNetwork` takes as input a Tensor of shape `[..., event_size]`
  and returns a Tensor of shape `[..., event_size, params]`.

  The output satisfies the autoregressive property.  That is, the layer is
  configured with some permutation `ord` of `{0, ..., event_size-1}` (i.e., an
  ordering of the input dimensions), and the output `output[batch_idx, i, ...]`
  for input dimension `i` depends only on inputs `x[batch_idx, j]` where
  `ord(j) < ord(i)`.  The autoregressive property allows us to use
  `output[batch_idx, i]` to parameterize conditional distributions:
    `p(x[batch_idx, i] | x[batch_idx, j] for ord(j) < ord(i))`
  which give us a tractable distribution over input `x[batch_idx]`:
    `p(x[batch_idx]) = prod_i p(x[batch_idx, ord(i)] | x[batch_idx, ord(0:i)])`

  For example, when `params` is 2, the output of the layer can parameterize
  the location and log-scale of an autoregressive Gaussian distribution.

  #### Example

  The `AutoregressiveNetwork` can be used to do density estimation as is shown
  in the below example:

  ```python
  # Generate data -- as in Figure 1 in [Papamakarios et al. (2017)][2]).
  n = 2000
  x2 = np.random.randn(n).astype(dtype=np.float32) * 2.
  x1 = np.random.randn(n).astype(dtype=np.float32) + (x2 * x2 / 4.)
  data = np.stack([x1, x2], axis=-1)

  # Density estimation with MADE.
  made = tfb.AutoregressiveNetwork(params=2, hidden_units=[10, 10])

  distribution = tfd.TransformedDistribution(
      distribution=tfd.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=[2]),
      bijector=tfb.MaskedAutoregressiveFlow(made))

  # Construct and fit model.
  x_ = tfkl.Input(shape=(2,), dtype=tf.float32)
  log_prob_ = distribution.log_prob(x_)
  model = tfk.Model(x_, log_prob_)

  model.compile(optimizer=tf.optimizers.Adam(),
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

  The `conditional` argument can be used to instead build a conditional density
  estimator. To do this the conditioning variable must be passed as a `kwarg`:

  ```python
  # Generate data as the mixture of two distributions.
  n = 2000
  c = np.r_[
    np.zeros(n//2),
    np.ones(n//2)
  ]
  mean_0, mean_1 = 0, 5
  x = np.r_[
    np.random.randn(n//2).astype(dtype=np.float32) + mean_0,
    np.random.randn(n//2).astype(dtype=np.float32) + mean_1
  ]

  # Density estimation with MADE.
  made = tfb.AutoregressiveNetwork(
    params=2,
    hidden_units=[2, 2],
    event_shape=(1,),
    conditional=True,
    kernel_initializer=tfk.initializers.VarianceScaling(0.1),
    conditional_event_shape=(1,)
  )

  distribution = tfd.TransformedDistribution(
    distribution=tfd.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=[1]),
    bijector=tfb.MaskedAutoregressiveFlow(made))

  # Construct and fit model.
  x_ = tfkl.Input(shape=(1,), dtype=tf.float32)
  c_ = tfkl.Input(shape=(1,), dtype=tf.float32)
  log_prob_ = distribution.log_prob(
    x_, bijector_kwargs={'conditional_input': c_})
  model = tfk.Model([x_, c_], log_prob_)

  model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1),
                loss=lambda _, log_prob: -log_prob)

  batch_size = 25
  model.fit(x=[x, c],
            y=np.zeros((n, 0), dtype=np.float32),
            batch_size=batch_size,
            epochs=3,
            steps_per_epoch=n // batch_size,
            shuffle=True,
            verbose=True)

  # Use the fitted distribution to sample condition on c = 1
  n_samples = 1000
  cond = 1
  samples = distribution.sample(
    (n_samples,),
    bijector_kwargs={'conditional_input': cond * np.ones((n_samples, 1))})
  ```

  #### Examples: Handling Rank-2+ Tensors

  `AutoregressiveNetwork` can be used as a building block to achieve different
  autoregressive structures over rank-2+ tensors.  For example, suppose we want
  to build an autoregressive distribution over images with dimension `[weight,
  height, channels]` with `channels = 3`:

   1. We can parameterize a 'fully autoregressive' distribution, with
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

      # Density estimation with MADE.
      made = tfb.AutoregressiveNetwork(params=2, event_shape=event_shape,
                                       hidden_units=[20, 20], activation='relu')
      distribution = tfd.TransformedDistribution(
      distribution=tfd.Sample(
          tfd.Normal(loc=0., scale=1.), sample_shape=[dims]),
      bijector=tfb.MaskedAutoregressiveFlow(made))

      # Construct and fit model.
      x_ = tfkl.Input(shape=event_shape, dtype=tf.float32)
      log_prob_ = distribution.log_prob(x_)
      model = tfk.Model(x_, log_prob_)

      model.compile(optimizer=tf.optimizers.Adam(),
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

      made = tfb.AutoregressiveNetwork(params=1, event_shape=[width * height],
                                       hidden_units=[20, 20], activation='relu')

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

      model.compile(optimizer=tf.optimizers.Adam(),
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
      `AutoregressiveNetwork` and `TransformedDistribution` for each channel,
      and combine them with a `tfd.Blockwise` distribution.

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
               conditional=False,
               conditional_event_shape=None,
               conditional_input_layers='all_layers',
               hidden_units=None,
               input_order='left-to-right',
               hidden_degrees='equal',
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
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
      conditional: Python boolean describing whether to add conditional inputs.
      conditional_event_shape: Python `list`-like of positive integers (or a
        single int), specifying the shape of the conditional input to this layer
        (without the batch dimensions). This must be specified if `conditional`
        is `True`.
      conditional_input_layers: Python `str` describing how to add conditional
        parameters to the autoregressive network. When "all_layers" the
        conditional input will be combined with the network at every layer,
        whilst "first_layer" combines the conditional input only at the first
        layer which is then passed through the network
        autoregressively. Default: 'all_layers'.
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
      kernel_initializer: Initializer for the `Dense` kernel weight
        matrices.  Default: 'glorot_uniform'.
      bias_initializer: Initializer for the `Dense` bias vectors. Default:
        'zeros'.
      kernel_regularizer: Regularizer function applied to the `Dense` kernel
        weight matrices.  Default: None.
      bias_regularizer: Regularizer function applied to the `Dense` bias
        weight vectors.  Default: None.
      kernel_constraint: Constraint function applied to the `Dense` kernel
        weight matrices.  Default: None.
      bias_constraint: Constraint function applied to the `Dense` bias
        weight vectors.  Default: None.
      validate_args: Python `bool`, default `False`. When `True`, layer
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      **kwargs: Additional keyword arguments passed to this layer (but not to
        the `tf.keras.layer.Dense` layers constructed by this layer).
    """
    super().__init__(**kwargs)

    self._params = params
    self._event_shape = _list(event_shape) if event_shape is not None else None
    self._conditional = conditional
    self._conditional_event_shape = (
        _list(conditional_event_shape)
        if conditional_event_shape is not None else None)
    self._conditional_layers = conditional_input_layers
    self._hidden_units = hidden_units if hidden_units is not None else []
    self._input_order_param = input_order
    self._hidden_degrees = hidden_degrees
    self._activation = activation
    self._use_bias = use_bias
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self._bias_constraint = bias_constraint
    self._validate_args = validate_args
    self._kwargs = kwargs

    if self._event_shape is not None:
      self._event_size = self._event_shape[-1]
      self._event_ndims = len(self._event_shape)

      if self._event_ndims != 1:
        raise ValueError('Parameter `event_shape` must describe a rank-1 '
                         'shape. `event_shape: {!r}`'.format(event_shape))

    if self._conditional:
      if self._event_shape is None:
        raise ValueError('`event_shape` must be provided when '
                         '`conditional` is True')
      if self._conditional_event_shape is None:
        raise ValueError('`conditional_event_shape` must be provided when '
                         '`conditional` is True')
      self._conditional_size = self._conditional_event_shape[-1]
      self._conditional_ndims = len(self._conditional_event_shape)
      if self._conditional_ndims != 1:
        raise ValueError('Parameter `conditional_event_shape` must describe a '
                         'rank-1 shape')
      if not ((self._conditional_layers == 'first_layer') or
              (self._conditional_layers == 'all_layers')):
        raise ValueError('`conditional_input_layers` must be '
                         '"first_layers" or "all_layers"')
    else:
      if self._conditional_event_shape is not None:
        raise ValueError('`conditional_event_shape` passed but `conditional` '
                         'is set to False.')

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
      raise ValueError('Invalid final dimension of `input_shape`. '
                       'Expected `{!r}`, but got `{!r}`'.format(
                           self._event_shape[-1], input_shape[-1]))

    # Construct the masks.
    self._input_order = _create_input_order(
        self._event_size,
        self._input_order_param,
    )
    self._masks = _make_dense_autoregressive_masks(
        params=self._params,
        event_size=self._event_size,
        hidden_units=self._hidden_units,
        input_order=self._input_order,
        hidden_degrees=self._hidden_degrees,
    )

    outputs = [tf.keras.Input((self._event_size,), dtype=self.dtype)]
    inputs = outputs[0]
    if self._conditional:
      conditional_input = tf.keras.Input((self._conditional_size,),
                                         dtype=self.dtype)
      inputs = [inputs, conditional_input]

    # Input-to-hidden, hidden-to-hidden, and hidden-to-output layers:
    #  [..., self._event_size] -> [..., self._hidden_units[0]].
    #  [..., self._hidden_units[k-1]] -> [..., self._hidden_units[k]].
    #  [..., self._hidden_units[-1]] -> [..., event_size * self._params].
    layer_output_sizes = self._hidden_units + [self._event_size * self._params]
    for k in range(len(self._masks)):
      autoregressive_output = tf.keras.layers.Dense(
          layer_output_sizes[k],
          activation=None,
          use_bias=self._use_bias,
          kernel_initializer=_make_masked_initializer(
              self._masks[k], self._kernel_initializer),
          bias_initializer=self._bias_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          kernel_constraint=_make_masked_constraint(
              self._masks[k], self._kernel_constraint),
          bias_constraint=self._bias_constraint,
          dtype=self.dtype)(outputs[-1])
      if (self._conditional and
          ((self._conditional_layers == 'all_layers') or
           ((self._conditional_layers == 'first_layer') and (k == 0)))):
        conditional_output = tf.keras.layers.Dense(
            layer_output_sizes[k],
            activation=None,
            use_bias=False,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=None,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=None,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=None,
            dtype=self.dtype)(conditional_input)
        outputs.append(tf.keras.layers.Add()([
            autoregressive_output,
            conditional_output]))
      else:
        outputs.append(autoregressive_output)
      if k + 1 < len(self._masks):
        outputs.append(
            tf.keras.layers.Activation(self._activation)
            (outputs[-1]))
    self._network = tf.keras.models.Model(
        inputs=inputs,
        outputs=outputs[-1])
    # Allow network to be called with inputs of shapes that don't match
    # the specs of the network's input layers.
    self._network.input_spec = None
    # Record that the layer has been built.
    super().build(input_shape)

  def call(self, x, conditional_input=None):
    """Transforms the inputs and returns the outputs.

    Suppose `x` has shape `batch_shape + event_shape` and `conditional_input`
    has shape `conditional_batch_shape + conditional_event_shape`. Then, the
    output shape is:
    `broadcast(batch_shape, conditional_batch_shape) + event_shape + [params]`.

    Also see `tfkl.Layer.call` for some generic discussion about Layer calling.

    Args:
      x: A `Tensor`. Primary input to the layer.
      conditional_input: A `Tensor. Conditional input to the layer. This is
        required iff the layer is conditional.

    Returns:
      y: A `Tensor`. The output of the layer. Note that the leading dimensions
         follow broadcasting rules described above.
    """
    with tf.name_scope(self.name or 'AutoregressiveNetwork_call'):
      x = tf.convert_to_tensor(x, dtype=self.dtype, name='x')
      # TODO(b/67594795): Better support for dynamic shapes.
      input_shape = ps.shape(x)
      if tensorshape_util.rank(x.shape) == 1:
        x = x[tf.newaxis, ...]
      if self._conditional:
        if conditional_input is None:
          raise ValueError('`conditional_input` must be passed as a named '
                           'argument')
        conditional_input = tf.convert_to_tensor(
            conditional_input, dtype=self.dtype, name='conditional_input')
        conditional_batch_shape = ps.shape(conditional_input)[:-1]
        if tensorshape_util.rank(conditional_input.shape) == 1:
          conditional_input = conditional_input[tf.newaxis, ...]
        x = [x, conditional_input]
        output_shape = ps.concat(
            [ps.broadcast_shape(conditional_batch_shape,
                                input_shape[:-1]),
             input_shape[-1:]], axis=0)
      else:
        output_shape = input_shape
      return tf.reshape(self._network(x),
                        tf.concat([output_shape, [self._params]], axis=0))

  def compute_output_shape(self, input_shape):
    """See tfkl.Layer.compute_output_shape."""
    return input_shape + (self._params,)

  @property
  def event_shape(self):
    return self._event_shape

  @property
  def params(self):
    return self._params


def _make_dense_autoregressive_masks(
    params,
    event_size,
    hidden_units,
    input_order='left-to-right',
    hidden_degrees='equal',
    seed=None,
):
  """Creates masks for use in dense MADE [Germain et al. (2015)][1] networks.

  See the documentation for `AutoregressiveNetwork` for the theory and
  application of MADE networks. This function lets you construct your own dense
  MADE networks by applying the returned masks to each dense layer. E.g. a
  consider an autoregressive network that takes `event_size`-dimensional vectors
  and produces `params`-parameters per input, with `num_hidden` hidden layers,
  with `hidden_size` hidden units each.

  ```python
  def random_made(x):
    masks = tfb._make_dense_autoregressive_masks(
        params=params,
        event_size=event_size,
        hidden_units=[hidden_size] * num_hidden)
    output_sizes = [hidden_size] * num_hidden
    input_size = event_size
    for (mask, output_size) in zip(masks, output_sizes):
      mask = tf.cast(mask, tf.float32)
      x = tf.matmul(x, tf.random.normal([input_size, output_size]) * mask)
      x = tf.nn.relu(x)
      input_size = output_size
    x = tf.matmul(
        x,
        tf.random.normal([input_size, params * event_size]) * masks[-1])
    x = tf.reshape(x, [-1, event_size, params])
    return x

  y = random_made(tf.zeros([1, event_size]))
  assert [1, event_size, params] == y.shape
  ```

  Each mask is a Numpy boolean array. All masks have the shape `[input_size,
  output_size]`. For example, if we `hidden_units` is a list of two integers,
  the mask shapes will be: `[event_size, hidden_units[0]], [hidden_units[0],
  hidden_units[1]], [hidden_units[1], params * event_size]`.

  You can extend this example with trainable parameters and constraints if
  necessary.

  Args:
    params: Python integer specifying the number of parameters to output
      per input.
    event_size: Python integer specifying the shape of the input to this layer.
    hidden_units: Python `list`-like of non-negative integers, specifying
      the number of units in each hidden layer.
    input_order: Order of degrees to the input units: 'random', 'left-to-right',
      'right-to-left', or an array of an explicit order. For example,
      'left-to-right' builds an autoregressive model
      p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
    hidden_degrees: Method for assigning degrees to the hidden units:
      'equal', 'random'. If 'equal', hidden units in each layer are allocated
      equally (up to a remainder term) to each degree. Default: 'equal'.
    seed: If not `None`, seed to use for 'random' `input_order` and
      `hidden_degrees`.

  Returns:
    masks: A list of masks that should be applied the dense matrices of
      individual densely connected layers in the MADE network. Each mask is a
      Numpy boolean array.

  #### References

  [1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
       Masked Autoencoder for Distribution Estimation. In _International
       Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509
  """
  if seed is None:
    input_order_seed = None
    degrees_seed = None
  else:
    input_order_seed, degrees_seed = np.random.RandomState(seed).randint(
        2**31, size=2)
  input_order = _create_input_order(
      event_size, input_order, seed=input_order_seed)
  masks = _create_masks(_create_degrees(
      input_size=event_size,
      hidden_units=hidden_units,
      input_order=input_order,
      hidden_degrees=hidden_degrees,
      seed=degrees_seed))
  # In the final layer, we will produce `params` outputs for each of the
  # `event_size` inputs.  But `masks[-1]` has shape `[hidden_units[-1],
  # event_size]`.  Thus, we need to expand the mask to `[hidden_units[-1],
  # event_size * params]` such that all units for the same input are masked
  # identically.  In particular, we tile the mask so the j-th element of
  # `tf.unstack(output, axis=-1)` is a tensor of the j-th parameter/unit for
  # each input.
  #
  # NOTE: Other orderings of the output could be faster -- should benchmark.
  masks[-1] = np.reshape(
      np.tile(masks[-1][..., tf.newaxis], [1, 1, params]),
      [masks[-1].shape[0], event_size * params])
  return masks


def _list(xs):
  """Convert the given argument to a list."""
  try:
    return list(xs)
  except TypeError:
    return [xs]


def _create_input_order(input_size, input_order='left-to-right', seed=None):
  """Returns a degree vectors for the input."""
  if isinstance(input_order, six.string_types):
    if input_order == 'left-to-right':
      return np.arange(start=1, stop=input_size + 1)
    elif input_order == 'right-to-left':
      return np.arange(start=input_size, stop=0, step=-1)
    elif input_order == 'random':
      ret = np.arange(start=1, stop=input_size + 1)
      if seed is None:
        rng = np.random
      else:
        rng = np.random.RandomState(seed)
      rng.shuffle(ret)
      return ret
  elif np.all(np.sort(np.array(input_order)) == np.arange(1, input_size + 1)):
    return np.array(input_order)

  raise ValueError('Invalid input order: "{}".'.format(input_order))


def _create_degrees(input_size,
                    hidden_units=None,
                    input_order='left-to-right',
                    hidden_degrees='equal',
                    seed=None):
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
    seed: If not `None`, use as a seed for the 'random' hidden_degrees.

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
      if hidden_degrees == 'random':
        if seed is None:
          rng = np.random
        else:
          rng = np.random.RandomState(seed)
        # samples from: [low, high)
        degrees.append(
            rng.randint(low=min(np.min(degrees[-1]), input_size - 1),
                        high=input_size,
                        size=units))
      elif hidden_degrees == 'equal':
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
    # `initializer` may be a `tf.initializers.Initializer` (which don't accept a
    # `partition_info` argument).
    if partition_info is None:
      x = initializer(shape, dtype)
    else:
      x = initializer(shape, dtype, partition_info)
    return tf.cast(mask, x.dtype) * x
  return masked_initializer


def _make_masked_constraint(mask, constraint=None):
  constraint = tf.keras.constraints.get(constraint)
  def masked_constraint(x):
    x = tf.convert_to_tensor(x, dtype_hint=tf.float32, name='x')
    if constraint is not None:
      x = constraint(x)
    return tf.cast(mask, x.dtype) * x
  return masked_constraint


def _validate_bijector_fn(bijector_fn):
  """Validates the output of `bijector_fn`."""

  def _wrapper(x, **condition_kwargs):
    """A wrapper that validates `bijector_fn`."""
    bijector = bijector_fn(x, **condition_kwargs)
    if bijector.forward_min_event_ndims != bijector.inverse_min_event_ndims:
      # Current code won't really work with this, but in principle we could
      # implement this.
      raise ValueError('Bijectors which alter `event_ndims` are not supported.')
    if bijector.forward_min_event_ndims > 0:
      # Mustn't break auto-regressivity,
      raise ValueError(
          'Bijectors with `forward_min_event_ndims` > 0 are not supported.')
    return bijector

  return _wrapper
