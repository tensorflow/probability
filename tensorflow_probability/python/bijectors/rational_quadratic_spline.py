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
"""Piecewise Rational Quadratic Spline bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


def _ensure_at_least_1d(t):
  t = tf.convert_to_tensor(t)
  return t + tf.zeros([1], dtype=t.dtype)


def _padded(t, lhs, rhs=None):
  """Left pads and optionally right pads the innermost axis of `t`."""
  lhs = tf.convert_to_tensor(lhs, dtype=t.dtype)
  zeros = tf.zeros([tf.rank(t) - 1, 2], dtype=tf.int32)
  lhs_paddings = tf.concat([zeros, [[1, 0]]], axis=0)
  result = tf.pad(t, paddings=lhs_paddings, constant_values=lhs)
  if rhs is not None:
    rhs = tf.convert_to_tensor(rhs, dtype=t.dtype)
    rhs_paddings = tf.concat([zeros, [[0, 1]]], axis=0)
    result = tf.pad(result, paddings=rhs_paddings, constant_values=rhs)
  return result


def _knot_positions(bin_sizes, range_min):
  return _padded(tf.cumsum(bin_sizes, axis=-1) + range_min, lhs=range_min)


_SplineShared = collections.namedtuple(
    'SplineShared', 'out_of_bounds,x_k,y_k,d_k,d_kp1,h_k,w_k,s_k')


class RationalQuadraticSpline(bijector.Bijector):
  """A piecewise rational quadratic spline, as developed in [1].

  This transformation represents a monotonically increasing piecewise rational
  quadratic function. Outside of the bounds of `knot_x`/`knot_y`, the transform
  behaves as an identity function.

  Typically this bijector will be used as part of a chain, with splines for
  trailing `x` dimensions conditioned on some of the earlier `x` dimensions, and
  with the inverse then solved first for unconditioned dimensions, then using
  conditioning derived from those inverses, and so forth. For example, if we
  split a 15-D `xs` vector into 3 components, we may implement a forward and
  inverse as follows:

  ```python
  nsplits = 3

  class SplineParams(tf.Module):

    def __init__(self, nbins=32):
      self._nbins = nbins
      self._built = False
      self._bin_widths = None
      self._bin_heights = None
      self._knot_slopes = None

    def _bin_positions(self, x):
      x = tf.reshape(x, [-1, self._nbins])
      return tf.math.softmax(x, axis=-1) * (2 - self._nbins * 1e-2) + 1e-2

    def _slopes(self, x):
      x = tf.reshape(x, [-1, self._nbins - 1])
      return tf.math.softplus(x) + 1e-2

    def __call__(self, x, nunits):
      if not self._built:
        self._bin_widths = tf.keras.layers.Dense(
            nunits * self._nbins, activation=self._bin_positions, name='w')
        self._bin_heights = tf.keras.layers.Dense(
            nunits * self._nbins, activation=self._bin_positions, name='h')
        self._knot_slopes = tf.keras.layers.Dense(
            nunits * (self._nbins - 1), activation=self._slopes, name='s')
        self._built = True
      return tfb.RationalQuadraticSpline(
          bin_widths=self._bin_widths(x),
          bin_heights=self._bin_heights(x),
          knot_slopes=self._knot_slopes(x))

  xs = np.random.randn(1, 15).astype(np.float32)  # Keras won't Dense(.)(vec).
  splines = [SplineParams() for _ in range(nsplits)]

  def spline_flow():
    stack = tfb.Identity()
    for i in range(nsplits):
      stack = tfb.RealNVP(5 * i, bijector_fn=splines[i])(stack)
    return stack

  ys = spline_flow().forward(xs)
  ys_inv = spline_flow().inverse(ys)  # ys_inv ~= xs
  ```

  For a one-at-a-time autoregressive flow as in [1], it would be profitable to
  implement a mask over `xs` to parallelize either the inverse or the forward
  pass and implement the other using a `tf.while_loop`. See
  `tfp.bijectors.MaskedAutoregressiveFlow` for support doing so (paired with
  `tfp.bijectors.Invert` depending which direction should be parallel).

  #### References

  [1]: Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios. Neural
       Spline Flows. _arXiv preprint arXiv:1906.04032_, 2019.
       https://arxiv.org/abs/1906.04032
  """

  def __init__(self,
               bin_widths,
               bin_heights,
               knot_slopes,
               range_min=-1,
               validate_args=False,
               name=None):
    """Construct a new RationalQuadraticSpline bijector.

    For each argument, the innermost axis indexes bins/knots and batch axes
    index axes of `x`/`y` spaces. A `RationalQuadraticSpline` with a separate
    transform for each of three dimensions might have `bin_widths` shaped
    `[3, 32]`. To use the same spline for each of `x`'s three dimensions we may
    broadcast against `x` and use a `bin_widths` parameter shaped `[32]`.
    Parameters will be broadcast against each other and against the input
    `x`/`y`s, so if we want fixed slopes, we can use kwarg `knot_slopes=1`.

    A typical recipe for acquiring compatible bin widths and heights would be:

    ```python
    nbins = unconstrained_vector.shape[-1]
    range_min, range_max, min_bin_size = -1, 1, 1e-2
    scale = range_max - range_min - nbins * min_bin_size
    bin_widths = tf.math.softmax(unconstrained_vector) * scale + min_bin_size
    ```

    Args:
      bin_widths: The widths of the spans between subsequent knot `x` positions,
        a floating point `Tensor`. Must be positive, and at least 1-D. Innermost
        axis must sum to the same value as `bin_heights`. The knot `x` positions
        will be a first at `range_min`, followed by knots at `range_min +
        cumsum(bin_widths, axis=-1)`.
      bin_heights: The heights of the spans between subsequent knot `y`
        positions, a floating point `Tensor`. Must be positive, and at least
        1-D. Innermost axis must sum to the same value as `bin_widths`. The knot
        `y` positions will be a first at `range_min`, followed by knots at
        `range_min + cumsum(bin_heights, axis=-1)`.
      knot_slopes: The slope of the spline at each knot, a floating point
        `Tensor`. Must be positive. `1`s are implicitly padded for the first and
        last implicit knots corresponding to `range_min` and `range_min +
        sum(bin_widths, axis=-1)`. Innermost axis size should be 1 less than
        that of `bin_widths`/`bin_heights`, or 1 for broadcasting.
      range_min: The `x`/`y` position of the first knot, which has implicit
        slope `1`. `range_max` is implicit, and can be computed as `range_min +
        sum(bin_widths, axis=-1)`. Scalar floating point `Tensor`.
      validate_args: Toggles argument validation (can hurt performance).
      name: Optional name scope for associated ops. (Defaults to
        `'RationalQuadraticSpline'`).
    """
    parameters = dict(locals())
    with tf.name_scope(name or 'RationalQuadraticSpline') as name:
      dtype = dtype_util.common_dtype(
          [bin_widths, bin_heights, knot_slopes, range_min],
          dtype_hint=tf.float32)
      self._bin_widths = tensor_util.convert_nonref_to_tensor(
          bin_widths, dtype=dtype, name='bin_widths')
      self._bin_heights = tensor_util.convert_nonref_to_tensor(
          bin_heights, dtype=dtype, name='bin_heights')
      self._knot_slopes = tensor_util.convert_nonref_to_tensor(
          knot_slopes, dtype=dtype, name='knot_slopes')
      self._range_min = tensor_util.convert_nonref_to_tensor(
          range_min, dtype=dtype, name='range_min')
      super(RationalQuadraticSpline, self).__init__(
          dtype=dtype,
          forward_min_event_ndims=0,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @property
  def bin_widths(self):
    return self._bin_widths

  @property
  def bin_heights(self):
    return self._bin_heights

  @property
  def knot_slopes(self):
    return self._knot_slopes

  @property
  def range_min(self):
    return self._range_min

  @classmethod
  def _is_increasing(cls):
    return True

  def _compute_shared(self, x=None, y=None):
    """Captures shared computations across forward/inverse/logdet.

    Only one of `x` or `y` should be specified.

    Args:
      x: The `x` values we will search for.
      y: The `y` values we will search for.

    Returns:
      data: A namedtuple with named fields containing shared computations.
    """
    assert (x is None) != (y is None)
    is_x = x is not None

    range_min = tf.convert_to_tensor(self.range_min, name='range_min')
    kx = _knot_positions(self.bin_widths, range_min)
    ky = _knot_positions(self.bin_heights, range_min)
    kd = _padded(_ensure_at_least_1d(self.knot_slopes), lhs=1, rhs=1)
    kx_or_ky = kx if is_x else ky
    kx_or_ky_min = kx_or_ky[..., 0]
    kx_or_ky_max = kx_or_ky[..., -1]
    x_or_y = x if is_x else y
    out_of_bounds = (x_or_y <= kx_or_ky_min) | (x_or_y >= kx_or_ky_max)
    x_or_y = tf.where(out_of_bounds, kx_or_ky_min, x_or_y)

    shape = functools.reduce(
        tf.broadcast_dynamic_shape,
        (
            tf.shape(x_or_y[..., tf.newaxis]),  # Add a n_knots dim.
            tf.shape(kx),
            tf.shape(ky),
            tf.shape(kd)))

    bc_x_or_y = tf.broadcast_to(x_or_y, shape[:-1])
    bc_kx = tf.broadcast_to(kx, shape)
    bc_ky = tf.broadcast_to(ky, shape)
    bc_kd = tf.broadcast_to(kd, shape)
    bc_kx_or_ky = bc_kx if is_x else bc_ky
    indices = tf.maximum(
        tf.zeros([], dtype=tf.int64),
        tf.searchsorted(
            bc_kx_or_ky[..., :-1],
            bc_x_or_y[..., tf.newaxis],
            side='right',
            out_type=tf.int64) - 1)

    def gather_squeeze(params, indices):
      rank = tensorshape_util.rank(indices.shape)
      if rank is None:
        raise ValueError('`indices` must have statically known rank.')
      return tf.gather(params, indices, axis=-1, batch_dims=rank - 1)[..., 0]

    x_k = gather_squeeze(bc_kx, indices)
    x_kp1 = gather_squeeze(bc_kx, indices + 1)
    y_k = gather_squeeze(bc_ky, indices)
    y_kp1 = gather_squeeze(bc_ky, indices + 1)
    d_k = gather_squeeze(bc_kd, indices)
    d_kp1 = gather_squeeze(bc_kd, indices + 1)
    h_k = y_kp1 - y_k
    w_k = x_kp1 - x_k
    s_k = h_k / w_k

    return _SplineShared(
        out_of_bounds=out_of_bounds,
        x_k=x_k,
        y_k=y_k,
        d_k=d_k,
        d_kp1=d_kp1,
        h_k=h_k,
        w_k=w_k,
        s_k=s_k)

  def _forward(self, x):
    """Compute the forward transformation (Appendix A.1)."""
    d = self._compute_shared(x=x)
    relx = (x - d.x_k) / d.w_k
    spline_val = (
        d.y_k + ((d.h_k * (d.s_k * relx**2 + d.d_k * relx * (1 - relx))) /
                 (d.s_k + (d.d_kp1 + d.d_k - 2 * d.s_k) * relx * (1 - relx))))
    y_val = tf.where(d.out_of_bounds, x, spline_val)
    return y_val

  def _inverse(self, y):
    """Compute the inverse transformation (Appendix A.3)."""
    d = self._compute_shared(y=y)
    rely = tf.where(d.out_of_bounds, tf.zeros([], dtype=y.dtype), y - d.y_k)
    term2 = rely * (d.d_kp1 + d.d_k - 2 * d.s_k)
    # These terms are the a, b, c terms of the quadratic formula.
    a = d.h_k * (d.s_k - d.d_k) + term2
    b = d.h_k * d.d_k - term2
    c = -d.s_k * rely
    # The expression used here has better numerical behavior for small 4*a*c.
    relx = tf.where(
        tf.equal(rely, 0), tf.zeros([], dtype=a.dtype),
        (2 * c) / (-b - tf.sqrt(b**2 - 4 * a * c)))
    return tf.where(d.out_of_bounds, y, relx * d.w_k + d.x_k)

  def _forward_log_det_jacobian(self, x):
    """Compute the forward derivative (Appendix A.2)."""
    d = self._compute_shared(x=x)
    relx = (x - d.x_k) / d.w_k
    relx = tf.where(d.out_of_bounds, tf.constant(.5, x.dtype), relx)
    grad = (
        2 * tf.math.log(d.s_k) +
        tf.math.log(d.d_kp1 * relx**2 + 2 * d.s_k * relx * (1 - relx) +  # newln
                    d.d_k * (1 - relx)**2) -
        2 * tf.math.log((d.d_kp1 + d.d_k - 2 * d.s_k) * relx *
                        (1 - relx) + d.s_k))
    return tf.where(d.out_of_bounds, tf.zeros([], dtype=x.dtype), grad)

  def _parameter_control_dependencies(self, is_init):
    """Validate parameters."""
    bw, bh, kd = None, None, None
    try:
      shape = tf.broadcast_static_shape(self.bin_widths.shape,
                                        self.bin_heights.shape)
    except ValueError as e:
      raise ValueError('`bin_widths`, `bin_heights` must broadcast: {}'.format(
          str(e)))
    bin_sizes_shape = shape
    try:
      shape = tf.broadcast_static_shape(shape[:-1], self.knot_slopes.shape[:-1])
    except ValueError as e:
      raise ValueError(
          '`bin_widths`, `bin_heights`, and `knot_slopes` must broadcast on '
          'batch axes: {}'.format(str(e)))

    assertions = []
    if (tensorshape_util.is_fully_defined(bin_sizes_shape[-1:]) and
        tensorshape_util.is_fully_defined(self.knot_slopes.shape[-1:])):
      if tensorshape_util.rank(self.knot_slopes.shape) > 0:
        num_interior_knots = tensorshape_util.dims(bin_sizes_shape)[-1] - 1
        if tensorshape_util.dims(
            self.knot_slopes.shape)[-1] not in (1, num_interior_knots):
          raise ValueError(
              'Innermost axis of non-scalar `knot_slopes` must broadcast with '
              '{}; got {}.'.format(num_interior_knots, self.knot_slopes.shape))
    elif self.validate_args:
      if is_init != any(
          tensor_util.is_ref(t)
          for t in (self.bin_widths, self.bin_heights, self.knot_slopes)):
        bw = tf.convert_to_tensor(self.bin_widths) if bw is None else bw
        bh = tf.convert_to_tensor(self.bin_heights) if bh is None else bh
        kd = _ensure_at_least_1d(self.knot_slopes) if kd is None else kd
        shape = tf.broadcast_dynamic_shape(
            tf.shape((bw + bh)[..., :-1]), tf.shape(kd))
        assertions.append(
            assert_util.assert_greater(
                tf.shape(shape)[0],
                tf.zeros([], dtype=shape.dtype),
                message='`(bin_widths + bin_heights)[..., :-1]` must broadcast '
                'with `knot_slopes` to at least 1-D.'))

    if not self.validate_args:
      assert not assertions
      return assertions

    if (is_init != tensor_util.is_ref(self.bin_widths) or
        is_init != tensor_util.is_ref(self.bin_heights)):
      bw = tf.convert_to_tensor(self.bin_widths) if bw is None else bw
      bh = tf.convert_to_tensor(self.bin_heights) if bh is None else bh
      assertions += [
          assert_util.assert_near(
              tf.reduce_sum(bw, axis=-1),
              tf.reduce_sum(bh, axis=-1),
              message='`sum(bin_widths, axis=-1)` must equal '
              '`sum(bin_heights, axis=-1)`.'),
      ]
    if is_init != tensor_util.is_ref(self.bin_widths):
      bw = tf.convert_to_tensor(self.bin_widths) if bw is None else bw
      assertions += [
          assert_util.assert_positive(
              bw, message='`bin_widths` must be positive.'),
      ]
    if is_init != tensor_util.is_ref(self.bin_heights):
      bh = tf.convert_to_tensor(self.bin_heights) if bh is None else bh
      assertions += [
          assert_util.assert_positive(
              bh, message='`bin_heights` must be positive.'),
      ]
    if is_init != tensor_util.is_ref(self.knot_slopes):
      kd = _ensure_at_least_1d(self.knot_slopes) if kd is None else kd
      assertions += [
          assert_util.assert_positive(
              kd, message='`knot_slopes` must be positive.'),
      ]
    return assertions
