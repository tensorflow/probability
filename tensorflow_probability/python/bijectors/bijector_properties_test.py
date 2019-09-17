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
"""Property-based tests for TFP bijectors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import parameterized
import hypothesis as hp
from hypothesis import strategies as hps
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import hypothesis_testlib as bijector_hps
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_case

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

flags.DEFINE_enum('tf_mode', 'graph', ['eager', 'graph'],
                  'TF execution mode to use')

FLAGS = flags.FLAGS

TF2_FRIENDLY_BIJECTORS = (
    'AffineScalar',
    'BatchNormalization',
    # 'CategoricalToDiscrete', TODO(b/137956955): Add support
    # for hypothesis testing
    'CholeskyOuterProduct',
    'Cumsum',
    'DiscreteCosineTransform',
    'Exp',
    'Expm1',
    'FillTriangular',
    'Gumbel',
    'Identity',
    'Inline',
    'Invert',
    'IteratedSigmoidCentered',
    'Kumaraswamy',
    'MatvecLU',
    'NormalCDF',
    'Ordered',
    'PowerTransform',
    'RationalQuadraticSpline',
    'Reciprocal',
    'Sigmoid',
    'SinhArcsinh',
    # 'Softfloor',  # TODO(b/141051113): Enable testing Softfloor.
    'Softplus',
    'Softsign',
    'Square',
    'Tanh',
    'Weibull',
)

BIJECTOR_PARAMS_NDIMS = {
    'AffineScalar': dict(shift=0, scale=0, log_scale=0),
    'Gumbel': dict(loc=0, scale=0),
    'Kumaraswamy': dict(concentration1=0, concentration0=0),
    'MatvecLU': dict(lower_upper=2, permutation=1),
    'SinhArcsinh': dict(skewness=0, tailweight=0),
    'Softfloor': dict(temperature=0),
    'Softplus': dict(hinge_softness=0),
    'RationalQuadraticSpline': dict(bin_widths=1, bin_heights=1, knot_slopes=1),
    'Weibull': dict(concentration=0, scale=0),
}

MUTEX_PARAMS = (
    set(['scale', 'log_scale']),
)

FLDJ = 'forward_log_det_jacobian'
ILDJ = 'inverse_log_det_jacobian'

INVERT_LDJ = {FLDJ: ILDJ, ILDJ: FLDJ}

NO_LDJ_GRADS_EXPECTED = {
    'AffineScalar': dict(shift={FLDJ, ILDJ}),
    'BatchNormalization': dict(beta={FLDJ, ILDJ}),
    'Gumbel': dict(loc={ILDJ}),
}


def is_invert(bijector):
  return isinstance(bijector, tfb.Invert)


# pylint is unable to handle @hps.composite (e.g. complains "No value for
# argument '...' in function call"), so disable this lint for the file.

# pylint: disable=no-value-for-parameter


@hps.composite
def broadcasting_params(draw,
                        bijector_name,
                        batch_shape,
                        event_dim=None,
                        enable_vars=False):
  """Draws a dict of parameters which should yield the given batch shape."""
  params_event_ndims = BIJECTOR_PARAMS_NDIMS.get(bijector_name, {})

  def _constraint(param):
    return constraint_for(bijector_name, param)

  return draw(
      tfp_hps.broadcasting_params(
          batch_shape,
          params_event_ndims,
          event_dim=event_dim,
          enable_vars=enable_vars,
          constraint_fn_for=_constraint,
          mutex_params=MUTEX_PARAMS))


class CallableModule(tf.Module):  # TODO(b/141098791): Eliminate this.
  """Convenience object for capturing variables closed over by Inline."""

  def __init__(self, fn, varobj):
    self._fn = fn
    self._vars = varobj

  def __call__(self, *args, **kwargs):
    return self._fn(*args, **kwargs)


@hps.composite
def bijectors(draw, bijector_name=None, batch_shape=None, event_dim=None,
              enable_vars=False):
  """Strategy for drawing Bijectors.

  The emitted bijector may be a basic bijector or an `Invert` of a basic
  bijector, but not a compound like `Chain`.

  Args:
    draw: Hypothesis strategy sampler supplied by `@hps.composite`.
    bijector_name: Optional Python `str`.  If given, the produced bijectors
      will all have this type.  If omitted, Hypothesis chooses one from
      the whitelist `TF2_FRIENDLY_BIJECTORS`.
    batch_shape: An optional `TensorShape`.  The batch shape of the resulting
      bijector.  Hypothesis will pick one if omitted.
    event_dim: Optional Python int giving the size of each of the underlying
      distribution's parameters' event dimensions.  This is shared across all
      parameters, permitting square event matrices, compatible location and
      scale Tensors, etc. If omitted, Hypothesis will choose one.
    enable_vars: TODO(bjp): Make this `True` all the time and put variable
      initialization in slicing_test.  If `False`, the returned parameters are
      all Tensors, never Variables or DeferredTensor.

  Returns:
    bijectors: A strategy for drawing bijectors with the specified `batch_shape`
      (or an arbitrary one if omitted).
  """
  if bijector_name is None:
    bijector_name = draw(hps.sampled_from(TF2_FRIENDLY_BIJECTORS))
  if batch_shape is None:
    batch_shape = draw(tfp_hps.shapes())
  if event_dim is None:
    event_dim = draw(hps.integers(min_value=2, max_value=6))
  if bijector_name == 'Invert':
    underlying_name = draw(
        hps.sampled_from(sorted(set(TF2_FRIENDLY_BIJECTORS) - {'Invert'})))
    underlying = draw(
        bijectors(
            bijector_name=underlying_name,
            batch_shape=batch_shape,
            event_dim=event_dim,
            enable_vars=enable_vars))
    bijector_params = {'bijector': underlying}
  elif bijector_name == 'Inline':
    scale = draw(hps.sampled_from([-2., 1., 2.]))
    scale = tf.Variable(scale, name='scale') if enable_vars else scale
    b = tfb.AffineScalar(scale=scale)

    bijector_params = dict(
        forward_fn=CallableModule(b.forward, b),
        inverse_fn=b.inverse,
        forward_log_det_jacobian_fn=lambda x: b.forward_log_det_jacobian(  # pylint: disable=g-long-lambda
            x, event_ndims=b.forward_min_event_ndims),
        forward_min_event_ndims=b.forward_min_event_ndims,
        is_constant_jacobian=b.is_constant_jacobian,
        is_increasing=b.is_increasing,
    )
  elif bijector_name == 'DiscreteCosineTransform':
    bijector_params = {'dct_type': draw(hps.integers(min_value=2, max_value=3))}
  elif bijector_name == 'PowerTransform':
    bijector_params = {'power': draw(hps.floats(min_value=1e-6, max_value=10.))}
  else:
    bijector_params = draw(
        broadcasting_params(bijector_name, batch_shape, event_dim=event_dim,
                            enable_vars=enable_vars))
  hp.note('bijector params: {}'.format(bijector_params))
  ctor = getattr(tfb, bijector_name)
  return ctor(validate_args=True, **bijector_params)


Support = tfp_hps.Support


def constrain_forward_shape(bijector, shape):
  """Constrain the shape so it is compatible with bijector.forward."""
  if is_invert(bijector):
    return constrain_inverse_shape(bijector.bijector, shape=shape)

  support = bijector_hps.bijector_supports()[
      type(bijector).__name__].forward
  if support == tfp_hps.Support.VECTOR_SIZE_TRIANGULAR:
    # Need to constrain the shape.
    shape[-1] = int(shape[-1] * (shape[-1] + 1) / 2)
  return shape


def constrain_inverse_shape(bijector, shape):
  """Constrain the shape so it is compatible with bijector.inverse."""
  if is_invert(bijector):
    return constrain_forward_shape(bijector.bijector, shape=shape)
  return shape


@hps.composite
def domain_tensors(draw, bijector, shape=None):
  """Strategy for drawing Tensors in the domain of a bijector.

  If the bijector's domain is constrained, this proceeds by drawing an
  unconstrained Tensor and then transforming it to fit.  The constraints are
  declared in `bijectors.hypothesis_testlib.bijector_supports`.  The
  transformations are defined by `tfp_hps.constrainer`.

  Args:
    draw: Hypothesis strategy sampler supplied by `@hps.composite`.
    bijector: A `Bijector` in whose domain the Tensors will be.
    shape: An optional `TensorShape`.  The shape of the resulting
      Tensors.  Hypothesis will pick one if omitted.

  Returns:
    tensors: A strategy for drawing domain Tensors for the desired bijector.
  """
  if is_invert(bijector):
    return draw(codomain_tensors(bijector.bijector, shape))
  if shape is None:
    shape = draw(tfp_hps.shapes())
  bijector_name = type(bijector).__name__
  support = bijector_hps.bijector_supports()[bijector_name].forward
  if isinstance(bijector, tfb.PowerTransform):
    constraint_fn = bijector_hps.power_transform_constraint(bijector.power)
  else:
    constraint_fn = tfp_hps.constrainer(support)
  return draw(tfp_hps.constrained_tensors(constraint_fn, shape))


@hps.composite
def codomain_tensors(draw, bijector, shape=None):
  """Strategy for drawing Tensors in the codomain of a bijector.

  If the bijector's codomain is constrained, this proceeds by drawing an
  unconstrained Tensor and then transforming it to fit.  The constraints are
  declared in `bijectors.hypothesis_testlib.bijector_supports`.  The
  transformations are defined by `tfp_hps.constrainer`.

  Args:
    draw: Hypothesis strategy sampler supplied by `@hps.composite`.
    bijector: A `Bijector` in whose codomain the Tensors will be.
    shape: An optional `TensorShape`.  The shape of the resulting
      Tensors.  Hypothesis will pick one if omitted.

  Returns:
    tensors: A strategy for drawing codomain Tensors for the desired bijector.
  """
  if is_invert(bijector):
    return draw(domain_tensors(bijector.bijector, shape))
  if shape is None:
    shape = draw(tfp_hps.shapes())
  bijector_name = type(bijector).__name__
  support = bijector_hps.bijector_supports()[bijector_name].inverse
  constraint_fn = tfp_hps.constrainer(support)
  return draw(tfp_hps.constrained_tensors(constraint_fn, shape))


def assert_no_none_grad(bijector, method, wrt_vars, grads):
  for var, grad in zip(wrt_vars, grads):
    expect_grad = var.dtype not in (tf.int32, tf.int64)
    if 'log_det_jacobian' in method:
      if tensor_util.is_ref(var):
        # We check tensor_util.is_ref to account for xs/ys being in vars.
        var_name = var.name.rstrip('_0123456789:').split('/')[-1]
      else:
        var_name = '[arg]'
      to_check = bijector.bijector if is_invert(bijector) else bijector
      to_check_method = INVERT_LDJ[method] if is_invert(bijector) else method
      if var_name == '[arg]' and bijector.is_constant_jacobian:
        expect_grad = False
      exempt_var_method = NO_LDJ_GRADS_EXPECTED.get(type(to_check).__name__, {})
      if to_check_method in exempt_var_method.get(var_name, ()):
        expect_grad = False

    if expect_grad != (grad is not None):
      raise AssertionError('{} `{}` -> {} grad for bijector {}'.format(
          'Missing' if expect_grad else 'Unexpected', method, var, bijector))


@test_util.run_all_in_graph_and_eager_modes
class BijectorPropertiesTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': bname, 'bijector_name': bname}
      for bname in TF2_FRIENDLY_BIJECTORS)
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testBijector(self, bijector_name, data):
    if tf.executing_eagerly() != (FLAGS.tf_mode == 'eager'):
      return
    event_dim = data.draw(hps.integers(min_value=2, max_value=6))
    bijector = data.draw(
        bijectors(bijector_name=bijector_name, event_dim=event_dim,
                  enable_vars=True))
    self.evaluate(tf.group(*[v.initializer for v in bijector.variables]))

    # Forward mapping: Check differentiation through forward mapping with
    # respect to the input and parameter variables.  Also check that any
    # variables are not referenced overmuch.
    # TODO(axch): Would be nice to get rid of all this shape inference logic and
    # just rely on a notion of batch and event shape for bijectors, so we can
    # pass those through `domain_tensors` and `codomain_tensors` and use
    # `tensors_in_support`.  However, `RationalQuadraticSpline` behaves weirdly
    # somehow and I got confused.
    codomain_event_shape = [event_dim] * bijector.inverse_min_event_ndims
    codomain_event_shape = constrain_inverse_shape(
        bijector, codomain_event_shape)
    shp = bijector.inverse_event_shape(codomain_event_shape)
    shp = tensorshape_util.concatenate(
        data.draw(
            tfp_hps.broadcast_compatible_shape(
                shp[:shp.ndims - bijector.forward_min_event_ndims])),
        shp[shp.ndims - bijector.forward_min_event_ndims:])
    xs = tf.identity(data.draw(domain_tensors(bijector, shape=shp)), name='xs')
    wrt_vars = [xs] + [v for v in bijector.trainable_variables
                       if v.dtype.is_floating]
    with tf.GradientTape() as tape:
      with tfp_hps.assert_no_excessive_var_usage(
          'method `forward` of {}'.format(bijector)):
        tape.watch(wrt_vars)
        # TODO(b/73073515): Fix graph mode gradients with bijector caching.
        ys = bijector.forward(xs + 0)
    grads = tape.gradient(ys, wrt_vars)
    assert_no_none_grad(bijector, 'forward', wrt_vars, grads)

    # For scalar bijectors, verify correctness of the is_increasing method.
    if (bijector.forward_min_event_ndims == 0 and
        bijector.inverse_min_event_ndims == 0):
      dydx = grads[0]
      hp.note('dydx: {}'.format(dydx))
      isfinite = tf.math.is_finite(dydx)
      incr_or_slope_eq0 = bijector.is_increasing() | tf.equal(dydx, 0)
      self.assertAllEqual(
          isfinite & incr_or_slope_eq0,
          isfinite & (dydx >= 0) | tf.zeros_like(incr_or_slope_eq0))

    # FLDJ: Check differentiation through forward log det jacobian with
    # respect to the input and parameter variables.  Also check that any
    # variables are not referenced overmuch.
    event_ndims = data.draw(
        hps.integers(
            min_value=bijector.forward_min_event_ndims,
            max_value=xs.shape.ndims))
    with tf.GradientTape() as tape:
      max_permitted = 2 if hasattr(bijector, '_forward_log_det_jacobian') else 4
      if is_invert(bijector):
        max_permitted = (2 if hasattr(bijector.bijector,
                                      '_inverse_log_det_jacobian') else 4)
      with tfp_hps.assert_no_excessive_var_usage(
          'method `forward_log_det_jacobian` of {}'.format(bijector),
          max_permissible=max_permitted):
        tape.watch(wrt_vars)
        # TODO(b/73073515): Fix graph mode gradients with bijector caching.
        ldj = bijector.forward_log_det_jacobian(xs + 0, event_ndims=event_ndims)
    grads = tape.gradient(ldj, wrt_vars)
    assert_no_none_grad(bijector, 'forward_log_det_jacobian', wrt_vars, grads)

    # Inverse mapping: Check differentiation through inverse mapping with
    # respect to the codomain "input" and parameter variables.  Also check that
    # any variables are not referenced overmuch.
    domain_event_shape = [event_dim] * bijector.forward_min_event_ndims
    domain_event_shape = constrain_forward_shape(bijector, domain_event_shape)
    shp = bijector.forward_event_shape(domain_event_shape)
    shp = tensorshape_util.concatenate(
        data.draw(
            tfp_hps.broadcast_compatible_shape(
                shp[:shp.ndims - bijector.inverse_min_event_ndims])),
        shp[shp.ndims - bijector.inverse_min_event_ndims:])
    ys = tf.identity(
        data.draw(codomain_tensors(bijector, shape=shp)), name='ys')
    wrt_vars = [ys] + [v for v in bijector.trainable_variables
                       if v.dtype.is_floating]
    with tf.GradientTape() as tape:
      with tfp_hps.assert_no_excessive_var_usage(
          'method `inverse` of {}'.format(bijector)):
        tape.watch(wrt_vars)
        # TODO(b/73073515): Fix graph mode gradients with bijector caching.
        xs = bijector.inverse(ys + 0)
    grads = tape.gradient(xs, wrt_vars)
    assert_no_none_grad(bijector, 'inverse', wrt_vars, grads)

    # ILDJ: Check differentiation through inverse log det jacobian with respect
    # to the codomain "input" and parameter variables.  Also check that any
    # variables are not referenced overmuch.
    event_ndims = data.draw(
        hps.integers(
            min_value=bijector.inverse_min_event_ndims,
            max_value=ys.shape.ndims))
    with tf.GradientTape() as tape:
      max_permitted = 2 if hasattr(bijector, '_inverse_log_det_jacobian') else 4
      if is_invert(bijector):
        max_permitted = (2 if hasattr(bijector.bijector,
                                      '_forward_log_det_jacobian') else 4)
      with tfp_hps.assert_no_excessive_var_usage(
          'method `inverse_log_det_jacobian` of {}'.format(bijector),
          max_permissible=max_permitted):
        tape.watch(wrt_vars)
        # TODO(b/73073515): Fix graph mode gradients with bijector caching.
        ldj = bijector.inverse_log_det_jacobian(ys + 0, event_ndims=event_ndims)
    grads = tape.gradient(ldj, wrt_vars)
    assert_no_none_grad(bijector, 'inverse_log_det_jacobian', wrt_vars, grads)


def ensure_nonzero(x):
  return tf.where(x < 1e-6, tf.constant(1e-3, x.dtype), x)


CONSTRAINTS = {
    'concentration':
        tfp_hps.softplus_plus_eps(),
    'concentration0':
        tfp_hps.softplus_plus_eps(),
    'concentration1':
        tfp_hps.softplus_plus_eps(),
    'hinge_softness':
        tfp_hps.softplus_plus_eps(),
    'scale':
        tfp_hps.softplus_plus_eps(),
    'tailweight':
        tfp_hps.softplus_plus_eps(),
    'AffineScalar.scale':
        tfp_hps.softplus_plus_eps(),
    'Softfloor.temperature':
        tfp_hps.softplus_plus_eps(.05),
    'bin_widths':
        bijector_hps.spline_bin_size_constraint,
    'bin_heights':
        bijector_hps.spline_bin_size_constraint,
    'knot_slopes':
        bijector_hps.spline_slope_constraint,
    'lower_upper':
        lambda x: tf.linalg.set_diag(x, ensure_nonzero(tf.linalg.diag_part(x))),
    'permutation':
        lambda x: tf.math.top_k(x, k=x.shape[-1]).indices,
}


def constraint_for(bijector_name=None, param=None):
  if param is not None:
    return CONSTRAINTS.get('{}.{}'.format(bijector_name, param),
                           CONSTRAINTS.get(param, tfp_hps.identity_fn))
  return CONSTRAINTS.get(bijector_name, tfp_hps.identity_fn)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
