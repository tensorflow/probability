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

import collections
import functools
from absl.testing import parameterized
import hypothesis as hp
from hypothesis import strategies as hps
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.bijectors import hypothesis_testlib as bhps
from tensorflow_probability.python.bijectors import invert as invert_lib
from tensorflow_probability.python.internal import auto_composite_tensor
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


MUTEX_PARAMS = (
    set(['scale', 'log_scale']),
)

FLDJ = 'forward_log_det_jacobian'
ILDJ = 'inverse_log_det_jacobian'

INVERT_LDJ = {FLDJ: ILDJ, ILDJ: FLDJ}

NO_LDJ_GRADS_EXPECTED = {
    'BatchNormalization': dict(beta={FLDJ, ILDJ}),
    'FrechetCDF': dict(loc={ILDJ}),
    'GeneralizedExtremeValueCDF': dict(loc={ILDJ}),
    'GumbelCDF': dict(loc={ILDJ}),
    'Householder': dict(reflection_axis={FLDJ, ILDJ}),
    'MoyalCDF': dict(loc={ILDJ}),
    'Shift': dict(shift={FLDJ, ILDJ}),
    'Softplus': dict(low={FLDJ}),
}

AUTOVECTORIZATION_IS_BROKEN = [
    'BatchNormalization',  # Might (necessarily) violate shape semantics?
]

AUTOVECTORIZATION_RTOL = collections.defaultdict(lambda: 1e-5)
AUTOVECTORIZATION_RTOL.update({
    'Invert': 1e-2,  # Can contain poorly-conditioned bijectors.
    'MatvecLU': 1e-4,  # TODO(b/156638569) tighten this.
    'ScaleMatvecLU': 1e-2,  # TODO(b/151041130) tighten this.
    'ScaleMatvecTriL': 1e-3})  # TODO(b/150250388) tighten this.
AUTOVECTORIZATION_ATOL = collections.defaultdict(lambda: 1e-5)
AUTOVECTORIZATION_ATOL.update({
    'MatvecLU': 1e-3,  # TODO(b/156638569); triggered by Hypothesis 4+ upgrade
    'ScaleMatvecLU': 1e-2,  # TODO(b/151041130) tighten this.
    'ScaleMatvecTriL': 1e-1})  # TODO(b/150250388) tighten this.


COMPOSITE_TENSOR_IS_BROKEN = [
    'BatchNormalization',  # tf.layers arg
    'RationalQuadraticSpline',  # TODO(b/185628453): Debug loss of static info.
]

COMPOSITE_TENSOR_RTOL = collections.defaultdict(lambda: 2e-6)
COMPOSITE_TENSOR_RTOL.update({
    'PowerTransform': 1e-5,
})
COMPOSITE_TENSOR_ATOL = collections.defaultdict(lambda: 1e-6)

SLICING_RTOL = collections.defaultdict(lambda: 1e-5)
SLICING_ATOL = collections.defaultdict(lambda: 1e-5)
SLICING_ATOL.update({
    'Householder': 1e-4,
})

INSTANTIABLE_BUT_NOT_SLICEABLE = [
    # TODO(b/146897388): These are sliceable but have parameter dependent
    # support. Perhaps re-write the slicing test to enable these bijectors.
    'FrechetCDF',
    'GeneralizedExtremeValueCDF'
]


def is_invert(bijector):
  return isinstance(bijector, (tfb.Invert, invert_lib._Invert))


def is_transform_diagonal(bijector):
  return isinstance(bijector, tfb.TransformDiagonal)


def is_generalized_pareto(bijector):
  return isinstance(bijector, tfb.GeneralizedPareto)


# pylint is unable to handle @hps.composite (e.g. complains "No value for
# argument '...' in function call"), so disable this lint for the file.

# pylint: disable=no-value-for-parameter


# TODO(b/141098791): Eliminate this.
@auto_composite_tensor.auto_composite_tensor
class CallableModule(tf.Module, auto_composite_tensor.AutoCompositeTensor):
  """Convenience object for capturing variables closed over by Inline."""

  def __init__(self, fn, varobj):
    self._fn = fn
    self._varobj = varobj

  def __call__(self, *args, **kwargs):
    return self._fn(*args, **kwargs)


@hps.composite
def bijectors(draw, bijector_name=None, batch_shape=None, event_dim=None,
              enable_vars=False, allowed_bijectors=None, validate_args=True,
              return_duplicate=False):
  """Strategy for drawing Bijectors.

  The emitted bijector may be a basic bijector or an `Invert` of a basic
  bijector, but not a compound like `Chain`.

  Args:
    draw: Hypothesis strategy sampler supplied by `@hps.composite`.
    bijector_name: Optional Python `str`.  If given, the produced bijectors
      will all have this type.  If omitted, Hypothesis chooses one from
      the allowlist `INSTANTIABLE_BIJECTORS`.
    batch_shape: An optional `TensorShape`.  The batch shape of the resulting
      bijector.  Hypothesis will pick one if omitted.
    event_dim: Optional Python int giving the size of each of the underlying
      bijector's parameters' event dimensions.  This is shared across all
      parameters, permitting square event matrices, compatible location and
      scale Tensors, etc. If omitted, Hypothesis will choose one.
    enable_vars: TODO(bjp): Make this `True` all the time and put variable
      initialization in slicing_test.  If `False`, the returned parameters are
      all `tf.Tensor`s and not {`tf.Variable`, `tfp.util.DeferredTensor`
      `tfp.util.TransformedVariable`}
    allowed_bijectors: Optional list of `str` Bijector names to sample from.
      Bijectors not in this list will not be returned or instantiated as
      part of a meta-bijector (Chain, Invert, etc.). Defaults to
      `INSTANTIABLE_BIJECTORS`.
    validate_args: Python `bool`; whether to enable runtime checks.
    return_duplicate: Python `bool`: If `False` return a single bijector. If
      `True` return a tuple of two bijectors of the same type, instantiated with
      the same parameters.

  Returns:
    bijectors: A strategy for drawing bijectors with the specified `batch_shape`
      (or an arbitrary one if omitted).
  """
  if allowed_bijectors is None:
    allowed_bijectors = bhps.INSTANTIABLE_BIJECTORS
  if bijector_name is None:
    bijector_name = draw(hps.sampled_from(allowed_bijectors))
  if batch_shape is None:
    batch_shape = draw(tfp_hps.shapes())
  if event_dim is None:
    event_dim = draw(hps.integers(min_value=2, max_value=6))
  if bijector_name == 'Invert':
    underlying_name = draw(
        hps.sampled_from(sorted(set(allowed_bijectors) - {'Invert'})))
    underlying = draw(
        bijectors(
            bijector_name=underlying_name,
            batch_shape=batch_shape,
            event_dim=event_dim,
            enable_vars=enable_vars,
            allowed_bijectors=allowed_bijectors,
            validate_args=validate_args))
    bijector_params = {'bijector': underlying}
    msg = 'Forming Invert bijector with underlying bijector {}.'
    hp.note(msg.format(underlying))
  elif bijector_name == 'TransformDiagonal':
    underlying_name = draw(
        hps.sampled_from(sorted(
            set(allowed_bijectors) & set(bhps.TRANSFORM_DIAGONAL_ALLOWLIST))))
    underlying = draw(
        bijectors(
            bijector_name=underlying_name,
            batch_shape=(),
            event_dim=event_dim,
            enable_vars=enable_vars,
            allowed_bijectors=allowed_bijectors,
            validate_args=validate_args))
    bijector_params = {'diag_bijector': underlying}
    msg = 'Forming TransformDiagonal bijector with underlying bijector {}.'
    hp.note(msg.format(underlying))
  elif bijector_name == 'Inline':
    scale = draw(tfp_hps.maybe_variable(
        hps.sampled_from(np.float32([1., -1., 2, -2.])), enable_vars))
    b = tfb.Scale(scale=scale)

    bijector_params = dict(
        forward_fn=CallableModule(b.forward, b),
        inverse_fn=b.inverse,
        forward_log_det_jacobian_fn=lambda x: b.forward_log_det_jacobian(  # pylint: disable=g-long-lambda
            x, event_ndims=b.forward_min_event_ndims),
        forward_min_event_ndims=b.forward_min_event_ndims,
        is_constant_jacobian=b.is_constant_jacobian,
        is_increasing=b._internal_is_increasing,  # pylint: disable=protected-access
    )
  elif bijector_name == 'DiscreteCosineTransform':
    dct_type = hps.integers(min_value=2, max_value=3)
    bijector_params = {'dct_type': draw(dct_type)}
  elif bijector_name == 'GeneralizedPareto':
    concentration = hps.floats(min_value=-200., max_value=200)
    scale = hps.floats(min_value=1e-2, max_value=200)
    loc = hps.floats(min_value=-200, max_value=200)
    bijector_params = {'concentration': draw(concentration),
                       'scale': draw(scale),
                       'loc': draw(loc)}
  elif bijector_name == 'PowerTransform':
    power = hps.floats(min_value=1e-6, max_value=10.)
    bijector_params = {'power': draw(power)}
  elif bijector_name == 'Permute':
    event_ndims = draw(hps.integers(min_value=1, max_value=2))
    axis = hps.integers(min_value=-event_ndims, max_value=-1)
    # This is a permutation of dimensions within an axis.
    # (Contrast with `Transpose` below.)
    bijector_params = {
        'axis': draw(axis),
        'permutation': draw(tfp_hps.maybe_variable(
            hps.permutations(np.arange(event_dim)), enable_vars,
            dtype=tf.int32))
    }
  elif bijector_name == 'Reshape':
    event_shape_out = draw(tfp_hps.shapes(min_ndims=1))
    # TODO(b/142135119): Wanted to draw general input and output shapes like the
    # following, but Hypothesis complained about filtering out too many things.
    # event_shape_in = draw(tfp_hps.shapes(min_ndims=1))
    # hp.assume(event_shape_out.num_elements() == event_shape_in.num_elements())
    event_shape_in = [event_shape_out.num_elements()]
    bijector_params = {'event_shape_out': event_shape_out,
                       'event_shape_in': event_shape_in}
  elif bijector_name == 'Transpose':
    event_ndims = draw(hps.integers(min_value=0, max_value=2))
    # This is a permutation of axes.
    # (Contrast with `Permute` above.)
    bijector_params = {'perm': draw(hps.permutations(np.arange(event_ndims)))}
  else:
    params_event_ndims = bhps.INSTANTIABLE_BIJECTORS[
        bijector_name].params_event_ndims
    bijector_params = draw(
        tfp_hps.broadcasting_params(
            batch_shape,
            params_event_ndims,
            event_dim=event_dim,
            enable_vars=enable_vars,
            constraint_fn_for=lambda param: constraint_for(bijector_name, param),  # pylint:disable=line-too-long
            mutex_params=MUTEX_PARAMS))
    bijector_params = constrain_params(bijector_params, bijector_name)

  ctor = getattr(tfb, bijector_name)
  hp.note('Forming {} bijector with params {}.'.format(
      bijector_name, bijector_params))
  bijector = ctor(validate_args=validate_args, **bijector_params)
  if not return_duplicate:
    return bijector
  return (bijector, ctor(validate_args=validate_args, **bijector_params))


def constrain_forward_shape(bijector, shape):
  """Constrain the shape so it is compatible with bijector.forward.

  Args:
    bijector: A `Bijector`.
    shape: A TensorShape or compatible, giving the desired event shape.

  Returns:
    shape: A TensorShape, giving an event shape compatible with
      `bijector.forward`, loosely inspired by the input `shape`.
  """
  if is_invert(bijector):
    return constrain_inverse_shape(bijector.bijector, shape=shape)

  # TODO(b/146897388): Enable bijectors with parameter-dependent support.
  support = bhps.bijector_supports()[
      type(bijector).__name__].forward
  if support == tfp_hps.Support.VECTOR_SIZE_TRIANGULAR:
    # Need to constrain the shape.
    shape[-1] = int(shape[-1] * (shape[-1] + 1) / 2)
  if isinstance(bijector, tfb.Reshape):
    # Note: This relies on the out event shape being fully determined
    shape = tf.get_static_value(bijector._event_shape_in)
  return tf.TensorShape(shape)


def constrain_inverse_shape(bijector, shape):
  """Constrain the shape so it is compatible with bijector.inverse.

  Args:
    bijector: A `Bijector`.
    shape: A TensorShape or compatible, giving the desired event shape.

  Returns:
    shape: A TensorShape, giving an event shape compatible with
      `bijector.inverse`, loosely inspired by the input `shape`.
  """
  if is_invert(bijector):
    return constrain_forward_shape(bijector.bijector, shape=shape)
  if isinstance(bijector, tfb.Reshape):
    # Note: This relies on the out event shape being fully determined
    shape = tf.get_static_value(bijector._event_shape_out)
  return tf.TensorShape(shape)


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
  elif is_transform_diagonal(bijector):
    return draw(domain_tensors(bijector.diag_bijector, shape))
  if shape is None:
    shape = draw(tfp_hps.shapes())
  bijector_name = type(bijector).__name__
  support = bhps.bijector_supports()[bijector_name].forward
  if isinstance(bijector, tfb.PowerTransform):
    constraint_fn = bhps.power_transform_constraint(bijector.power)
  elif isinstance(bijector, tfb.FrechetCDF):
    constraint_fn = bhps.frechet_constraint(bijector.loc)
  elif isinstance(bijector, tfb.GeneralizedExtremeValueCDF):
    constraint_fn = bhps.gev_constraint(bijector.loc,
                                        bijector.scale,
                                        bijector.concentration)
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
  elif is_transform_diagonal(bijector):
    return draw(codomain_tensors(bijector.diag_bijector, shape))
  if shape is None:
    shape = draw(tfp_hps.shapes())
  bijector_name = type(bijector).__name__
  support = bhps.bijector_supports()[bijector_name].inverse
  if is_generalized_pareto(bijector):
    constraint_fn = bhps.generalized_pareto_constraint(
        bijector.loc, bijector.scale, bijector.concentration)
  elif isinstance(bijector, tfb.SoftClip):
    constraint_fn = bhps.softclip_constraint(
        bijector.low, bijector.high)
  else:
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
      to_check = bijector
      while is_invert(to_check) or is_transform_diagonal(to_check):
        to_check = to_check.bijector if is_invert(to_check) else to_check
        to_check = (to_check.diag_bijector
                    if is_transform_diagonal(to_check) else to_check)
      to_check_method = INVERT_LDJ[method] if is_invert(bijector) else method
      if var_name == '[arg]' and bijector.is_constant_jacobian:
        expect_grad = False
      exempt_var_method = NO_LDJ_GRADS_EXPECTED.get(type(to_check).__name__, {})
      if to_check_method in exempt_var_method.get(var_name, ()):
        expect_grad = False

    if expect_grad != (grad is not None):
      raise AssertionError('{} `{}` -> {} grad for bijector {}'.format(
          'Missing' if expect_grad else 'Unexpected', method, var, bijector))


def _ldj_tensor_conversions_allowed(bijector, is_forward):
  if is_invert(bijector):
    return _ldj_tensor_conversions_allowed(bijector.bijector, not is_forward)
  elif is_transform_diagonal(bijector):
    return _ldj_tensor_conversions_allowed(bijector.diag_bijector, is_forward)
  elif is_generalized_pareto(bijector):
    return max(
        _ldj_tensor_conversions_allowed(
            bijector._negative_concentration_bijector(), is_forward),
        _ldj_tensor_conversions_allowed(
            bijector._non_negative_concentration_bijector, is_forward))
  elif is_forward:
    return 2 if hasattr(bijector, '_forward_log_det_jacobian') else 4
  else:
    return 2 if hasattr(bijector, '_inverse_log_det_jacobian') else 4


@test_util.test_all_tf_execution_regimes
class BijectorPropertiesTest(test_util.TestCase):

  def _draw_bijector(self, bijector_name, data,
                     batch_shape=None, allowed_bijectors=None,
                     validate_args=True):
    event_dim = data.draw(hps.integers(min_value=2, max_value=6))
    bijector = data.draw(
        bijectors(bijector_name=bijector_name, event_dim=event_dim,
                  enable_vars=True, batch_shape=batch_shape,
                  allowed_bijectors=allowed_bijectors,
                  validate_args=validate_args))
    self.evaluate(tf.group(*[v.initializer for v in bijector.variables]))
    return bijector, event_dim

  def _draw_domain_tensor(self, bijector, data, event_dim, sample_shape=()):
    # TODO(axch): Would be nice to get rid of all this shape inference logic and
    # just rely on a notion of batch and event shape for bijectors, so we can
    # pass those through `domain_tensors` and `codomain_tensors` and use
    # `tensors_in_support`.  However, `RationalQuadraticSpline` behaves weirdly
    # somehow and I got confused.
    codomain_event_shape = [event_dim] * bijector.inverse_min_event_ndims
    codomain_event_shape = constrain_inverse_shape(
        bijector, codomain_event_shape)
    shp = bijector.inverse_event_shape(codomain_event_shape)
    shp = functools.reduce(tensorshape_util.concatenate, [
        sample_shape,
        data.draw(
            tfp_hps.broadcast_compatible_shape(
                shp[:shp.ndims - bijector.forward_min_event_ndims])),
        shp[shp.ndims - bijector.forward_min_event_ndims:]])
    xs = tf.identity(data.draw(domain_tensors(bijector, shape=shp)), name='xs')

    return xs

  def _draw_codomain_tensor(self, bijector, data, event_dim, sample_shape=()):
    return self._draw_domain_tensor(tfb.Invert(bijector),
                                    data=data,
                                    event_dim=event_dim,
                                    sample_shape=sample_shape)

  @parameterized.named_parameters(
      {'testcase_name': bname, 'bijector_name': bname}
      for bname in set(bhps.INSTANTIABLE_BIJECTORS))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testBijector(self, bijector_name, data):
    tfp_hps.guitar_skip_if_matches('Tanh', bijector_name, 'b/144163991')

    bijector, event_dim = self._draw_bijector(bijector_name, data)

    # Forward mapping: Check differentiation through forward mapping with
    # respect to the input and parameter variables.  Also check that any
    # variables are not referenced overmuch.
    xs = self._draw_domain_tensor(bijector, data, event_dim)
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

    # For scalar bijectors, verify correctness of the _is_increasing method.
    # TODO(b/148459057): Except, don't verify Softfloor on Guitar because
    # of numerical problem.
    def exception(bijector):
      if not tfp_hps.running_under_guitar():
        return False
      if isinstance(bijector, tfb.Softfloor):
        return True
      if is_invert(bijector):
        return exception(bijector.bijector)
      return False
    if (bijector.forward_min_event_ndims == 0 and
        bijector.inverse_min_event_ndims == 0 and
        not exception(bijector)):
      dydx = grads[0]
      hp.note('dydx: {}'.format(dydx))
      isfinite = tf.math.is_finite(dydx)
      incr_or_slope_eq0 = bijector._internal_is_increasing() | tf.equal(dydx, 0)  # pylint: disable=protected-access
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
      max_permitted = _ldj_tensor_conversions_allowed(bijector, is_forward=True)
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
    ys = self._draw_codomain_tensor(bijector, data, event_dim)
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
      max_permitted = _ldj_tensor_conversions_allowed(
          bijector, is_forward=False)
      with tfp_hps.assert_no_excessive_var_usage(
          'method `inverse_log_det_jacobian` of {}'.format(bijector),
          max_permissible=max_permitted):
        tape.watch(wrt_vars)
        # TODO(b/73073515): Fix graph mode gradients with bijector caching.
        ldj = bijector.inverse_log_det_jacobian(ys + 0, event_ndims=event_ndims)
    grads = tape.gradient(ldj, wrt_vars)
    assert_no_none_grad(bijector, 'inverse_log_det_jacobian', wrt_vars, grads)

    # Verify that `_is_permutation` implies constant zero Jacobian.
    if bijector._is_permutation:
      self.assertTrue(bijector._is_constant_jacobian)
      self.assertAllEqual(ldj, 0.)

    # Verify correctness of batch shape.
    xs_batch_shapes = tf.nest.map_structure(
        lambda x, nd: ps.shape(x)[:ps.rank(x) - nd],
        xs,
        bijector.inverse_event_ndims(event_ndims))
    empirical_batch_shape = functools.reduce(
        ps.broadcast_shape,
        nest.flatten_up_to(bijector.forward_min_event_ndims, xs_batch_shapes))
    batch_shape = bijector.experimental_batch_shape(y_event_ndims=event_ndims)
    if tensorshape_util.is_fully_defined(batch_shape):
      self.assertAllEqual(empirical_batch_shape, batch_shape)
    self.assertAllEqual(empirical_batch_shape,
                        bijector.experimental_batch_shape_tensor(
                            y_event_ndims=event_ndims))

    # Check that the outputs of forward_dtype and inverse_dtype match the dtypes
    # of the outputs of forward and inverse.
    self.assertAllEqualNested(ys.dtype, bijector.forward_dtype(xs.dtype))
    self.assertAllEqualNested(xs.dtype, bijector.inverse_dtype(ys.dtype))

  @parameterized.named_parameters(
      {'testcase_name': bname, 'bijector_name': bname}
      for bname in set(bhps.INSTANTIABLE_BIJECTORS))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testParameterProperties(self, bijector_name, data):
    if tf.config.functions_run_eagerly() or not tf.executing_eagerly():
      self.skipTest('To reduce test weight, parameter properties tests run in '
                    'eager mode only.')

    non_trainable_params = (
        'bijector',  # Several.
        'forward_fn',  # Inline.
        'inverse_fn',  # Inline.
        'forward_min_event_ndims',  # Inline.
        'inverse_min_event_ndims',  # Inline.
        'event_shape_out',  # Reshape.
        'event_shape_in',  # Reshape.
        'perm',  # Transpose.
        'rightmost_transposed_ndims',  # Transpose.
        'diag_bijector',  # TransformDiagonal.
        'diag_shift'  # FillScaleTriL (doesn't support batch shape).
    )
    bijector, event_dim = self._draw_bijector(
        bijector_name, data,
        validate_args=True,
        allowed_bijectors=bhps.INSTANTIABLE_BIJECTORS)

    # Extract the full shape of an output from this bijector.
    xs = self._draw_domain_tensor(bijector, data, event_dim)
    ys = bijector.forward(xs)
    output_shape = ps.shape(ys)
    sample_and_batch_ndims = (ps.rank_from_shape(output_shape) -
                              bijector.inverse_min_event_ndims)

    try:
      params = type(bijector).parameter_properties()
      params64 = type(bijector).parameter_properties(dtype=tf.float64)
    except NotImplementedError as e:
      self.skipTest(str(e))

    seeds = samplers.split_seed(test_util.test_seed(), n=len(params))
    new_parameters = {}
    for i, (param_name, param) in enumerate(params.items()):
      if param_name in non_trainable_params:
        continue

      # Check that the shape_fn is consistent with event_ndims.
      try:
        param_shape = param.shape_fn(sample_shape=output_shape)
      except NotImplementedError:
        self.skipTest('No shape function implemented for bijector {} '
                      'parameter {}.'.format(bijector_name, param_name))
      self.assertGreaterEqual(
          param.event_ndims,
          ps.rank_from_shape(param_shape) - sample_and_batch_ndims)

      if param.is_preferred:
        try:
          param_bijector = param.default_constraining_bijector_fn()
        except NotImplementedError:
          self.skipTest('No constraining bijector implemented for {} '
                        'parameter {}.'.format(bijector_name, param_name))
        unconstrained_shape = (
            param_bijector.inverse_event_shape_tensor(param_shape))
        unconstrained_param = samplers.normal(
            unconstrained_shape, seed=seeds[i])
        new_parameters[param_name] = param_bijector.forward(unconstrained_param)

        # Check that passing a float64 `eps` works with float64 parameters.
        b_float64 = params64[param_name].default_constraining_bijector_fn()
        b_float64(tf.cast(unconstrained_param, tf.float64))

    # Copy over any non-trainable parameters.
    new_parameters.update({
        k: v
        for (k, v) in bijector.parameters.items()
        if k in non_trainable_params
    })

    # Sanity check that we got valid parameters.
    new_parameters['validate_args'] = True
    new_bijector = type(bijector)(**new_parameters)
    self.evaluate(tf.group(*[v.initializer for v in new_bijector.variables]))
    xs = self._draw_domain_tensor(new_bijector, data, event_dim)
    self.evaluate(new_bijector.forward(xs))

  @parameterized.named_parameters(
      {'testcase_name': bname, 'bijector_name': bname}  # pylint:disable=g-complex-comprehension
      for bname in (set(bhps.INSTANTIABLE_BIJECTORS) -
                    set(AUTOVECTORIZATION_IS_BROKEN)))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testAutoVectorization(self, bijector_name, data):

    # TODO(b/150161911): reconcile numeric behavior of eager and graph mode.
    if tf.executing_eagerly():
      return

    bijector, event_dim = self._draw_bijector(
        bijector_name, data,
        batch_shape=[],  # Avoid conflict with vmap sample dimension.
        validate_args=False,  # Work around lack of `If` support in vmap.
        allowed_bijectors=(set(bhps.INSTANTIABLE_BIJECTORS) -
                           set(AUTOVECTORIZATION_IS_BROKEN)))
    atol = AUTOVECTORIZATION_ATOL[bijector_name]
    rtol = AUTOVECTORIZATION_RTOL[bijector_name]

    # Forward
    n = 3
    xs = self._draw_domain_tensor(bijector, data, event_dim, sample_shape=[n])
    ys = bijector.forward(xs)
    vectorized_ys = tf.vectorized_map(bijector.forward, xs,
                                      fallback_to_while_loop=False)
    self.assertAllClose(*self.evaluate((ys, vectorized_ys)),
                        atol=atol, rtol=rtol)

    # FLDJ
    event_ndims = data.draw(
        hps.integers(
            min_value=bijector.forward_min_event_ndims,
            max_value=ps.rank_from_shape(xs.shape) - 1))
    fldj_fn = functools.partial(bijector.forward_log_det_jacobian,
                                event_ndims=event_ndims)
    vectorized_fldj = tf.vectorized_map(fldj_fn, xs,
                                        fallback_to_while_loop=False)
    fldj = tf.broadcast_to(fldj_fn(xs), tf.shape(vectorized_fldj))
    self.assertAllClose(*self.evaluate((fldj, vectorized_fldj)),
                        atol=atol, rtol=rtol)

    # Inverse
    ys = self._draw_codomain_tensor(bijector, data, event_dim, sample_shape=[n])
    xs = bijector.inverse(ys)
    vectorized_xs = tf.vectorized_map(bijector.inverse, ys,
                                      fallback_to_while_loop=False)
    self.assertAllClose(*self.evaluate((xs, vectorized_xs)),
                        atol=atol, rtol=rtol)

    # ILDJ
    event_ndims = data.draw(
        hps.integers(
            min_value=bijector.inverse_min_event_ndims,
            max_value=ps.rank_from_shape(ys.shape) - 1))
    ildj_fn = functools.partial(bijector.inverse_log_det_jacobian,
                                event_ndims=event_ndims)
    vectorized_ildj = tf.vectorized_map(ildj_fn, ys,
                                        fallback_to_while_loop=False)
    ildj = tf.broadcast_to(ildj_fn(ys), tf.shape(vectorized_ildj))
    self.assertAllClose(*self.evaluate((ildj, vectorized_ildj)),
                        atol=atol, rtol=rtol)

  @parameterized.named_parameters(
      {'testcase_name': bname, 'bijector_name': bname}
      for bname in set(bhps.INSTANTIABLE_BIJECTORS))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testHashing(self, bijector_name, data):
    bijector_1, bijector_2 = data.draw(
        bijectors(bijector_name=bijector_name,
                  enable_vars=True, return_duplicate=True))
    self.assertEqual(hash(bijector_1), hash(bijector_2))

  @parameterized.named_parameters(
      {'testcase_name': bname, 'bijector_name': bname}
      for bname in set(bhps.INSTANTIABLE_BIJECTORS))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testEquality(self, bijector_name, data):
    bijector_1, bijector_2 = data.draw(
        bijectors(bijector_name=bijector_name,
                  enable_vars=True, return_duplicate=True))
    self.assertEqual(bijector_1, bijector_2)
    self.assertFalse(bijector_1 != bijector_2)  # pylint: disable=g-generic-assert

  @parameterized.named_parameters(
      {'testcase_name': bname, 'bijector_name': bname}  # pylint:disable=g-complex-comprehension
      for bname in (set(bhps.INSTANTIABLE_BIJECTORS) -
                    set(COMPOSITE_TENSOR_IS_BROKEN)))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testCompositeTensor(self, bijector_name, data):

    bijector, event_dim = self._draw_bijector(
        bijector_name, data,
        batch_shape=[],
        validate_args=True,
        allowed_bijectors=(set(bhps.INSTANTIABLE_BIJECTORS) -
                           set(COMPOSITE_TENSOR_IS_BROKEN)))

    if type(bijector) is invert_lib._Invert:  # pylint: disable=unidiomatic-typecheck
      if isinstance(bijector.bijector, tf.__internal__.CompositeTensor):
        raise TypeError('`_Invert` should wrap only non-`CompositeTensor` '
                        'bijectors.')
      self.skipTest('`_Invert` bijectors are not `CompositeTensor`s.')

    self.assertIsInstance(bijector, tf.__internal__.CompositeTensor)
    flat = tf.nest.flatten(bijector, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(bijector, flat, expand_composites=True)

    # Compare forward maps before and after compositing.
    n = 3
    xs = self._draw_domain_tensor(bijector, data, event_dim, sample_shape=[n])
    before_ys = bijector.forward(xs)
    after_ys = unflat.forward(xs)
    self.assertAllClose(*self.evaluate((before_ys, after_ys)))

    # Compare inverse maps before and after compositing.
    ys = self._draw_codomain_tensor(bijector, data, event_dim, sample_shape=[n])
    before_xs = bijector.inverse(ys)
    after_xs = unflat.inverse(ys)
    self.assertAllClose(*self.evaluate((before_xs, after_xs)))

    # Input to tf.function
    self.assertAllClose(
        before_ys,
        tf.function(lambda b: b.forward(xs))(bijector),
        rtol=COMPOSITE_TENSOR_RTOL[bijector_name],
        atol=COMPOSITE_TENSOR_ATOL[bijector_name])

    # Forward mapping: Check differentiation through forward mapping with
    # respect to the input and parameter variables.  Also check that any
    # variables are not referenced overmuch.
    xs = self._draw_domain_tensor(bijector, data, event_dim)
    wrt_vars = [xs] + [v for v in bijector.trainable_variables
                       if v.dtype.is_floating]
    with tf.GradientTape() as tape:
      tape.watch(wrt_vars)
      # TODO(b/73073515): Fix graph mode gradients with bijector caching.
      ys = bijector.forward(xs + 0)
    grads = tape.gradient(ys, wrt_vars)
    assert_no_none_grad(bijector, 'forward', wrt_vars, grads)

    self.assertConvertVariablesToTensorsWorks(bijector)


@test_util.test_all_tf_execution_regimes
class BijectorSlicingTest(test_util.TestCase):

  def _draw_domain_tensor(self, bijector, data, event_dim, sample_shape=()):
    codomain_event_shape = [event_dim] * bijector.inverse_min_event_ndims
    codomain_event_shape = constrain_inverse_shape(
        bijector, codomain_event_shape)
    shp = bijector.inverse_event_shape(codomain_event_shape).as_list()
    xs = tf.identity(data.draw(domain_tensors(bijector, shape=shp)), name='xs')

    return xs

  def _test_slicing(self, data, bijector_name, bijector, event_dim):
    batch_shape = bijector.experimental_batch_shape()
    slices = data.draw(tfp_hps.valid_slices(batch_shape))
    slice_str = 'bijector[{}]'.format(', '.join(tfp_hps.stringify_slices(
        slices)))
    # Make sure the slice string appears in Hypothesis' attempted example log
    hp.note('Using slice ' + slice_str)
    if not slices:  # Nothing further to check.
      return

    sliced_bijector = bijector[slices]

    # This should have no effect on a trivially instantiable bijector.
    if bijector_name in bhps.trivially_instantiable_bijectors():
      self.assertAllEqual(
          bijector.experimental_batch_shape(),
          sliced_bijector.experimental_batch_shape())
      return

    sliced_zeros = np.zeros(batch_shape)[slices]
    hp.note('Using sliced bijector {}.'.format(sliced_bijector))
    # Check that slicing modifies batch shape as expected.
    self.assertAllEqual(
        sliced_zeros.shape, sliced_bijector.experimental_batch_shape())

    if not sliced_zeros.size:
      # TODO(b/128924708): Fix bijectors that fail on degenerate empty
      #     shapes.
      return

    xs = self._draw_domain_tensor(bijector, data, event_dim)
    with tfp_hps.no_tf_rank_errors():
      forward = self.evaluate(bijector.forward(xs))
      sliced_bijector_forward = self.evaluate(sliced_bijector.forward(xs))

    # Come up with the slices for samples (which must also include event dims).
    forward_slices = (
        tuple(slices) if isinstance(slices, collections.abc.Sequence) else
        (slices,))
    if Ellipsis not in forward_slices:
      forward_slices += (Ellipsis,)
    forward_slices += tuple([slice(None)] * bijector.inverse_min_event_ndims)

    sliced_forward = forward[forward_slices]

    # Check that slicing the bijector computation has the same shape as
    # using the sliced bijector and computing results.
    self.assertAllEqual(sliced_forward.shape, sliced_bijector_forward.shape)
    self.assertAllClose(
        sliced_forward,
        sliced_bijector_forward,
        atol=SLICING_ATOL[bijector_name],
        rtol=SLICING_RTOL[bijector_name])

  @parameterized.named_parameters(
      {'testcase_name': bname, 'bijector_name': bname}
      for bname in (set(bhps.INSTANTIABLE_BIJECTORS) -
                    set(INSTANTIABLE_BUT_NOT_SLICEABLE)))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testBijectors(self, bijector_name, data):
    event_dim = data.draw(hps.integers(min_value=2, max_value=6))
    bijector = data.draw(
        bijectors(bijector_name=bijector_name,
                  event_dim=event_dim,
                  enable_vars=False,
                  batch_shape=data.draw(
                      tfp_hps.shapes(min_ndims=0, max_ndims=2, max_side=5)),
                  validate_args=True))

    # Check that all bijectors still register as non-iterable despite
    # defining __getitem__.  (Because __getitem__ magically makes an object
    # iterable for some reason.)
    with self.assertRaisesRegex(TypeError, 'not iterable'):
      iter(bijector)

    # Test slicing
    self._test_slicing(data, bijector_name, bijector, event_dim)


def ensure_nonzero(x):
  return tf.where(x < 1e-6, tf.constant(1e-3, x.dtype), x)


def fix_sigmoid(d):
  # Ensure that low < high.
  return dict(d, high=tfp_hps.ensure_high_gt_low(
      d['low'], d['high']))


def fix_softclip(d):
  # Ensure that low < high.
  return dict(d, high=tfp_hps.ensure_high_gt_low(
      d['low'], d['high']))


def fix_rational_quadratic(d):
  return dict(d, range_min=-1)


CONSTRAINTS = {
    'concentration':
        tfp_hps.softplus_plus_eps(),
    'concentration0':
        tfp_hps.softplus_plus_eps(),
    'concentration1':
        tfp_hps.softplus_plus_eps(),
    'hinge_softness':
        tfp_hps.softplus_plus_eps(),
    'power':
        # Restrict to positive since `Invert(Power(...))` tests the negation.
        tfp_hps.softplus_plus_eps(),
    'rate':
        tfp_hps.softplus_plus_eps(),
    'reflection_axis':
        tfp_hps.softplus_plus_eps(),
    'scale':
        tfp_hps.softplus_plus_eps(),
    'tailweight':
        tfp_hps.softplus_plus_eps(),
    'temperature':
        tfp_hps.softplus_plus_eps(eps=0.5),
    'RationalQuadraticSpline':
        fix_rational_quadratic,
    'Scale.scale':
        tfp_hps.softplus_plus_eps(),
    'ScaleMatvecDiag.scale_diag':
        tfp_hps.softplus_plus_eps(),
    'ScaleMatvecTriL.scale_tril':
        tfp_hps.lower_tril_positive_definite,
    # Lower bound concentration to 1e-1 to avoid
    # overflow for the inverse.
    'ShiftedGompertzCDF.concentration':
        lambda x: tf.math.softplus(x) + 1e-1,
    'Sigmoid': fix_sigmoid,
    'SoftClip': fix_softclip,
    'bin_widths':
        bhps.spline_bin_size_constraint,
    'bin_heights':
        bhps.spline_bin_size_constraint,
    'knot_slopes':
        bhps.spline_slope_constraint,
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


def constrain_params(params_unconstrained, bijector_name):
  """Constrains a parameters dictionary to a bijector's parameter space."""
  # Constrain them to legal values
  params_constrained = constraint_for(bijector_name)(params_unconstrained)

  # Sometimes the "bijector constraint" fn may replace c2t-tracking
  # DeferredTensor params with Tensor params (e.g. fix_triangular). In such
  # cases, we preserve the c2t-tracking DeferredTensors by wrapping them but
  # ignoring the value.  We similarly reinstate raw tf.Variables, so they
  # appear in the bijector's `variables` list and can be initialized.
  for k in params_constrained:
    if (k in params_unconstrained and
        isinstance(params_unconstrained[k],
                   (tfp_util.DeferredTensor, tf.Variable))
        and params_unconstrained[k] is not params_constrained[k]):

      def constrained_value(v, val=params_constrained[k]):  # pylint: disable=cell-var-from-loop
        # While the gradient to v will be 0, we only care about the c2t
        # counts.
        return v * 0 + val

      params_constrained[k] = tfp_util.DeferredTensor(
          params_unconstrained[k], constrained_value)
  hp.note('Forming bijector {} with constrained parameters {}'.format(
      bijector_name, params_constrained))
  return params_constrained


if __name__ == '__main__':
  np.set_printoptions(floatmode='unique', precision=None)
  test_util.main()
