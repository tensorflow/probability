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
import tensorflow_probability as tfp
from tensorflow_probability.python.bijectors import hypothesis_testlib as bijector_hps
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

tfb = tfp.bijectors
tfd = tfp.distributions

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
    bijector_name = draw(hps.one_of(map(hps.just, TF2_FRIENDLY_BIJECTORS)))
  if batch_shape is None:
    batch_shape = draw(tfp_hps.shapes())
  if event_dim is None:
    event_dim = draw(hps.integers(min_value=2, max_value=6))
  if bijector_name == 'Invert':
    underlying_name = draw(
        hps.one_of(map(hps.just,
                       set(TF2_FRIENDLY_BIJECTORS) - {'Invert'})))
    underlying = draw(
        bijectors(
            bijector_name=underlying_name,
            batch_shape=batch_shape,
            event_dim=event_dim,
            enable_vars=enable_vars))
    return tfb.Invert(underlying, validate_args=True)
  if bijector_name == 'Inline':
    if enable_vars:
      scale = tf.Variable(1., name='scale')
    else:
      scale = 2.
    b = tfb.AffineScalar(scale=scale)

    inline = tfb.Inline(
        forward_fn=b.forward,
        inverse_fn=b.inverse,
        forward_log_det_jacobian_fn=lambda x: b.forward_log_det_jacobian(  # pylint: disable=g-long-lambda
            x, event_ndims=b.forward_min_event_ndims),
        forward_min_event_ndims=b.forward_min_event_ndims,
        is_constant_jacobian=b.is_constant_jacobian,
    )
    inline.b = b
    return inline
  if bijector_name == 'DiscreteCosineTransform':
    dct_type = draw(hps.integers(min_value=2, max_value=3))
    return tfb.DiscreteCosineTransform(
        validate_args=True, dct_type=dct_type)
  if bijector_name == 'PowerTransform':
    power = draw(hps.floats(min_value=0., max_value=10.))
    return tfb.PowerTransform(validate_args=True, power=power)

  bijector_params = draw(
      broadcasting_params(bijector_name, batch_shape, event_dim=event_dim,
                          enable_vars=enable_vars))
  ctor = getattr(tfb, bijector_name)
  return ctor(validate_args=True, **bijector_params)


Support = tfp_hps.Support


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
class BijectorPropertiesTest(tf.test.TestCase, parameterized.TestCase):

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

    # Forward mapping: Check differentiation through forward mapping with
    # respect to the input and parameter variables.  Also check that any
    # variables are not referenced overmuch.
    # TODO(axch): Would be nice to get rid of all this shape inference logic and
    # just rely on a notion of batch and event shape for bijectors, so we can
    # pass those through `domain_tensors` and `codomain_tensors` and use
    # `tensors_in_support`.  However, `RationalQuadraticSpline` behaves weirdly
    # somehow and I got confused.
    shp = bijector.inverse_event_shape([event_dim] *
                                       bijector.inverse_min_event_ndims)
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

    # FLDJ: Check differentiation through forward log det jacobian with
    # respect to the input and parameter variables.  Also check that any
    # variables are not referenced overmuch.
    event_ndims = data.draw(
        hps.integers(
            min_value=bijector.forward_min_event_ndims,
            max_value=bijector.forward_event_shape(xs.shape).ndims))
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
    shp = bijector.forward_event_shape([event_dim] *
                                       bijector.forward_min_event_ndims)
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
            max_value=bijector.inverse_event_shape(ys.shape).ndims))
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
        xs = bijector.inverse_log_det_jacobian(ys + 0, event_ndims=event_ndims)
    grads = tape.gradient(xs, wrt_vars)
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
    'scale':
        tfp_hps.softplus_plus_eps(),
    'tailweight':
        tfp_hps.softplus_plus_eps(),
    'AffineScalar.scale':
        tfp_hps.softplus_plus_eps(),
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
