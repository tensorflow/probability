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
    # 'AffineScalar',  # TODO(b/136265958): Cached jacobian breaks grad to scale
    'CholeskyOuterProduct',
    'Cumsum',
    'Exp',
    'Expm1',
    'Gumbel',
    'Identity',
    'IteratedSigmoidCentered',
    'Invert',
    'Kumaraswamy',
    'NormalCDF',
    'Ordered',
    'Reciprocal',
    'Sigmoid',
    'SinhArcsinh',
    'Softsign',
    'Square',
    'Tanh',
)

BIJECTOR_PARAMS_NDIMS = {
    'AffineScalar': dict(shift=0, scale=0, log_scale=0),
    'Gumbel': dict(loc=0, scale=0),
    'Kumaraswamy': dict(concentration1=0, concentration0=0),
    'SinhArcsinh': dict(skewness=0, tailweight=0),
}

MUTEX_PARAMS = (
    set(['scale', 'log_scale']),
)

NO_LDJ_GRADS_EXPECTED = {
    'AffineScalar': {'[arg]', 'shift'},
    'Cumsum': {'[arg]'},
    'Gumbel': {'loc'},
    'Identity': {'[arg]'},
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
          event_dim=event_dim,
          enable_vars=enable_vars,
          params_event_ndims=params_event_ndims,
          constraint_fn_for=_constraint,
          mutex_params=MUTEX_PARAMS))


@hps.composite
def bijectors(draw, bijector_name=None, batch_shape=None, enable_vars=False):
  if bijector_name is None:
    bijector_name = draw(hps.one_of(map(hps.just, TF2_FRIENDLY_BIJECTORS)))
  if batch_shape is None:
    batch_shape = draw(tfp_hps.batch_shapes())
  if bijector_name == 'Invert':
    underlying_name = draw(
        hps.one_of(map(hps.just,
                       set(TF2_FRIENDLY_BIJECTORS) - {'Invert'})))
    underlying, batch_shape = draw(
        bijectors(
            bijector_name=underlying_name,
            batch_shape=batch_shape,
            enable_vars=enable_vars))
    return tfb.Invert(underlying, validate_args=True), batch_shape

  bijector_params = draw(
      broadcasting_params(bijector_name, batch_shape, enable_vars=enable_vars))
  ctor = getattr(tfb, bijector_name)
  return ctor(validate_args=True, **bijector_params), batch_shape


Support = bijector_hps.Support


def scalar_constrainer(support):

  def nonzero(x):
    return tf.where(tf.equal(x, 0), 1e-6, x)

  constrainers = {
      Support.SCALAR_IN_0_1: tf.math.sigmoid,
      Support.SCALAR_GT_NEG1: tfp_hps.softplus_plus_eps(-1 + 1e-6),
      Support.SCALAR_NON_ZERO: nonzero,
      Support.SCALAR_IN_NEG1_1: lambda x: tf.math.tanh(x) * (1 - 1e-6),
      Support.SCALAR_NON_NEGATIVE: tf.math.softplus,
      Support.SCALAR_POSITIVE: tfp_hps.softplus_plus_eps(),
      Support.SCALAR_UNCONSTRAINED: tf.identity,
  }
  return constrainers[support]


def vector_constrainer(support):

  def l1norm(x):
    x = tf.concat([x, tf.ones_like(x[..., :1]) * 1e-6], axis=-1)
    x = x / tf.linalg.norm(x, ord=1, axis=-1, keepdims=True)
    return x

  constrainers = {
      Support.VECTOR_UNCONSTRAINED:
          tfp_hps.identity_fn,
      Support.VECTOR_STRICTLY_INCREASING:
          lambda x: tf.cumsum(tf.abs(x) + 1e-3, axis=-1),
      Support.VECTOR_WITH_L1_NORM_1_SIZE_GT1:
          l1norm,
  }
  return constrainers[support]


def matrix_constrainer(support):

  constrainers = {
      Support.MATRIX_POSITIVE_DEFINITE:
          tfp_hps.positive_definite,
      Support.MATRIX_LOWER_TRIL_POSITIVE_DEFINITE:
          tfp_hps.lower_tril_positive_definite,
  }
  return constrainers[support]


def constrainer(support):
  if support.startswith('SCALAR_'):
    return scalar_constrainer(support)
  if support.startswith('VECTOR_'):
    return vector_constrainer(support)
  if support.startswith('MATRIX_'):
    return matrix_constrainer(support)
  raise NotImplementedError(support)


@hps.composite
def domain_tensors(draw, bijector, shape=None):
  if is_invert(bijector):
    return draw(codomain_tensors(bijector.bijector, shape))
  if shape is None:
    shape = draw(tfp_hps.batch_shapes())
  bijector_name = type(bijector).__name__
  support = bijector_hps.bijector_supports()[bijector_name].forward
  constraint_fn = constrainer(support)
  return draw(tfp_hps.constrained_tensors(constraint_fn, shape))


@hps.composite
def codomain_tensors(draw, bijector, shape=None):
  if is_invert(bijector):
    return draw(domain_tensors(bijector.bijector, shape))
  if shape is None:
    shape = draw(tfp_hps.batch_shapes())
  bijector_name = type(bijector).__name__
  support = bijector_hps.bijector_supports()[bijector_name].inverse
  constraint_fn = constrainer(support)
  return draw(tfp_hps.constrained_tensors(constraint_fn, shape))


def assert_no_none_grad(bijector, method, wrt_vars, grads):
  for var, grad in zip(wrt_vars, grads):
    if 'log_det_jacobian' in method:
      if tensor_util.is_mutable(var):
        # We check tensor_util.is_mutable to accounts for xs/ys being in vars.
        var_name = var.name.rstrip('_0123456789:')
      else:
        var_name = '[arg]'
      to_check = bijector.bijector if is_invert(bijector) else bijector
      if var_name in NO_LDJ_GRADS_EXPECTED.get(type(to_check).__name__, ()):
        continue
    if grad is None:
      raise AssertionError('Missing `{}` -> {} grad for bijector {}'.format(
          method, var, bijector))


@test_util.run_all_in_graph_and_eager_modes
class BijectorPropertiesTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters((bname,) for bname in TF2_FRIENDLY_BIJECTORS)
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testBijector(self, bijector_name, data):
    if tf.executing_eagerly() != (FLAGS.tf_mode == 'eager'):
      return
    bijector, batch_shape = data.draw(
        bijectors(bijector_name=bijector_name, enable_vars=True))
    del batch_shape

    event_dim = data.draw(hps.integers(min_value=2, max_value=6))

    # Forward mapping.
    shp = bijector.inverse_event_shape([event_dim] *
                                       bijector.inverse_min_event_ndims)
    shp = tensorshape_util.concatenate(
        data.draw(
            tfp_hps.broadcast_compatible_shape(
                shp[:shp.ndims - bijector.forward_min_event_ndims])),
        shp[shp.ndims - bijector.forward_min_event_ndims:])
    xs = tf.identity(data.draw(domain_tensors(bijector, shape=shp)), name='xs')
    wrt_vars = [xs] + list(bijector.variables)
    with tf.GradientTape() as tape:
      with tfp_hps.assert_no_excessive_var_usage(
          'method `forward` of {}'.format(bijector)):
        tape.watch(wrt_vars)
        # TODO(b/73073515): Fix graph mode gradients with bijector caching.
        ys = bijector.forward(xs + 0)
    grads = tape.gradient(ys, wrt_vars)
    assert_no_none_grad(bijector, 'forward', wrt_vars, grads)

    # FLDJ.
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

    # Inverse mapping.
    shp = bijector.forward_event_shape([event_dim] *
                                       bijector.forward_min_event_ndims)
    shp = tensorshape_util.concatenate(
        data.draw(
            tfp_hps.broadcast_compatible_shape(
                shp[:shp.ndims - bijector.inverse_min_event_ndims])),
        shp[shp.ndims - bijector.inverse_min_event_ndims:])
    ys = tf.identity(
        data.draw(codomain_tensors(bijector, shape=shp)), name='ys')
    wrt_vars = [ys] + list(bijector.variables)
    with tf.GradientTape() as tape:
      with tfp_hps.assert_no_excessive_var_usage(
          'method `inverse` of {}'.format(bijector)):
        tape.watch(wrt_vars)
        # TODO(b/73073515): Fix graph mode gradients with bijector caching.
        xs = bijector.inverse(ys + 0)
    grads = tape.gradient(xs, wrt_vars)
    assert_no_none_grad(bijector, 'inverse', wrt_vars, grads)

    # ILDJ.
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


CONSTRAINTS = {
    'concentration0': tfp_hps.softplus_plus_eps(),
    'concentration1': tfp_hps.softplus_plus_eps(),
    'scale': tfp_hps.softplus_plus_eps(),
    'tailweight': tfp_hps.softplus_plus_eps(),
    'AffineScalar.scale': tfp_hps.softplus_plus_eps(),
}


def constraint_for(bijector_name=None, param=None):
  if param is not None:
    return CONSTRAINTS.get('{}.{}'.format(bijector_name, param),
                           CONSTRAINTS.get(param, tfp_hps.identity_fn))
  return CONSTRAINTS.get(bijector_name, tfp_hps.identity_fn)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
