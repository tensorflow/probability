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
"""Utilities for hypothesis testing of bijectors."""

import collections
import inspect

from absl import logging
import hypothesis.strategies as hps

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.distributions import lkj
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import tensorshape_util

SPECIAL_BIJECTORS = [
    'Composition',
    'FillScaleTriL',
    'FillTriangular',
    'Inline',
    'Invert',
    'TransformDiagonal',
]

NONINSTANTIABLE_BIJECTORS = [
    'AbsoluteValue',  # Non-invertible
    # Base classes.
    'AutoCompositeTensorBijector',
    'Bijector',
    # TODO(b/199173655): Add hypothesis generation for the below bijectors.
    'BatchNormalization',
    'Blockwise',
    'Chain',
    'CorrelationCholesky',
    'FFJORD',
    'Glow',
    'JointMap',
    'LambertWTail',
    'MaskedAutoregressiveFlow',
    'Pad',
    'RationalQuadraticSpline',
    'RealNVP',
    'Restructure',
    'ScaleMatvecLinearOperator',
    'ScaleMatvecLinearOperatorBlock',
    'Split',
    # Tests need to be added.
    # TODO(b/199174510): Remove non-invertible matrix errors.
    'CholeskyToInvCholesky',
    # TODO(b/199175367): SoftClip errors from having non-trivial batch shapes
    # in DeferredTensor.
    'SoftClip',
]

TRANSFORM_DIAGONAL_ALLOWLIST = {
    'DiscreteCosineTransform',
    'Exp',
    'Expm1',
    'GompertzCDF',
    'GumbelCDF',
    'GeneralizedExtremeValueCDF',
    'GeneralizedPareto',
    'Identity',
    'Inline',
    'KumaraswamyCDF',
    'MoyalCDF',
    'NormalCDF',
    'PowerTransform',
    'Power',
    'RayleighCDF',
    'Reciprocal',
    'Scale',
    'ScaleMatvecDiag',
    'ScaleMatvecLU',
    'ScaleMatvecTriL',
    'Shift',
    'ShiftedGompertzCDF',
    'Sigmoid',
    'Sinh',
    'SinhArcsinh',
    'Softplus',
    'Softsign',
    'Square',
    'Tanh',
    'WeibullCDF',
}


class BijectorInfo(collections.namedtuple(
    'BijectorInfo', ['cls', 'params_event_ndims'])):
  """Sufficient information to instantiate a Bijector.

  To wit

  - The Python class `cls` giving the class, and
  - A Python dict `params_event_ndims` giving the event dimensions for the
    parameters (so that parameters can be built with predictable batch shapes).

  Specifically, the `params_event_ndims` dict maps string parameter names to
  Python integers.  Each integer gives how many (trailing) dimensions of that
  parameter are part of the event.
  """
  __slots__ = ()


def _instantiable_base_bijectors():
  """Computes the table of mechanically instantiable base Bijectors.

  A Bijector is mechanically instantiable if

  - The class appears as a symbol binding in `tfp.bijectors`;
  - The class defines a `_params_event_ndims` method (necessary
    to generate parameter Tensors with predictable batch shapes); and
  - The name is not blocklisted in `SPECIAL_BIJECTORS` or
    `NONINSTANTIABLE_BIJECTORS`.

  Returns:
    instantiable_base_bijectors: A Python dict mapping bijector name
      (as a string) to a `BijectorInfo` carrying the information necessary to
      instantiate it.
  """
  result = {}
  for bijector_name in dir(tfb):
    bijector_class = getattr(tfb, bijector_name)
    if (not inspect.isclass(bijector_class) or
        not issubclass(bijector_class, tfb.Bijector) or
        bijector_name in SPECIAL_BIJECTORS or
        bijector_name in NONINSTANTIABLE_BIJECTORS):
      continue
    try:
      params_event_ndims = {
          k: p.event_ndims
          for (k, p) in bijector_class.parameter_properties().items()
          if p.is_tensor and p.event_ndims is not None
      }
      has_concrete_event_ndims = all(
          isinstance(nd, int) for nd in params_event_ndims.values())
    except NotImplementedError:
      has_concrete_event_ndims = False
    if has_concrete_event_ndims:
      result[bijector_name] = BijectorInfo(bijector_class, params_event_ndims)
    else:
      logging.warning(
          'Unable to test tfb.%s: `parameter_properties()` is not '
          'implemented or does not define concrete (integer) `event_ndims` '
          'for all parameters.',
          bijector_name)
  return result


# INSTANTIABLE_BIJECTORS is a map from str->BijectorInfo
INSTANTIABLE_BIJECTORS = _instantiable_base_bijectors()


TRIVIALLY_INSTANTIABLE_BIJECTORS = None


def trivially_instantiable_bijectors():
  """Identifies bijectors that are instantiable without any arguments."""
  global TRIVIALLY_INSTANTIABLE_BIJECTORS
  if TRIVIALLY_INSTANTIABLE_BIJECTORS is not None:
    return TRIVIALLY_INSTANTIABLE_BIJECTORS

  result = {}
  for bijector_name in dir(tfb):
    bijector_class = getattr(tfb, bijector_name)
    if (not inspect.isclass(bijector_class) or
        not issubclass(bijector_class, tfb.Bijector) or
        bijector_name in SPECIAL_BIJECTORS or
        bijector_name in NONINSTANTIABLE_BIJECTORS):
      continue

    if not bijector_class.parameter_properties():
      if not bijector_class()._is_injective:  # pylint: disable=protected-access
        continue
      result[bijector_name] = bijector_class

  result['Invert'] = tfb.Invert
  # Add these bijectors since they subclass Invert but actually have no
  # parameters.
  result['Log'] = tfb.Log
  result['Log1p'] = tfb.Log1p

  for bijector_name in sorted(result):
    logging.warning('Trivially supported bijectors: tfb.%s', bijector_name)

  TRIVIALLY_INSTANTIABLE_BIJECTORS = result
  return TRIVIALLY_INSTANTIABLE_BIJECTORS


class BijectorSupport(collections.namedtuple(
    'BijectorSupport', ['forward', 'inverse'])):
  """Specification of the domain and codomain of a bijector.

  The `forward` slot is the support of the forward computation, i.e., the
  domain, and the `inverse` slot is the support of the inverse computation,
  i.e., the codomain.
  """
  __slots__ = ()

  def invert(self):
    """Returns the inverse of this `BijectorSupport`."""
    return BijectorSupport(self.inverse, self.forward)


BIJECTOR_SUPPORTS = None


def bijector_supports():
  """Returns a dict of supports for each instantiable bijector.

  Warns if any `instantiable_bijectors` are found to have no declared supports,
  once per Python process.

  Returns:
    supports: Python `dict` mapping `str` bijector name to the corresponding
      `BijectorSupport` object.
  """
  global BIJECTOR_SUPPORTS
  if BIJECTOR_SUPPORTS is not None:
    return BIJECTOR_SUPPORTS
  Support = tfp_hps.Support  # pylint: disable=invalid-name
  supports = {
      '_Invert':
          BijectorSupport(Support.OTHER, Support.OTHER),
      'Ascending':
          BijectorSupport(Support.VECTOR_UNCONSTRAINED,
                          Support.VECTOR_STRICTLY_INCREASING),
      'BatchNormalization':
          BijectorSupport(Support.VECTOR_UNCONSTRAINED,
                          Support.VECTOR_UNCONSTRAINED),
      'CholeskyOuterProduct':
          BijectorSupport(Support.MATRIX_LOWER_TRIL_POSITIVE_DEFINITE,
                          Support.MATRIX_POSITIVE_DEFINITE),
      'CholeskyToInvCholesky':
          BijectorSupport(Support.MATRIX_LOWER_TRIL_POSITIVE_DEFINITE,
                          Support.MATRIX_LOWER_TRIL_POSITIVE_DEFINITE),
      'CorrelationCholesky':
          BijectorSupport(Support.VECTOR_SIZE_TRIANGULAR,
                          Support.CORRELATION_CHOLESKY),
      'Cumsum':
          BijectorSupport(Support.VECTOR_UNCONSTRAINED,
                          Support.VECTOR_UNCONSTRAINED),
      'DiscreteCosineTransform':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_UNCONSTRAINED),
      'Exp':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_POSITIVE),
      'Expm1':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED, Support.SCALAR_GT_NEG1),
      'FillScaleTriL':
          BijectorSupport(Support.VECTOR_SIZE_TRIANGULAR,
                          Support.MATRIX_LOWER_TRIL_POSITIVE_DEFINITE),
      'FillTriangular':
          BijectorSupport(Support.VECTOR_SIZE_TRIANGULAR,
                          Support.MATRIX_LOWER_TRIL),
      'FrechetCDF':  # The domain is parameter dependent.
          BijectorSupport(Support.OTHER, Support.SCALAR_IN_0_1),
      'GeneralizedExtremeValueCDF':  # The domain is parameter dependent.
          BijectorSupport(Support.OTHER, Support.SCALAR_IN_0_1),
      'GeneralizedPareto':  # The range is parameter dependent.
          BijectorSupport(Support.SCALAR_UNCONSTRAINED, Support.OTHER),
      'GompertzCDF':
          BijectorSupport(Support.SCALAR_POSITIVE, Support.SCALAR_IN_0_1),
      'GumbelCDF':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED, Support.SCALAR_IN_0_1),
      'Householder':
          BijectorSupport(Support.VECTOR_UNCONSTRAINED,
                          Support.VECTOR_UNCONSTRAINED),
      'Identity':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_UNCONSTRAINED),
      'Inline':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_UNCONSTRAINED),
      'Invert':
          BijectorSupport(Support.OTHER, Support.OTHER),
      'IteratedSigmoidCentered':
          BijectorSupport(Support.VECTOR_UNCONSTRAINED,
                          Support.VECTOR_POSITIVE_WITH_L1_NORM_1_SIZE_GT1),
      'KumaraswamyCDF':
          BijectorSupport(Support.SCALAR_IN_0_1, Support.SCALAR_IN_0_1),
      'Log':
          BijectorSupport(Support.SCALAR_POSITIVE,
                          Support.SCALAR_UNCONSTRAINED),
      'Log1p':
          BijectorSupport(Support.SCALAR_GT_NEG1, Support.SCALAR_UNCONSTRAINED),
      'MatrixInverseTriL':
          BijectorSupport(Support.MATRIX_LOWER_TRIL_POSITIVE_DEFINITE,
                          Support.MATRIX_LOWER_TRIL_POSITIVE_DEFINITE),
      'MatvecLU':
          BijectorSupport(Support.VECTOR_UNCONSTRAINED,
                          Support.VECTOR_UNCONSTRAINED),
      'MoyalCDF':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED, Support.SCALAR_IN_0_1),
      'NormalCDF':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED, Support.SCALAR_IN_0_1),
      'Ordered':
          BijectorSupport(Support.VECTOR_STRICTLY_INCREASING,
                          Support.VECTOR_UNCONSTRAINED),
      'Permute':
          BijectorSupport(Support.VECTOR_UNCONSTRAINED,
                          Support.VECTOR_UNCONSTRAINED),
      'Power':
          BijectorSupport(Support.SCALAR_POSITIVE,
                          Support.SCALAR_POSITIVE),
      'PowerTransform':  # The domain is parameter dependent.
          BijectorSupport(Support.OTHER, Support.SCALAR_POSITIVE),
      'RationalQuadraticSpline':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_UNCONSTRAINED),
      'RayleighCDF':
          BijectorSupport(Support.SCALAR_NON_NEGATIVE,
                          Support.SCALAR_IN_0_1),
      'Reciprocal':
          BijectorSupport(Support.SCALAR_NON_ZERO, Support.SCALAR_NON_ZERO),
      'Reshape':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_UNCONSTRAINED),
      'Scale':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_UNCONSTRAINED),
      'ScaleMatvecDiag':
          BijectorSupport(Support.VECTOR_UNCONSTRAINED,
                          Support.VECTOR_UNCONSTRAINED),
      'ScaleMatvecLU':
          BijectorSupport(Support.VECTOR_UNCONSTRAINED,
                          Support.VECTOR_UNCONSTRAINED),
      'ScaleMatvecTriL':
          BijectorSupport(Support.VECTOR_UNCONSTRAINED,
                          Support.VECTOR_UNCONSTRAINED),
      'Shift':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_UNCONSTRAINED),
      'ShiftedGompertzCDF':
          BijectorSupport(Support.SCALAR_POSITIVE, Support.SCALAR_IN_0_1),
      'Sigmoid':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED, Support.SCALAR_IN_0_1),
      'Sinh':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_UNCONSTRAINED),
      'SinhArcsinh':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_UNCONSTRAINED),
      'SoftClip':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.OTHER),
      'Softfloor':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_UNCONSTRAINED),
      'Softplus':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_POSITIVE),
      'Softsign':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_IN_NEG1_1),
      'SoftmaxCentered':
          BijectorSupport(Support.VECTOR_UNCONSTRAINED,
                          Support.VECTOR_POSITIVE_WITH_L1_NORM_1_SIZE_GT1),
      'Square':
          BijectorSupport(Support.SCALAR_NON_NEGATIVE,
                          Support.SCALAR_NON_NEGATIVE),
      'Tanh':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_IN_NEG1_1),
      'TransformDiagonal':
          BijectorSupport(Support.MATRIX_UNCONSTRAINED, Support.OTHER),
      'Transpose':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_UNCONSTRAINED),
      'UnitVector':
          BijectorSupport(Support.VECTOR_UNCONSTRAINED,
                          Support.VECTOR_WITH_L2_NORM_1_SIZE_GT1),
      'WeibullCDF':
          BijectorSupport(Support.SCALAR_NON_NEGATIVE, Support.SCALAR_IN_0_1),
  }
  missing_keys = set(INSTANTIABLE_BIJECTORS.keys()) - set(supports.keys())
  if missing_keys:
    raise ValueError('Missing bijector supports: {}'.format(missing_keys))
  BIJECTOR_SUPPORTS = supports
  return BIJECTOR_SUPPORTS


@hps.composite
def unconstrained_bijectors(draw, max_forward_event_ndims=None,
                            must_preserve_event_ndims=False,
                            validate_args=True):
  """Strategy for drawing forward-unconstrained bijectors.

  The bijectors emitted by this are those whose `forward` computation
  can act on all of R^n, with n <= `max_forward_event_ndims`.

  Args:
    draw: Strategy sampler supplied by `@hps.composite`.
    max_forward_event_ndims: Optional python `int`, maximum acceptable bijector
      `forward_event_ndims`.
    must_preserve_event_ndims: Optional python `bool`, `True` if the returned
      bijector must preserve the rank of the event.
    validate_args: Python `bool`; whether to enable runtime assertions.

  Returns:
    unconstrained: A strategy for drawing such bijectors.
  """
  if max_forward_event_ndims is None:
    max_forward_event_ndims = float('inf')

  ndims_by_prefix = dict(SCALAR=0, VECTOR=1, MATRIX=2, OTHER=-1)

  def is_acceptable(support):
    """Determines if a `BijectorSupport` object is acceptable."""
    if 'UNCONSTRAINED' not in support.forward:
      return False
    forward_prefix = support.forward.split('_')[0]
    if ndims_by_prefix[forward_prefix] > max_forward_event_ndims:
      return False
    if must_preserve_event_ndims:
      inverse_prefix = support.inverse.split('_')[0]
      if ndims_by_prefix[forward_prefix] != ndims_by_prefix[inverse_prefix]:
        return False
    return True

  supports = bijector_supports()
  bijectors = trivially_instantiable_bijectors()
  acceptable_keys = sorted([k for k in bijectors
                            if k == 'Invert' or is_acceptable(supports[k])])
  bijector_name = draw(hps.sampled_from(acceptable_keys))
  if bijector_name == 'Invert':
    acceptable_keys = [k for k in bijectors
                       if is_acceptable(supports[k].invert())]
    underlying = draw(hps.sampled_from(acceptable_keys))
    underlying = bijectors[underlying](validate_args=validate_args)
    return tfb.Invert(underlying, validate_args=validate_args)
  return bijectors[bijector_name](validate_args=validate_args)


def distribution_eligilibility_filter_for(bijector):
  """Returns a function which filters distribution names, where possible."""
  if isinstance(bijector, tfb.CorrelationCholesky):
    return 'LKJ'.__eq__

  return lambda name: True


def distribution_filter_for(bijector):
  """Returns a function checking Distribution compatibility with this bijector.

  That is, `distribution_filter_for(bijector)(dist) == True` implies
  that `bijector` can act on `dist` (i.e., they are safe to compose with
  `TransformedDistribution`).

  TODO(bjp): Make this sensitive to supports.  Currently assumes `bijector` acts
  on an unconstrained space, and just checks compatible ranks.

  Args:
    bijector: A `Bijector` instance to check compatibility with.

  Returns:
    filter: A Python callable filtering Distributions for compatibility with
      this bijector.
  """
  if isinstance(bijector, tfb.CholeskyToInvCholesky):

    def additional_check(dist):
      return (tensorshape_util.rank(dist.event_shape) == 2 and
              int(dist.event_shape[0]) == int(dist.event_shape[1]))
  elif isinstance(bijector, tfb.CorrelationCholesky):

    def additional_check(dist):
      # The isinstance check will be redundant when the
      # `distribution_eligilibility_filter_for` above has been used, but we keep
      # it here for safety.
      return isinstance(dist, lkj.LKJ) and dist.input_output_cholesky
  else:
    additional_check = lambda dist: True

  def distribution_filter(dist):
    if not dtype_util.is_floating(dist.dtype):
      return False
    if bijector.forward_min_event_ndims > tensorshape_util.rank(
        dist.event_shape):
      return False
    return additional_check(dist)

  return distribution_filter


def padded(t, lhs, rhs=None):
  """Left pads and optionally right pads the innermost axis of `t`."""
  t = tf.convert_to_tensor(t)
  lhs = tf.convert_to_tensor(lhs, dtype=t.dtype)
  zeros = tf.zeros([tf.rank(t) - 1, 2], dtype=tf.int32)
  lhs_paddings = tf.concat([zeros, [[1, 0]]], axis=0)
  result = tf.pad(t, paddings=lhs_paddings, constant_values=lhs)
  if rhs is not None:
    rhs = tf.convert_to_tensor(rhs, dtype=t.dtype)
    rhs_paddings = tf.concat([zeros, [[0, 1]]], axis=0)
    result = tf.pad(result, paddings=rhs_paddings, constant_values=rhs)
  return result


def spline_bin_size_constraint(x, lo=-1, hi=1, dtype=tf.float32):
  """Maps innermost axis of `x` to positive values."""
  nbins = tf.cast(tf.shape(x)[-1], dtype)
  min_width = 1e-2
  scale = hi - lo - nbins * min_width
  return tf.math.softmax(tf.cast(x, dtype)) * scale + min_width


def spline_slope_constraint(s, dtype=tf.float32):
  """Maps `s` to all positive with `s[..., 0] == s[..., -1] == 1`."""
  # Slice off a position since this is nknots - 2 vs nknots - 1 for bin sizes.
  min_slope = 1e-2
  return tf.math.softplus(tf.cast(s[..., :-1], dtype)) + min_slope


def power_transform_constraint(power):
  """Maps `s` to [-1 / power, inf)."""
  def constrain(x):
    if power == 0:
      return x
    return tf.math.softplus(x) - 1. / power
  return constrain


def frechet_constraint(loc):
  """Maps `s` to [loc, inf)."""
  def constrain(x):
    return loc + tf.math.softplus(x)
  return constrain


def gev_constraint(loc, scale, conc):
  """Maps `s` to support based on `loc`, `scale` and `conc`."""
  def constrain(x):
    c = tf.convert_to_tensor(conc)
    # We intentionally compute the endpoint with (1.0 / concentration) * scale,
    # for the same reason as in GeneralizedExtremeValueCDF._maybe_assert_valid_x
    endpoint = loc - (1.0 / c) * scale
    return tf.where(c > 0.,
                    tf.math.softplus(x) + endpoint,
                    tf.where(
                        tf.equal(0., c),
                        x, endpoint - tf.math.softplus(x)))
  return constrain


def generalized_pareto_constraint(loc, scale, conc):
  """Maps `s` to support based on `loc`, `scale` and `conc`."""
  def constrain(x):
    conc_ = tf.convert_to_tensor(conc)
    loc_ = tf.convert_to_tensor(loc)
    return tf.where(conc_ >= 0.,
                    tf.math.softplus(x) + loc_,
                    loc_ - tf.math.sigmoid(x) * scale / conc_)
  return constrain


def softclip_constraint(low, high):
  """Maps `s` to support based on `low` and `high`."""
  def constrain(x):
    low_ = tf.convert_to_tensor(low)
    high_ = tf.convert_to_tensor(high)
    # Ensure the values are within (low, high).
    return (0.5 * (high_ - low_) * tf.math.sigmoid(x) +
            (high_ - low_) / 4 + low_)
  return constrain
