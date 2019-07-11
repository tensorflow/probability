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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import inspect

from absl import logging
import hypothesis.strategies as hps
import six

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensorshape_util

tfb = tfp.bijectors
tfd = tfp.distributions

SPECIAL_BIJECTORS = ['Invert']

# INSTANTIABLE_BIJECTORS is a map from str->(BijectorClass,)
INSTANTIABLE_BIJECTORS = None


def instantiable_bijectors():
  """Identifies bijectors that are trivially instantiable.

  Here, "trivially" means things like `Exp` for which no parameters need to be
  generated; i.e., the only arguments to the constructor are `self`, `name`, and
  `validate_args`.

  This finds the bijectors by traversing the `tfp.bijectors` namespace.  The
  traversal is cached so it only happens once per Python process.

  Returns:
    instantiable: A Python `dict` mapping the `str` bijector name to a singleton
      tuple containing the bijector class object.
  """
  global INSTANTIABLE_BIJECTORS
  if INSTANTIABLE_BIJECTORS is not None:
    return INSTANTIABLE_BIJECTORS

  result = {}
  for (bijector_name, bijector_class) in six.iteritems(tfb.__dict__):
    if (not inspect.isclass(bijector_class) or
        not issubclass(bijector_class, tfb.Bijector) or
        bijector_name in SPECIAL_BIJECTORS):
      continue
    # ArgSpec(args, varargs, keywords, defaults)
    spec = inspect.getargspec(bijector_class.__init__)
    ctor_args = set(spec.args) | set(
        [arg for arg in (spec.varargs, spec.keywords) if arg is not None])
    unsupported_args = set(ctor_args) - set(['name', 'self', 'validate_args'])
    if unsupported_args:
      logging.warning('Unable to test tfb.%s: unsupported args %s',
                      bijector_name, unsupported_args)
      continue
    if not bijector_class()._is_injective:  # pylint: disable=protected-access
      logging.warning('Unable to test non-injective tfb.%s.', bijector_name)
      continue
    result[bijector_name] = (bijector_class,)
  result['Invert'] = (tfb.Invert,)

  for bijector_name in sorted(result):
    logging.warning('Supported bijector: tfb.%s', bijector_name)
  INSTANTIABLE_BIJECTORS = result
  return INSTANTIABLE_BIJECTORS


class Support(object):
  """Classification of bijector domains and codomains."""
  SCALAR_UNCONSTRAINED = 'SCALAR_UNCONSTRAINED'
  SCALAR_NON_NEGATIVE = 'SCALAR_NON_NEGATIVE'
  SCALAR_NON_ZERO = 'SCALAR_NON_ZERO'
  SCALAR_POSITIVE = 'SCALAR_POSITIVE'
  SCALAR_GT_NEG1 = 'SCALAR_GT_NEG1'
  SCALAR_IN_NEG1_1 = 'SCALAR_IN_NEG1_1'
  SCALAR_IN_0_1 = 'SCALAR_IN_0_1'
  VECTOR_UNCONSTRAINED = 'VECTOR_UNCONSTRAINED'
  VECTOR_SIZE_TRIANGULAR = 'VECTOR_SIZE_TRIANGULAR'
  VECTOR_WITH_L1_NORM_1_SIZE_GT1 = 'VECTOR_WITH_L1_NORM_1_SIZE_GT1'
  VECTOR_STRICTLY_INCREASING = 'VECTOR_STRICTLY_INCREASING'
  MATRIX_LOWER_TRIL_POSITIVE_DEFINITE = 'MATRIX_LOWER_TRIL_POSITIVE_DEFINITE'
  MATRIX_POSITIVE_DEFINITE = 'MATRIX_POSITIVE_DEFINITE'
  CORRELATION_CHOLESKY = 'CORRELATION_CHOLESKY'
  OTHER = 'OTHER'


class BijectorSupport(collections.namedtuple(
    'BijectorSupport', ['forward', 'inverse'])):
  """Specification of the domain and codomain of a bijector.

  The `forward` slot is the support of the forward computation, i.e., the
  domain, and the `inverse` slot is the support of the inverse computation,
  i.e., the codomain.
  """
  __slots__ = ()


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
  supports = {
      'AffineScalar':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_UNCONSTRAINED),
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
      'MatrixInverseTriL':
          BijectorSupport(Support.MATRIX_LOWER_TRIL_POSITIVE_DEFINITE,
                          Support.MATRIX_LOWER_TRIL_POSITIVE_DEFINITE),
      'MatvecLU':
          BijectorSupport(Support.VECTOR_UNCONSTRAINED,
                          Support.VECTOR_UNCONSTRAINED),
      'NormalCDF':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED, Support.SCALAR_IN_0_1),
      'Exp':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_POSITIVE),
      'Expm1':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED, Support.SCALAR_GT_NEG1),
      'Gumbel':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED, Support.SCALAR_IN_0_1),
      'Identity':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_UNCONSTRAINED),
      'Invert':
          BijectorSupport(Support.OTHER, Support.OTHER),
      'IteratedSigmoidCentered':
          BijectorSupport(Support.VECTOR_UNCONSTRAINED,
                          Support.VECTOR_WITH_L1_NORM_1_SIZE_GT1),
      'Kumaraswamy':
          BijectorSupport(Support.SCALAR_IN_0_1, Support.SCALAR_IN_0_1),
      'Ordered':
          BijectorSupport(Support.VECTOR_STRICTLY_INCREASING,
                          Support.VECTOR_UNCONSTRAINED),
      'RationalQuadraticSpline':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_UNCONSTRAINED),
      'Reciprocal':
          BijectorSupport(Support.SCALAR_NON_ZERO, Support.SCALAR_NON_ZERO),
      'Sigmoid':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED, Support.SCALAR_IN_0_1),
      'SinhArcsinh':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_UNCONSTRAINED),
      'Softsign':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_IN_NEG1_1),
      'SoftmaxCentered':
          BijectorSupport(Support.VECTOR_UNCONSTRAINED,
                          Support.VECTOR_WITH_L1_NORM_1_SIZE_GT1),
      'Square':
          BijectorSupport(Support.SCALAR_NON_NEGATIVE,
                          Support.SCALAR_NON_NEGATIVE),
      'Tanh':
          BijectorSupport(Support.SCALAR_UNCONSTRAINED,
                          Support.SCALAR_IN_NEG1_1),
  }
  missing_keys = set(instantiable_bijectors().keys()) - set(supports.keys())
  if missing_keys:
    raise ValueError('Missing bijector supports: {}'.format(missing_keys))
  BIJECTOR_SUPPORTS = supports
  return BIJECTOR_SUPPORTS


@hps.composite
def unconstrained_bijectors(draw):
  """Strategy for drawing forward-unconstrained bijectors.

  The bijectors emitted by this are those whose `forward` computation
  can act on all of R^n.

  Args:
    draw: Hypothesis MacGuffin.  Supplied by `@hps.composite`.

  Returns:
    unconstrained: A strategy for drawing such bijectors.
  """
  bijector_names = hps.one_of(map(hps.just, instantiable_bijectors().keys()))
  bijector_name = draw(
      bijector_names.filter(lambda b: (  # pylint: disable=g-long-lambda
          b == 'Invert' or 'UNCONSTRAINED' in bijector_supports()[b].forward)))
  if bijector_name == 'Invert':
    underlying = draw(
        bijector_names.filter(
            lambda b: 'UNCONSTRAINED' in bijector_supports()[b].inverse))
    underlying = instantiable_bijectors()[underlying][0](validate_args=True)
    return tfb.Invert(underlying, validate_args=True)
  return instantiable_bijectors()[bijector_name][0](validate_args=True)


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
      return isinstance(dist, tfd.LKJ) and dist.input_output_cholesky
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
