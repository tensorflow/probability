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
"""Utilities for property-based testing for TFP distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import inspect

from absl import logging
import hypothesis as hp
from hypothesis import strategies as hps
import numpy as np
import numpy as onp  # pylint: disable=reimported
import six
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.bijectors import hypothesis_testlib as bijector_hps
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import tensorshape_util


JAX_MODE = False

# pylint is unable to handle @hps.composite (e.g. complains "No value for
# argument 'batch_shape' in function call"), so disable this lint for the file.

# pylint: disable=no-value-for-parameter


# SPECIAL_DISTS are distributions that cannot be drawn by
# base_distributions.
SPECIAL_DISTS = (
    'BatchReshape',
    'Distribution',
    'Empirical',
    'Independent',
    'MixtureSameFamily',
    'TransformedDistribution',
)


# MUTEX_PARAMS are mutually exclusive parameters that cannot be drawn together
# in broadcasting_params.
MUTEX_PARAMS = (
    set(['logits', 'probs']),
    set(['probits', 'probs']),
    set(['rate', 'log_rate']),
    set(['scale', 'scale_tril', 'scale_diag', 'scale_identity_multiplier']),
)


# Whitelist of underlying distributions for QuantizedDistribution (must have
# continuous, infinite support -- QuantizedDistribution also works for finite-
# support distributions for which the length of the support along each dimension
# is at least 1, though it is difficult to construct draws of these
# distributions in general, and wouldn't contribute much to test coverage.)
QUANTIZED_BASE_DISTS = (
    'Chi2',
    'Exponential',
    'LogNormal',
    'Logistic',
    'Normal',
    'Pareto',
    'Poisson',
    'StudentT',
)


# Functions used to constrain randomly sampled parameter ndarrays.
# TODO(b/128518790): Eliminate / minimize the fudge factors in here.


def constrain_between_eps_and_one_minus_eps(eps=1e-6):
  return lambda x: eps + (1 - 2 * eps) * tf.sigmoid(x)


def ensure_high_gt_low(low, high):
  """Returns a value with shape matching `high` and gt broadcastable `low`."""
  new_high = tf.maximum(low + tf.abs(low) * .1 + .1, high)
  reduce_dims = []
  if (tensorshape_util.rank(new_high.shape) >
      tensorshape_util.rank(high.shape)):
    reduced_leading_axes = tf.range(
        tensorshape_util.rank(new_high.shape) -
        tensorshape_util.rank(high.shape))
    new_high = tf.math.reduce_max(
        new_high, axis=reduced_leading_axes)
  reduce_dims = [
      d for d in range(tensorshape_util.rank(high.shape))
      if high.shape[d] < new_high.shape[d]
  ]
  if reduce_dims:
    new_high = tf.math.reduce_max(
        new_high, axis=reduce_dims, keepdims=True)
  return new_high


def fix_finite_discrete(d):
  size = d.get('probs', d.get('logits', None)).shape[-1]
  return dict(d, outcomes=tf.linspace(-1.0, 1.0, size))


def fix_lkj(d):
  return dict(d, concentration=d['concentration'] + 1, dimension=3)


def fix_pert(d):
  peak = ensure_high_gt_low(d['low'], d['peak'])
  high = ensure_high_gt_low(peak, d['high'])
  temperature = ensure_high_gt_low(
      np.zeros(d['temperature'].shape, dtype=np.float32), d['temperature'])
  return dict(d, peak=peak, high=high, temperature=temperature)


def fix_triangular(d):
  peak = ensure_high_gt_low(d['low'], d['peak'])
  high = ensure_high_gt_low(peak, d['high'])
  return dict(d, peak=peak, high=high)


def fix_wishart(d):
  df = d['df']
  scale = d.get('scale', d.get('scale_tril'))
  return dict(d, df=tf.maximum(df, tf.cast(scale.shape[-1], df.dtype)))


CONSTRAINTS = {
    'atol':
        tf.math.softplus,
    'rtol':
        tf.math.softplus,
    'concentration':
        tfp_hps.softplus_plus_eps(),
    'GeneralizedPareto.concentration':  # Permits +ve and -ve concentrations.
        lambda x: tf.math.tanh(x) * 0.24,
    'concentration0':
        tfp_hps.softplus_plus_eps(),
    'concentration1':
        tfp_hps.softplus_plus_eps(),
    'covariance_matrix':
        tfp_hps.positive_definite,
    'df':
        tfp_hps.softplus_plus_eps(),
    'InverseGaussian.loc':
        tfp_hps.softplus_plus_eps(),
    'JohnsonSU.tailweight':
        tfp_hps.softplus_plus_eps(),
    'VonMisesFisher.mean_direction':  # max ndims is 3 to avoid instability.
        lambda x: tf.math.l2_normalize(tf.math.sigmoid(x[..., :3]) + 1e-6, -1),
    'Categorical.probs':
        tf.math.softmax,
    'ExpRelaxedOneHotCategorical.probs':
        tf.math.softmax,
    'FiniteDiscrete.probs':
        tf.math.softmax,
    'Multinomial.probs':
        tf.math.softmax,
    'OneHotCategorical.probs':
        tf.math.softmax,
    'RelaxedCategorical.probs':
        tf.math.softmax,
    'Zipf.power':
        tfp_hps.softplus_plus_eps(1 + 1e-6),  # strictly > 1
    'ContinuousBernoulli.probs':
        tf.sigmoid,
    'Geometric.logits':  # TODO(b/128410109): re-enable down to -50
        # Capping at 15. so that probability is less than 1, and entropy is
        # defined. b/147394924
        lambda x: tf.minimum(tf.maximum(x, -16.), 15.),  # works around the bug
    'Geometric.probs':
        constrain_between_eps_and_one_minus_eps(),
    'Binomial.probs':
        tf.sigmoid,
    'NegativeBinomial.probs':
        tf.sigmoid,
    'Bernoulli.probs':
        tf.sigmoid,
    'PlackettLuce.scores':
        tfp_hps.softplus_plus_eps(),
    'ProbitBernoulli.probs':
        tf.sigmoid,
    'RelaxedBernoulli.probs':
        tf.sigmoid,
    'cutpoints':
        # Permit values that aren't too large
        lambda x: tfb.Ordered().inverse(10 * tf.math.tanh(x)),
    'log_rate':
        lambda x: tf.maximum(x, -16.),
    'mixing_concentration':
        tfp_hps.softplus_plus_eps(),
    'mixing_rate':
        tfp_hps.softplus_plus_eps(),
    'rate':
        tfp_hps.softplus_plus_eps(),
    'scale':
        tfp_hps.softplus_plus_eps(),
    'Wishart.scale':
        tfp_hps.positive_definite,
    'scale_diag':
        tfp_hps.softplus_plus_eps(),
    'scale_identity_multiplier':
        tfp_hps.softplus_plus_eps(),
    'scale_tril':
        tfp_hps.lower_tril_positive_definite,
    'tailweight':
        tfp_hps.softplus_plus_eps(),
    'temperature':
        tfp_hps.softplus_plus_eps(),
    'total_count':
        lambda x: tf.floor(tf.sigmoid(x / 100) * 100) + 1,
    'Bernoulli':
        lambda d: dict(d, dtype=tf.float32),
    'CholeskyLKJ':
        fix_lkj,
    'LKJ':
        fix_lkj,
    'PERT':
        fix_pert,
    'Triangular':
        fix_triangular,
    'TruncatedNormal':
        lambda d: dict(d, high=ensure_high_gt_low(d['low'], d['high'])),
    'Uniform':
        lambda d: dict(d, high=ensure_high_gt_low(d['low'], d['high'])),
    'Wishart':
        fix_wishart,
    'WishartTriL':
        fix_wishart,
    'Zipf':
        lambda d: dict(d, dtype=tf.float32),
    'FiniteDiscrete':
        fix_finite_discrete,
}


def constraint_for(dist=None, param=None):
  if param is not None:
    return CONSTRAINTS.get('{}.{}'.format(dist, param),
                           CONSTRAINTS.get(param, tfp_hps.identity_fn))
  return CONSTRAINTS.get(dist, tfp_hps.identity_fn)


class DistInfo(collections.namedtuple(
    'DistInfo', ['cls', 'params_event_ndims'])):
  """Sufficient information to instantiate a Distribution.

  To wit

  - The Python class `cls` giving the class, and
  - A Python dict `params_event_ndims` giving the event dimensions for the
    parameters (so that parameters can be built with predictable batch shapes).

  Specifically, the `params_event_ndims` dict maps string parameter names to
  Python integers.  Each integer gives how many (trailing) dimensions of that
  parameter are part of the event.
  """
  __slots__ = ()


def _instantiable_base_dists():
  """Computes the table of mechanically instantiable base Distributions.

  A Distribution is mechanically instantiable if

  - The class appears as a symbol binding in `tfp.distributions`;
  - The class defines a `_params_event_ndims` method (necessary
    to generate parameter Tensors with predictable batch shapes); and
  - The name is not blacklisted in `SPECIAL_DISTS`.

  Additionally, the Empricial distribution is hardcoded with special
  instantiation rules for each choice of event_ndims among 0, 1, and 2.

  Compound distributions like TransformedDistribution have their own
  instantiation rules hard-coded in the `distributions` strategy.

  Returns:
    instantiable_base_dists: A Python dict mapping distribution name
      (as a string) to a `DistInfo` carrying the information necessary to
      instantiate it.
  """
  result = {}
  for (dist_name, dist_class) in six.iteritems(tfd.__dict__):
    if (not inspect.isclass(dist_class) or
        not issubclass(dist_class, tfd.Distribution) or
        dist_name in SPECIAL_DISTS):
      continue
    try:
      params_event_ndims = dist_class._params_event_ndims()  # pylint: disable=protected-access
    except NotImplementedError:
      msg = 'Unable to test tfd.%s: _params_event_ndims not implemented'
      logging.warning(msg, dist_name)
      continue
    result[dist_name] = DistInfo(dist_class, params_event_ndims)

  # Empirical._params_event_ndims depends on `self.event_ndims`, so we have to
  # explicitly list these entries.
  result['Empirical|event_ndims=0'] = DistInfo(  #
      functools.partial(tfd.Empirical, event_ndims=0), dict(samples=1))
  result['Empirical|event_ndims=1'] = DistInfo(  #
      functools.partial(tfd.Empirical, event_ndims=1), dict(samples=2))
  result['Empirical|event_ndims=2'] = DistInfo(  #
      functools.partial(tfd.Empirical, event_ndims=2), dict(samples=3))
  return result


# INSTANTIABLE_BASE_DISTS is a map from str->(DistClass, params_event_ndims)
INSTANTIABLE_BASE_DISTS = _instantiable_base_dists()
del _instantiable_base_dists


INSTANTIABLE_META_DISTS = (
    'BatchReshape',
    'Independent',
    'MixtureSameFamily',
    'TransformedDistribution',
    'QuantizedDistribution',
)


@hps.composite
def valid_slices(draw, batch_shape):
  """Samples a legal (possibly empty) slice for shape batch_shape."""
  # We build up a list of slices in several stages:
  # 1. Choose 0 to batch_rank slices to come before an Ellipsis (...).
  # 2. Decide whether or not to add an Ellipsis; if using, updating the indexing
  #    used (e.g. batch_shape[i]) to identify safe bounds.
  # 3. Choose 0 to [remaining_dims] slices to come last.
  # 4. Decide where to insert between 0 and 3 newaxis slices.
  batch_shape = tf.TensorShape(batch_shape).as_list()
  slices = []
  batch_rank = len(batch_shape)
  arbitrary_slices = hps.tuples(
      hps.one_of(hps.just(None), hps.integers(min_value=-100, max_value=100)),
      hps.one_of(hps.just(None), hps.integers(min_value=-100, max_value=100)),
      hps.one_of(
          hps.just(None),
          hps.integers(min_value=-100, max_value=100).filter(lambda x: x != 0))
  ).map(lambda tup: slice(*tup))

  # 1. Choose 0 to batch_rank slices to come before an Ellipsis (...).
  nslc_before_ellipsis = draw(hps.integers(min_value=0, max_value=batch_rank))
  for i in range(nslc_before_ellipsis):
    slc = draw(
        hps.one_of(
            hps.integers(min_value=0, max_value=batch_shape[i] - 1),
            arbitrary_slices))
    slices.append(slc)
  # 2. Decide whether or not to add an Ellipsis; if using, updating the indexing
  #    used (e.g. batch_shape[i]) to identify safe bounds.
  has_ellipsis = draw(hps.booleans().map(lambda x: (Ellipsis, x)))[1]
  nslc_after_ellipsis = draw(
      hps.integers(min_value=0, max_value=batch_rank - nslc_before_ellipsis))
  if has_ellipsis:
    slices.append(Ellipsis)
    remain_start, remain_end = (batch_rank - nslc_after_ellipsis, batch_rank)
  else:
    remain_start = nslc_before_ellipsis
    remain_end = nslc_before_ellipsis + nslc_after_ellipsis
  # 3. Choose 0 to [remaining_dims] slices to come last.
  for i in range(remain_start, remain_end):
    slc = draw(
        hps.one_of(
            hps.integers(min_value=0, max_value=batch_shape[i] - 1),
            arbitrary_slices))
    slices.append(slc)
  # 4. Decide where to insert between 0 and 3 newaxis slices.
  newaxis_positions = draw(
      hps.lists(hps.integers(min_value=0, max_value=len(slices)), max_size=3))
  for i in sorted(newaxis_positions, reverse=True):
    slices.insert(i, tf.newaxis)
  slices = tuple(slices)
  # Since `d[0]` ==> `d.__getitem__(0)` instead of `d.__getitem__((0,))`;
  # and similarly `d[:3]` ==> `d.__getitem__(slice(None, 3))` instead of
  # `d.__getitem__((slice(None, 3),))`; it is useful to test such scenarios.
  if len(slices) == 1 and draw(hps.booleans()):
    # Sometimes only a single item non-tuple.
    return slices[0]
  return slices


def stringify_slices(slices):
  """Returns a list of strings describing the items in `slices`.

  Each returned string (in order) encodes what to do with one dimension of the
  slicee:

  - That number for a single integer slice;
  - 'a:b:c' for a start-stop-step slice, omitting any missing components;
  - 'tf.newaxis' for an axis insertion; or
  - The ellipsis '...' for an arbitrary-rank gap.

  Args:
    slices: A single-dimension slice or a Python tuple of single-dimension
      slices.

  Returns:
    pretty_slices: A list of Python strings encoding each slice.
  """
  pretty_slices = []
  slices = slices if isinstance(slices, tuple) else (slices,)
  for slc in slices:
    if slc == Ellipsis:
      pretty_slices.append('...')
    elif isinstance(slc, slice):
      pretty_slices.append('{}:{}:{}'.format(
          *['' if s is None else s for s in (slc.start, slc.stop, slc.step)]))
    elif isinstance(slc, int) or tf.is_tensor(slc):
      pretty_slices.append(str(slc))
    elif slc is tf.newaxis:
      pretty_slices.append('tf.newaxis')
    else:
      raise ValueError('Unexpected slice type: {}'.format(type(slc)))
  return pretty_slices


@hps.composite
def broadcasting_params(draw,
                        dist_name,
                        batch_shape,
                        event_dim=None,
                        enable_vars=False):
  """Strategy for drawing parameters broadcasting to `batch_shape`."""
  if dist_name not in INSTANTIABLE_BASE_DISTS:
    raise ValueError('Unknown Distribution name {}'.format(dist_name))

  params_event_ndims = INSTANTIABLE_BASE_DISTS[dist_name].params_event_ndims

  def _constraint(param):
    return constraint_for(dist_name, param)

  return draw(
      tfp_hps.broadcasting_params(
          batch_shape,
          params_event_ndims,
          event_dim=event_dim,
          enable_vars=enable_vars,
          constraint_fn_for=_constraint,
          mutex_params=MUTEX_PARAMS))


def assert_shapes_unchanged(target_shaped_dict, possibly_bcast_dict):
  for param, target_param_val in six.iteritems(target_shaped_dict):
    onp.testing.assert_array_equal(
        tensorshape_util.as_list(target_param_val.shape),
        tensorshape_util.as_list(possibly_bcast_dict[param].shape))


@hps.composite
def base_distributions(draw,
                       dist_name=None,
                       batch_shape=None,
                       event_dim=None,
                       enable_vars=False,
                       eligibility_filter=lambda name: True,
                       validate_args=True):
  """Strategy for drawing arbitrary base Distributions.

  This does not draw compound distributions like `Independent`,
  `MixtureSameFamily`, or `TransformedDistribution`; only base Distributions
  that do not accept other Distributions as arguments.

  Args:
    draw: Hypothesis strategy sampler supplied by `@hps.composite`.
    dist_name: Optional Python `str`.  If given, the produced distributions
      will all have this type.
    batch_shape: An optional `TensorShape`.  The batch shape of the resulting
      Distribution.  Hypothesis will pick a batch shape if omitted.
    event_dim: Optional Python int giving the size of each of the
      distribution's parameters' event dimensions.  This is shared across all
      parameters, permitting square event matrices, compatible location and
      scale Tensors, etc. If omitted, Hypothesis will choose one.
    enable_vars: TODO(bjp): Make this `True` all the time and put variable
      initialization in slicing_test.  If `False`, the returned parameters are
      all `tf.Tensor`s and not {`tf.Variable`, `tfp.util.DeferredTensor`
      `tfp.util.TransformedVariable`}.
    eligibility_filter: Optional Python callable.  Blacklists some Distribution
      class names so they will not be drawn at the top level.
    validate_args: Python `bool`; whether to enable runtime assertions.

  Returns:
    dists: A strategy for drawing Distributions with the specified `batch_shape`
      (or an arbitrary one if omitted).
  """
  if dist_name is None:
    names = [k for k in INSTANTIABLE_BASE_DISTS if eligibility_filter(k)]
    dist_name = draw(hps.sampled_from(sorted(names)))

  if dist_name == 'Empirical':
    variants = [k for k in INSTANTIABLE_BASE_DISTS
                if eligibility_filter(k) and 'Empirical' in k]
    dist_name = draw(hps.sampled_from(sorted(variants)))

  if batch_shape is None:
    batch_shape = draw(tfp_hps.shapes())

  # Draw raw parameters
  params_kwargs = draw(
      broadcasting_params(
          dist_name, batch_shape, event_dim=event_dim, enable_vars=enable_vars))
  hp.note('Forming dist {} with raw parameters {}'.format(
      dist_name, params_kwargs))

  # Constrain them to legal values
  params_constrained = constraint_for(dist_name)(params_kwargs)

  # Sometimes the "distribution constraint" fn may replace c2t-tracking
  # DeferredTensor params with Tensor params (e.g. fix_triangular). In such
  # cases, we preserve the c2t-tracking DeferredTensors by wrapping them but
  # ignoring the value.  We similarly reinstate raw tf.Variables, so they
  # appear in the distribution's `variables` list and can be initialized.
  for k in params_constrained:
    # In JAX_MODE, tfp_util.DeferredTensor is a function, not a class, so we
    # disable this check entirely.
    if (not JAX_MODE and k in params_kwargs and
        isinstance(params_kwargs[k], (tfp_util.DeferredTensor, tf.Variable)) and
        params_kwargs[k] is not params_constrained[k]):

      def constrained_value(v, val=params_constrained[k]):  # pylint: disable=cell-var-from-loop
        # While the gradient to v will be 0, we only care about the c2t counts.
        return v * 0 + val

      params_constrained[k] = tfp_util.DeferredTensor(
          params_kwargs[k], constrained_value)

  hp.note('Forming dist {} with constrained parameters {}'.format(
      dist_name, params_constrained))
  assert_shapes_unchanged(params_kwargs, params_constrained)
  params_constrained['validate_args'] = validate_args

  if dist_name in ['Wishart', 'WishartTriL']:
    # With the default `input_output_cholesky = False`, Wishart occasionally
    # produces samples for which the Cholesky decompositions fail, causing
    # an error in testDistribution when `log_prob` is called on a sample.
    params_constrained['input_output_cholesky'] = True

  # Actually construct the distribution
  dist_cls = INSTANTIABLE_BASE_DISTS[dist_name].cls
  result_dist = dist_cls(**params_constrained)

  # Check that the batch shape came out as expected
  if batch_shape != result_dist.batch_shape:
    msg = ('Distributions strategy generated a bad batch shape '
           'for {}, should have been {}.').format(result_dist, batch_shape)
    raise AssertionError(msg)
  return result_dist


def depths():
  return hps.integers(min_value=0, max_value=4)


def params_used(dist):
  return [k for k, v in six.iteritems(dist.parameters) if v is not None]


@hps.composite
def batch_reshapes(
    draw, batch_shape=None, event_dim=None,
    enable_vars=False, depth=None,
    eligibility_filter=lambda name: True, validate_args=True):
  """Strategy for drawing `BatchReshape` distributions.

  The underlying distribution is drawn from the `distributions` strategy.

  Args:
    draw: Hypothesis strategy sampler supplied by `@hps.composite`.
    batch_shape: An optional `TensorShape`.  The batch shape of the resulting
      `BatchReshape` distribution.  Note that the underlying distribution will
      in general have a different batch shape, to make the reshaping
      non-trivial.  Hypothesis will pick one if omitted.
    event_dim: Optional Python int giving the size of each of the underlying
      distribution's parameters' event dimensions.  This is shared across all
      parameters, permitting square event matrices, compatible location and
      scale Tensors, etc. If omitted, Hypothesis will choose one.
    enable_vars: TODO(bjp): Make this `True` all the time and put variable
      initialization in slicing_test.  If `False`, the returned parameters are
      all `tf.Tensor`s and not {`tf.Variable`, `tfp.util.DeferredTensor`
      `tfp.util.TransformedVariable`}
    depth: Python `int` giving maximum nesting depth of compound Distributions.
    eligibility_filter: Optional Python callable.  Blacklists some Distribution
      class names so they will not be drawn.
    validate_args: Python `bool`; whether to enable runtime assertions.

  Returns:
    dists: A strategy for drawing `BatchReshape` distributions with the
      specified `batch_shape` (or an arbitrary one if omitted).
  """
  if depth is None:
    depth = draw(depths())

  if batch_shape is None:
    batch_shape = draw(tfp_hps.shapes(min_ndims=1, max_side=4))

  # TODO(b/142135119): Wanted to draw general input and output shapes like the
  # following, but Hypothesis complained about filtering out too many things.
  # underlying_batch_shape = draw(tfp_hps.shapes(min_ndims=1))
  # hp.assume(
  #   batch_shape.num_elements() == underlying_batch_shape.num_elements())
  underlying_batch_shape = [tensorshape_util.num_elements(batch_shape)]

  underlying = draw(
      distributions(
          batch_shape=underlying_batch_shape,
          event_dim=event_dim,
          enable_vars=enable_vars,
          depth=depth - 1,
          eligibility_filter=eligibility_filter,
          validate_args=validate_args))
  hp.note('Forming BatchReshape with underlying dist {}; '
          'parameters {}; batch_shape {}'.format(
              underlying, params_used(underlying), batch_shape))
  result_dist = tfd.BatchReshape(
      underlying, batch_shape=batch_shape, validate_args=True)
  return result_dist


@hps.composite
def independents(
    draw, batch_shape=None, event_dim=None,
    enable_vars=False, depth=None, eligibility_filter=lambda name: True,
    validate_args=True):
  """Strategy for drawing `Independent` distributions.

  The underlying distribution is drawn from the `distributions` strategy.

  Args:
    draw: Hypothesis strategy sampler supplied by `@hps.composite`.
    batch_shape: An optional `TensorShape`.  The batch shape of the resulting
      `Independent` distribution.  Note that the underlying distribution will in
      general have a higher-rank batch shape, to make room for reinterpreting
      some of those dimensions as the `Independent`'s event.  Hypothesis will
      pick one if omitted.
    event_dim: Optional Python int giving the size of each of the underlying
      distribution's parameters' event dimensions.  This is shared across all
      parameters, permitting square event matrices, compatible location and
      scale Tensors, etc. If omitted, Hypothesis will choose one.
    enable_vars: TODO(bjp): Make this `True` all the time and put variable
      initialization in slicing_test.  If `False`, the returned parameters are
      all `tf.Tensor`s and not {`tf.Variable`, `tfp.util.DeferredTensor`
      `tfp.util.TransformedVariable`}
    depth: Python `int` giving maximum nesting depth of compound Distributions.
    eligibility_filter: Optional Python callable.  Blacklists some Distribution
      class names so they will not be drawn.
    validate_args: Python `bool`; whether to enable runtime assertions.

  Returns:
    dists: A strategy for drawing `Independent` distributions with the specified
      `batch_shape` (or an arbitrary one if omitted).
  """
  if depth is None:
    depth = draw(depths())

  reinterpreted_batch_ndims = draw(hps.integers(min_value=0, max_value=2))

  if batch_shape is None:
    batch_shape = draw(
        tfp_hps.shapes(min_ndims=reinterpreted_batch_ndims))
  else:  # This independent adds some batch dims to its underlying distribution.
    batch_shape = tensorshape_util.concatenate(
        batch_shape,
        draw(tfp_hps.shapes(
            min_ndims=reinterpreted_batch_ndims,
            max_ndims=reinterpreted_batch_ndims)))

  underlying = draw(
      distributions(
          batch_shape=batch_shape,
          event_dim=event_dim,
          enable_vars=enable_vars,
          depth=depth - 1,
          eligibility_filter=eligibility_filter,
          validate_args=validate_args))
  hp.note('Forming Independent with underlying dist {}; '
          'parameters {}; reinterpreted_batch_ndims {}'.format(
              underlying, params_used(underlying), reinterpreted_batch_ndims))
  result_dist = tfd.Independent(
      underlying,
      reinterpreted_batch_ndims=reinterpreted_batch_ndims,
      validate_args=validate_args)
  expected_shape = batch_shape[:len(batch_shape) - reinterpreted_batch_ndims]
  if expected_shape != result_dist.batch_shape:
    msg = ('Independent strategy generated a bad batch shape '
           'for {}, should have been {}.').format(result_dist, expected_shape)
    raise AssertionError(msg)
  return result_dist


@hps.composite
def transformed_distributions(draw,
                              batch_shape=None,
                              event_dim=None,
                              enable_vars=False,
                              depth=None,
                              eligibility_filter=lambda name: True,
                              validate_args=True):
  """Strategy for drawing `TransformedDistribution`s.

  The transforming bijector is drawn from the
  `bijectors.hypothesis_testlib.unconstrained_bijectors` strategy.

  The underlying distribution is drawn from the `distributions` strategy, except
  that it must be compatible with the bijector according to
  `bijectors.hypothesis_testlib.distribution_filter_for` (these generally check
  that vector bijectors are not combined with scalar distributions, etc).

  Args:
    draw: Hypothesis strategy sampler supplied by `@hps.composite`.
    batch_shape: An optional `TensorShape`.  The batch shape of the resulting
      `TransformedDistribution`.  The underlying distribution will sometimes
      have the same `batch_shape`, and sometimes have scalar batch shape.
      Hypothesis will pick a `batch_shape` if omitted.
    event_dim: Optional Python int giving the size of each of the underlying
      distribution's parameters' event dimensions.  This is shared across all
      parameters, permitting square event matrices, compatible location and
      scale Tensors, etc. If omitted, Hypothesis will choose one.
    enable_vars: TODO(bjp): Make this `True` all the time and put variable
      initialization in slicing_test.  If `False`, the returned parameters are
      all `tf.Tensor`s and not {`tf.Variable`, `tfp.util.DeferredTensor`
      `tfp.util.TransformedVariable`}
    depth: Python `int` giving maximum nesting depth of compound Distributions.
    eligibility_filter: Optional Python callable.  Blacklists some Distribution
      class names so they will not be drawn.
    validate_args: Python `bool`; whether to enable runtime assertions.

  Returns:
    dists: A strategy for drawing `TransformedDistribution`s with the specified
      `batch_shape` (or an arbitrary one if omitted).
  """
  if depth is None:
    depth = draw(depths())

  bijector = draw(bijector_hps.unconstrained_bijectors())
  hp.note('Drawing TransformedDistribution with bijector {}'.format(bijector))
  if batch_shape is None:
    batch_shape = draw(tfp_hps.shapes())
  underlying_batch_shape = batch_shape
  # TODO(b/151180729): Remove `batch_shape_arg` when `TransformedDistribution`
  # shape overrides are deprecated.
  batch_shape_arg = None
  if draw(hps.booleans()):
    # Use batch_shape overrides.
    underlying_batch_shape = tf.TensorShape([])  # scalar underlying batch
    batch_shape_arg = batch_shape
  underlyings = distributions(
      batch_shape=underlying_batch_shape,
      event_dim=event_dim,
      enable_vars=enable_vars,
      depth=depth - 1,
      eligibility_filter=eligibility_filter,
      validate_args=validate_args).filter(
          bijector_hps.distribution_filter_for(bijector))
  to_transform = draw(underlyings)
  hp.note('Forming TransformedDistribution with '
          'underlying distribution {}; parameters {}'.format(
              to_transform, params_used(to_transform)))
  result_dist = tfd.TransformedDistribution(
      bijector=bijector,
      distribution=to_transform,
      batch_shape=batch_shape_arg,
      validate_args=validate_args)
  if batch_shape != result_dist.batch_shape:
    msg = ('TransformedDistribution strategy generated a bad batch shape '
           'for {}, should have been {}.').format(result_dist, batch_shape)
    raise AssertionError(msg)
  return result_dist


@hps.composite
def quantized_distributions(draw,
                            batch_shape=None,
                            event_dim=None,
                            enable_vars=False,
                            eligibility_filter=lambda name: True,
                            validate_args=True):
  """Strategy for drawing `QuantizedDistribution`s.

  The underlying distribution is drawn from the `base_distributions` strategy.

  Args:
    draw: Hypothesis strategy sampler supplied by `@hps.composite`.
    batch_shape: An optional `TensorShape`.  The batch shape of the resulting
      `QuantizedDistribution`. Hypothesis will pick a `batch_shape` if omitted.
    event_dim: Optional Python int giving the size of each of the underlying
      distribution's parameters' event dimensions.  This is shared across all
      parameters, permitting square event matrices, compatible location and
      scale Tensors, etc. If omitted, Hypothesis will choose one.
    enable_vars: TODO(bjp): Make this `True` all the time and put variable
      initialization in slicing_test.  If `False`, the returned parameters are
      all Tensors, never Variables or DeferredTensor.
    eligibility_filter: Optional Python callable.  Blacklists some Distribution
      class names so they will not be drawn.
    validate_args: Python `bool`; whether to enable runtime assertions.

  Returns:
    dists: A strategy for drawing `QuantizedDistribution`s with the specified
      `batch_shape` (or an arbitrary one if omitted).
  """

  if batch_shape is None:
    batch_shape = draw(tfp_hps.shapes())

  low_quantile = draw(
      hps.one_of(
          hps.just(None),
          hps.floats(min_value=0.01, max_value=0.7)))
  high_quantile = draw(
      hps.one_of(
          hps.just(None),
          hps.floats(min_value=0.3, max_value=.99)))

  def ok(name):
    return eligibility_filter(name) and name in QUANTIZED_BASE_DISTS
  underlyings = base_distributions(
      batch_shape=batch_shape,
      event_dim=event_dim,
      enable_vars=enable_vars,
      eligibility_filter=ok,
  )
  underlying = draw(underlyings)

  if high_quantile is not None:
    high_quantile = tf.convert_to_tensor(high_quantile, dtype=underlying.dtype)
  if low_quantile is not None:
    low_quantile = tf.convert_to_tensor(low_quantile, dtype=underlying.dtype)
    if high_quantile is not None:
      high_quantile = ensure_high_gt_low(low_quantile, high_quantile)

  hp.note('Drawing QuantizedDistribution with underlying distribution'
          ' {}'.format(underlying))

  try:
    low = None if low_quantile is None else underlying.quantile(low_quantile)
    high = None if high_quantile is None else underlying.quantile(high_quantile)
  except NotImplementedError:
    # The following code makes ReproducibilityTest flaky in graph mode (but not
    # eager). Failures are due either to partial mismatch in the samples in
    # ReproducibilityTest or to `low` and/or `high` being NaN. For now, to avoid
    # this, we set `low` and `high` to `None` for distributions not implementing
    # `quantile`.

    # seed = test_util.test_seed(hardcoded_seed=123)
    # low = (None if low_quantile is None
    #        else underlying.sample(low_quantile.shape, seed=seed))
    # high = (None if high_quantile is None else
    #         underlying.sample(high_quantile.shape, seed=seed))
    low = None
    high = None

  # Ensure that `low` and `high` are ints contained in distribution support
  # and span at least a few bins.
  if high is not None:
    high = tf.clip_by_value(high, -2**23, 2**23)
    high = tf.math.ceil(high + 5.)

  if low is not None:
    low = tf.clip_by_value(low, -2**23, 2**23)
    low = tf.math.ceil(low)

  result_dist = tfd.QuantizedDistribution(
      distribution=underlying,
      low=low,
      high=high,
      validate_args=validate_args)

  return result_dist


@hps.composite
def mixtures_same_family(draw,
                         batch_shape=None,
                         event_dim=None,
                         enable_vars=False,
                         depth=None,
                         eligibility_filter=lambda name: True,
                         validate_args=True):
  """Strategy for drawing `MixtureSameFamily` distributions.

  The component distribution is drawn from the `distributions` strategy.

  The Categorical mixture distributions are either shared across all batch
  members, or drawn independently for the full batch (as required by
  `MixtureSameFamily`).

  Args:
    draw: Hypothesis strategy sampler supplied by `@hps.composite`.
    batch_shape: An optional `TensorShape`.  The batch shape of the resulting
      `MixtureSameFamily` distribution.  The component distribution will have a
      batch shape of 1 rank higher (for the components being mixed).  Hypothesis
      will pick a batch shape if omitted.
    event_dim: Optional Python int giving the size of each of the component
      distribution's parameters' event dimensions.  This is shared across all
      parameters, permitting square event matrices, compatible location and
      scale Tensors, etc. If omitted, Hypothesis will choose one.
    enable_vars: TODO(bjp): Make this `True` all the time and put variable
      initialization in slicing_test.  If `False`, the returned parameters are
      all `tf.Tensor`s and not {`tf.Variable`, `tfp.util.DeferredTensor`
      `tfp.util.TransformedVariable`}
    depth: Python `int` giving maximum nesting depth of compound Distributions.
    eligibility_filter: Optional Python callable.  Blacklists some Distribution
      class names so they will not be drawn.
    validate_args: Python `bool`; whether to enable runtime assertions.

  Returns:
    dists: A strategy for drawing `MixtureSameFamily` distributions with the
      specified `batch_shape` (or an arbitrary one if omitted).
  """
  if depth is None:
    depth = draw(depths())

  if batch_shape is None:
    # Ensure the components dist has at least one batch dim (a component dim).
    batch_shape = draw(tfp_hps.shapes(min_ndims=1, min_lastdimsize=2))
  else:  # This mixture adds a batch dim to its underlying components dist.
    batch_shape = tensorshape_util.concatenate(
        batch_shape,
        draw(tfp_hps.shapes(min_ndims=1, max_ndims=1, min_lastdimsize=2)))

  component = draw(
      distributions(
          batch_shape=batch_shape,
          event_dim=event_dim,
          enable_vars=enable_vars,
          eligibility_filter=eligibility_filter,
          depth=depth - 1))
  hp.note('Drawing MixtureSameFamily with component {}; parameters {}'.format(
      component, params_used(component)))
  # scalar or same-shaped categorical?
  mixture_batch_shape = draw(
      hps.one_of(hps.just(batch_shape[:-1]), hps.just(tf.TensorShape([]))))
  mixture_dist = draw(base_distributions(
      dist_name='Categorical',
      batch_shape=mixture_batch_shape,
      event_dim=tensorshape_util.as_list(batch_shape)[-1],
      enable_vars=enable_vars,
      validate_args=validate_args))
  hp.note(('Forming MixtureSameFamily with '
           'mixture distribution {}; parameters {}').format(
               mixture_dist, params_used(mixture_dist)))
  result_dist = tfd.MixtureSameFamily(
      components_distribution=component,
      mixture_distribution=mixture_dist,
      validate_args=validate_args)
  if batch_shape[:-1] != result_dist.batch_shape:
    msg = ('MixtureSameFamily strategy generated a bad batch shape '
           'for {}, should have been {}.').format(result_dist, batch_shape[:-1])
    raise AssertionError(msg)
  return result_dist


@hps.composite
def distributions(draw,
                  dist_name=None,
                  batch_shape=None,
                  event_dim=None,
                  enable_vars=False,
                  depth=None,
                  eligibility_filter=lambda name: True,
                  validate_args=True):
  """Strategy for drawing arbitrary Distributions.

  This may draw compound distributions (i.e., `Independent`,
  `MixtureSameFamily`, and/or `TransformedDistribution`), in which case the
  underlying distributions are drawn recursively from this strategy as well.

  Args:
    draw: Hypothesis strategy sampler supplied by `@hps.composite`.
    dist_name: Optional Python `str`.  If given, the produced distributions
      will all have this type.
    batch_shape: An optional `TensorShape`.  The batch shape of the resulting
      Distribution.  Hypothesis will pick a batch shape if omitted.
    event_dim: Optional Python int giving the size of each of the
      distribution's parameters' event dimensions.  This is shared across all
      parameters, permitting square event matrices, compatible location and
      scale Tensors, etc. If omitted, Hypothesis will choose one.
    enable_vars: TODO(bjp): Make this `True` all the time and put variable
      initialization in slicing_test.  If `False`, the returned parameters are
      all `tf.Tensor`s and not {`tf.Variable`, `tfp.util.DeferredTensor`
      `tfp.util.TransformedVariable`}.
    depth: Python `int` giving maximum nesting depth of compound Distributions.
      If `None`, Hypothesis will bias choose one, with a bias towards shallow
      nests.
    eligibility_filter: Optional Python callable.  Blacklists some Distribution
      class names so they will not be drawn.
    validate_args: Python `bool`; whether to enable runtime assertions.

  Returns:
    dists: A strategy for drawing Distributions with the specified `batch_shape`
      (or an arbitrary one if omitted).

  Raises:
    ValueError: If it doesn't know how to instantiate a Distribution of class
      `dist_name`.
  """
  if depth is None:
    depth = draw(depths())

  if dist_name is None and depth > 0:
    bases = hps.just(None)
    candidates = ['BatchReshape', 'Independent',
                  'MixtureSameFamily', 'TransformedDistribution']
    names = [name for name in candidates if eligibility_filter(name)]
    compounds = hps.one_of(map(hps.just, names))
    dist_name = draw(hps.one_of([bases, compounds]))

  if (dist_name is None
      or dist_name in INSTANTIABLE_BASE_DISTS
      or dist_name == 'Empirical'):
    return draw(base_distributions(
        dist_name, batch_shape, event_dim, enable_vars,
        eligibility_filter, validate_args))
  if dist_name == 'BatchReshape':
    return draw(batch_reshapes(
        batch_shape, event_dim, enable_vars, depth,
        eligibility_filter, validate_args))
  if dist_name == 'Independent':
    return draw(independents(
        batch_shape, event_dim, enable_vars, depth,
        eligibility_filter, validate_args))
  if dist_name == 'MixtureSameFamily':
    return draw(mixtures_same_family(
        batch_shape, event_dim, enable_vars, depth,
        eligibility_filter, validate_args))
  if dist_name == 'TransformedDistribution':
    return draw(transformed_distributions(
        batch_shape, event_dim, enable_vars, depth,
        eligibility_filter, validate_args))
  if dist_name == 'QuantizedDistribution':
    return draw(quantized_distributions(
        batch_shape, event_dim, enable_vars,
        eligibility_filter, validate_args))
  raise ValueError('Unknown Distribution name {}'.format(dist_name))
