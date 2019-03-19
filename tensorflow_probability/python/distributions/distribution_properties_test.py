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
"""Property-based testing for TFP distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect
import traceback

from absl import logging
import hypothesis as hp
from hypothesis import strategies as hps
from hypothesis.extra import numpy as hpnp
import numpy as np
import six
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

tfd = tfp.distributions

MUTEX_PARAMS = [
    set(['logits', 'probs']),
    set(['rate', 'log_rate']),
    set(['scale', 'scale_tril']),
]

SPECIAL_DISTS = [
    'ConditionalDistribution', 'ConditionalTransformedDistribution',
    'Distribution', 'Empirical', 'Independent'
]


def instantiable_dists():
  result = {}
  for (dist_name, dist_class) in six.iteritems(tfd.__dict__):
    if (not inspect.isclass(dist_class) or
        not issubclass(dist_class, tfd.Distribution) or
        dist_name in SPECIAL_DISTS):
      continue
    try:
      params_event_ndims = dist_class._params_event_ndims()
    except NotImplementedError:
      logging.warning('Unable to test tfd.%s: %s', dist_name,
                      traceback.format_exc())

      continue
    result[dist_name] = (dist_class, params_event_ndims)

  del result['InverseGamma'][1]['rate']  # deprecated parameter

  # Empirical._params_event_ndims depends on `self.event_ndims`, so we have to
  # explicitly list these entries.
  result['Empirical|event_ndims=0'] = (  #
      functools.partial(tfd.Empirical, event_ndims=0), dict(samples=1))
  result['Empirical|event_ndims=1'] = (  #
      functools.partial(tfd.Empirical, event_ndims=1), dict(samples=2))
  result['Empirical|event_ndims=2'] = (  #
      functools.partial(tfd.Empirical, event_ndims=2), dict(samples=3))

  result['Independent'] = (tfd.Independent, None)
  return result


# INSTANTIABLE_DISTS is a map from str->(DistClass, params_event_ndims)
INSTANTIABLE_DISTS = instantiable_dists()
del instantiable_dists


def rank_only_shape(mindims, maxdims):
  return hps.integers(
      min_value=mindims, max_value=maxdims).map(tf.TensorShape(None).with_rank)


def compute_rank_and_fullsize_reqd(data, batch_shape, current_batch_shape,
                                   is_last_param):
  """Returns a param rank and a list of bools for full-size-required by axis.

  Args:
    data: Hypothesis data sampler.
    batch_shape: Target broadcasted batch shape.
    current_batch_shape: Broadcasted batch shape of params selected thus far.
      This is ignored for non-last parameters.
    is_last_param: bool indicator of whether this is the last param (in which
      case, we must achieve the target batch_shape).

  Returns:
    param_batch_rank: Sampled rank for this parameter.
    force_fullsize_dim: `param_batch_rank`-sized list of bool indicating whether
      the corresponding axis of the parameter must be full-sized (True) or is
      allowed to be 1 (i.e., broadcast) (False).
  """
  batch_rank = batch_shape.ndims
  if is_last_param:
    # We must force full size dim on any mismatched axes, and proper rank.
    full_rank_current = tf.broadcast_static_shape(
        current_batch_shape, tf.TensorShape([1] * batch_rank))
    # Identify axes in which the target shape is not yet matched.
    axis_is_mismatched = [
        full_rank_current[i] != batch_shape[i] for i in range(batch_rank)
    ]
    min_rank = batch_rank
    if current_batch_shape.ndims == batch_rank:
      # Current rank might be already correct, but we could have a case like
      # batch_shape=[4,3,2] and current_batch_shape=[4,1,2], in which case
      # we must have at least 2 axes on this param's batch shape.
      min_rank -= (axis_is_mismatched + [True]).index(True)
    param_batch_rank = data.draw(rank_only_shape(min_rank, batch_rank)).ndims
    # Get the last param_batch_rank (possibly 0!) items.
    force_fullsize_dim = axis_is_mismatched[batch_rank - param_batch_rank:]
  else:
    # There are remaining params to be drawn, so we will be able to force full
    # size axes on subsequent params.
    param_batch_rank = data.draw(rank_only_shape(0, batch_rank)).ndims
    force_fullsize_dim = [False] * param_batch_rank
  return param_batch_rank, force_fullsize_dim


def draw_broadcasting_shapes(data, batch_shape, param_names):
  """Draws a set of parameter batch shapes that broadcast to `batch_shape`.

  For each parameter we need to choose its batch rank, and whether or not each
  axis i is 1 or batch_shape[i]. This function chooses a set of shapes that
  have possibly mismatched ranks, and possibly broadcasting axes, with the
  promise that the broadcast of the set of all shapes matches `batch_shape`.

  Args:
    data: Hypothesis sampler.
    batch_shape: `tf.TensorShape`, the target (fully-defined) batch shape.
    param_names: Iterable of `str`, the parameters whose batch shapes need
      determination.

  Returns:
    param_batch_shapes: `dict` of `str->tf.TensorShape` where the set of
      shapes broadcast to `batch_shape`. The shapes are fully defined.
  """
  batch_rank = batch_shape.ndims
  result = {}
  remaining_params = set(param_names)
  current_batch_shape = tf.TensorShape([])
  while remaining_params:
    next_param = data.draw(hps.one_of(map(hps.just, remaining_params)))
    remaining_params.remove(next_param)
    param_batch_rank, force_fullsize_dim = compute_rank_and_fullsize_reqd(
        data,
        batch_shape,
        current_batch_shape,
        is_last_param=not remaining_params)

    # Get the last param_batch_rank (possibly 0!) dimensions.
    param_batch_shape = batch_shape[batch_rank - param_batch_rank:].as_list()
    for i, force_fullsize in enumerate(force_fullsize_dim):
      if not force_fullsize and data.draw(hps.booleans()):
        # Choose to make this param broadcast against some other param.
        param_batch_shape[i] = 1
    param_batch_shape = tf.TensorShape(param_batch_shape)
    current_batch_shape = tf.broadcast_static_shape(current_batch_shape,
                                                    param_batch_shape)
    result[next_param] = param_batch_shape
  return result


def draw_valid_slices(data, batch_shape):
  """Samples a legal (possibly empty) slice for shape batch_shape."""
  # We build up a list of slices in several stages:
  # 1. Choose 0 to batch_rank slices to come before an Ellipsis (...).
  # 2. Decide whether or not to add an Ellipsis; if using, updating the indexing
  #    used (e.g. batch_shape[i]) to identify safe bounds.
  # 3. Choose 0 to [remaining_dims] slices to come last.
  # 4. Decide where to insert between 0 and 7 newaxis slices.
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
  nslc_before_ellipsis = data.draw(
      hps.integers(min_value=0, max_value=batch_rank))
  for i in range(nslc_before_ellipsis):
    slc = data.draw(
        hps.one_of(
            hps.integers(min_value=0, max_value=batch_shape[i] - 1),
            arbitrary_slices))
    slices.append(slc)
  # 2. Decide whether or not to add an Ellipsis; if using, updating the indexing
  #    used (e.g. batch_shape[i]) to identify safe bounds.
  has_ellipsis = data.draw(hps.booleans().map(lambda x: (Ellipsis, x)))[1]
  nslc_after_ellipsis = data.draw(
      hps.integers(min_value=0, max_value=batch_rank - nslc_before_ellipsis))
  if has_ellipsis:
    slices.append(Ellipsis)
    remain_start, remain_end = (batch_rank - nslc_after_ellipsis, batch_rank)
  else:
    remain_start = nslc_before_ellipsis
    remain_end = nslc_before_ellipsis + nslc_after_ellipsis
  # 3. Choose 0 to [remaining_dims] slices to come last.
  for i in range(remain_start, remain_end):
    slc = data.draw(
        hps.one_of(
            hps.integers(min_value=0, max_value=batch_shape[i] - 1),
            arbitrary_slices))
    slices.append(slc)
  # 4. Decide where to insert between 0 and 7 newaxis slices.
  newaxis_positions = data.draw(
      hps.lists(hps.integers(min_value=0, max_value=len(slices)), max_size=7))
  for i in sorted(newaxis_positions, reverse=True):
    slices.insert(i, tf.newaxis)
  slices = tuple(slices)
  # Since `d[0]` ==> `d.__getitem__(0)` instead of `d.__getitem__((0,))`;
  # and similarly `d[:3]` ==> `d.__getitem__(slice(None, 3))` instead of
  # `d.__getitem__((slice(None, 3),))`; it is useful to test such scenarios.
  if len(slices) == 1 and data.draw(hps.booleans()):
    # Sometimes only a single item non-tuple.
    return slices[0]
  return slices


def stringify_slices(slices):
  """Returns a list of strings describing the items in `slices`."""
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


@test_util.run_all_in_graph_and_eager_modes
class DistributionPropertiesTest(tf.test.TestCase):

  def _draw_param(self, data, constraint_fn, param_shape):
    """Draws the value of a single distribution parameter."""
    # TODO(bjp): Allow a wider range of floats.
    # float32s = hps.floats(
    #     np.finfo(np.float32).min / 2, np.finfo(np.float32).max / 2,
    #     allow_nan=False, allow_infinity=False)
    float32s = hps.floats(-200, 200, allow_nan=False, allow_infinity=False)
    param_val = data.draw(
        hpnp.arrays(dtype=np.float32, shape=param_shape, elements=float32s).map(
            tf.convert_to_tensor).map(constraint_fn).map(self.evaluate))
    self.assertTrue(np.all(np.isfinite(param_val)))
    return tf.convert_to_tensor(value=param_val, dtype=tf.float32)

  def _draw_params(self, data, dist_name, batch_shape):
    """Draws a dict of parameters which should yield the given batch shape."""
    _, params_event_ndims = INSTANTIABLE_DISTS[dist_name]
    event_dim = data.draw(hps.integers(min_value=2, max_value=6))

    remaining_params = set(params_event_ndims.keys())
    params_to_use = []
    while remaining_params:
      param = data.draw(hps.one_of(map(hps.just, remaining_params)))
      params_to_use.append(param)
      remaining_params.remove(param)
      for mutex_set in MUTEX_PARAMS:
        if param in mutex_set:
          remaining_params -= mutex_set

    param_batch_shapes = draw_broadcasting_shapes(data, batch_shape,
                                                  params_to_use)
    params_kwargs = dict()
    for param in params_to_use:
      param_batch_shape = param_batch_shapes[param]
      param_event_rank = params_event_ndims[param]
      params_kwargs[param] = self._draw_param(
          data, constraint_for(dist_name, param),
          param_batch_shape.as_list() + [event_dim] * param_event_rank)
    return params_kwargs

  def _assert_shapes_match(self, target_shaped_dict, possibly_bcast_dict):
    for param, target_param_val in six.iteritems(target_shaped_dict):
      self.assertAllEqual(target_param_val.shape,
                          possibly_bcast_dict[param].shape)

  def _make_distribution(self, data, dist_name=None):
    """Samples one a set of supported distributions."""
    dist_names = hps.one_of(map(hps.just, INSTANTIABLE_DISTS.keys()))
    if dist_name is None:
      dist_name = data.draw(dist_names)

    dist_cls, _ = INSTANTIABLE_DISTS[dist_name]
    if dist_name == 'Independent':
      underlying_dist_name = data.draw(
          dist_names.filter(lambda d: d != 'Independent'))
      underlying, batch_shape = self._make_distribution(
          data, dist_name=underlying_dist_name)
      reinterpreted_batch_ndims = data.draw(
          hps.integers(min_value=0, max_value=len(batch_shape)))
      return (dist_cls(
          underlying, reinterpreted_batch_ndims=reinterpreted_batch_ndims),
              batch_shape[:len(batch_shape) - reinterpreted_batch_ndims])

    batch_shape = data.draw(rank_only_shape(0, 4))
    batch_rank = batch_shape.ndims
    if batch_rank:
      batch_shape = data.draw(
          hpnp.array_shapes(min_dims=batch_rank,
                            max_dims=batch_rank).map(tf.TensorShape))

    params_kwargs = self._draw_params(data, dist_name, batch_shape)
    params_constrained = constraint_for(dist_name)(params_kwargs)
    self._assert_shapes_match(params_kwargs, params_constrained)
    params_constrained['validate_args'] = True
    dist = dist_cls(**params_constrained)
    logging.info(
        'distribution: %s; parameters used: %s', dist,
        [k for k, v in six.iteritems(dist.parameters) if v is not None])
    return dist, batch_shape

  def _test_slicing(self, data, dist, batch_shape):
    slices = draw_valid_slices(data, batch_shape)
    if not slices:  # Nothing further to check.
      return
    slice_str = 'dist[{}]'.format(', '.join(stringify_slices(slices)))
    logging.info('slice used: %s', slice_str)
    # Make sure the slice string appears in Hypothesis' attempted example log,
    # by drawing and discarding it.
    data.draw(hps.just(slice_str))
    sliced_dist = dist.__getitem__(slices)
    self.assertAllEqual(
        np.zeros(batch_shape).__getitem__(slices).shape,
        sliced_dist.batch_shape)

  def _run_test(self, data):
    dist, batch_shape = self._make_distribution(data)
    self.assertAllEqual(batch_shape, dist.batch_shape)

    self._test_slicing(data, dist, batch_shape)

    # TODO(bjp): Enable sampling and log_prob checks. Currently, too many errors
    #     from out-of-domain samples.
    # if not isinstance(dist, tfd.Binomial):  # Binomial has no sampler.
    #   self.evaluate(dist.log_prob(dist.sample()))

  # We have a separate test for each of eager and graph in order to be able to
  # use @hp.reproduce_failure(..).
  @hp.given(hps.data())
  @hp.settings(deadline=None, suppress_health_check=[hp.HealthCheck.too_slow])
  def testDistributionsEager(self, data):
    if not tf.executing_eagerly(): return
    self._run_test(data)

  # We have a separate test for each of eager and graph in order to be able to
  # use @hp.reproduce_failure(..).
  @hp.given(hps.data())
  @hp.settings(deadline=None, suppress_health_check=[hp.HealthCheck.too_slow])
  def testDistributionsGraph(self, data):
    if tf.executing_eagerly(): return
    self._run_test(data)


# Functions used to constrain randomly sampled parameter ndarrays.
# TODO(b/128518790): Eliminate / minimize the fudge factors in here.


def identity_fn(x):
  return x


def softplus_plus_eps(eps=1e-6):
  return lambda x: tf.nn.softplus(x) + eps


def sigmoid_plus_eps(eps=1e-6):
  return lambda x: tf.sigmoid(x) * (1 - eps) + eps


def ensure_high_gt_low(low, high):
  """Returns a value with shape matching `high` and gt broadcastable `low`."""
  new_high = tf.maximum(low + tf.abs(low) * .1 + .1, high)
  reduce_dims = []
  if new_high.shape.ndims > high.shape.ndims:
    reduced_leading_axes = tf.range(new_high.shape.ndims - high.shape.ndims)
    new_high = tf.math.reduce_max(
        input_tensor=new_high, axis=reduced_leading_axes)
  reduce_dims = [
      d for d in range(high.shape.ndims) if high.shape[d] < new_high.shape[d]
  ]
  if reduce_dims:
    new_high = tf.math.reduce_max(
        input_tensor=new_high, axis=reduce_dims, keepdims=True)
  return new_high


def positive_definite(x):
  shp = x.shape.as_list()
  return (tf.matmul(x, x, transpose_b=True) +
          .1 * tf.linalg.eye(shp[-1], batch_shape=shp[:-2]))


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
        tf.nn.softplus,
    'rtol':
        tf.nn.softplus,
    'concentration':
        softplus_plus_eps(),
    'concentration0':
        softplus_plus_eps(),
    'concentration1':
        softplus_plus_eps(),
    'covariance_matrix':
        positive_definite,
    'df':
        softplus_plus_eps(),
    'Chi2WithAbsDf.df':
        softplus_plus_eps(1),  # does floor(abs(x)) for some reason
    'InverseGaussian.loc':
        softplus_plus_eps(),
    'VonMisesFisher.mean_direction':  # max ndims is 5
        lambda x: tf.nn.l2_normalize(tf.nn.sigmoid(x[..., :5]) + 1e-6, -1),
    'Categorical.probs':
        tf.nn.softmax,
    'ExpRelaxedOneHotCategorical.probs':
        tf.nn.softmax,
    'Multinomial.probs':
        tf.nn.softmax,
    'OneHotCategorical.probs':
        tf.nn.softmax,
    'RelaxedCategorical.probs':
        tf.nn.softmax,
    'Zipf.power':
        softplus_plus_eps(1 + 1e-6),  # strictly > 1
    'Geometric.logits':  # TODO(b/128410109): re-enable down to -50
        lambda x: tf.maximum(x, -17),  # works around the bug
    'Geometric.probs':
        sigmoid_plus_eps(),
    'Binomial.probs':
        tf.sigmoid,
    'NegativeBinomial.probs':
        tf.sigmoid,
    'Bernoulli.probs':
        tf.sigmoid,
    'RelaxedBernoulli.probs':
        tf.sigmoid,
    'mixing_concentration':
        softplus_plus_eps(),
    'mixing_rate':
        softplus_plus_eps(),
    'rate':
        softplus_plus_eps(),
    'scale':
        softplus_plus_eps(),
    'Wishart.scale':
        positive_definite,
    'scale_tril':
        lambda x: tf.linalg.band_part(  # pylint: disable=g-long-lambda
            tfd.matrix_diag_transform(x, softplus_plus_eps()), -1, 0),
    'temperature':
        softplus_plus_eps(),
    'total_count':
        lambda x: tf.floor(tf.sigmoid(x / 100) * 100) + 1,
    'LKJ':
        lambda d: dict(d, concentration=d['concentration'] + 1, dimension=3),
    'Triangular':
        fix_triangular,
    'TruncatedNormal':
        lambda d: dict(d, high=ensure_high_gt_low(d['low'], d['high'])),
    'Uniform':
        lambda d: dict(d, high=ensure_high_gt_low(d['low'], d['high'])),
    'Wishart':
        fix_wishart,
}


def constraint_for(dist=None, param=None):
  if param is not None:
    return CONSTRAINTS.get('{}.{}'.format(dist, param),
                           CONSTRAINTS.get(param, identity_fn))
  return CONSTRAINTS.get(dist, identity_fn)


if __name__ == '__main__':
  tf.test.main()
