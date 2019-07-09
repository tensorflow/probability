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

import collections
import functools
import inspect
import traceback

from absl import flags
from absl import logging
from absl.testing import parameterized
import hypothesis as hp
from hypothesis import strategies as hps
from hypothesis.extra import numpy as hpnp
import numpy as np
import six
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.bijectors import hypothesis_testlib as bijector_hps
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

tfd = tfp.distributions

flags.DEFINE_enum('tf_mode', 'graph', ['eager', 'graph'],
                  'TF execution mode to use')

FLAGS = flags.FLAGS


TF2_FRIENDLY_DISTS = (
    'Bernoulli',
    'Beta',
    'Categorical',
    'Deterministic',
    'Dirichlet',
    'Exponential',
    'Gamma',
    'GeneralizedPareto',
    'HalfNormal',
    'Laplace',
    'LogNormal',
    'Logistic',
    'Normal',
    'Multinomial',
    'Poisson',
    'ProbitBernoulli',
    'StudentT'
)

NO_SAMPLE_PARAM_GRADS = {
    'Deterministic': ('atol', 'rtol'),
}
NO_LOG_PROB_PARAM_GRADS = ('Deterministic',)
NO_KL_PARAM_GRADS = ('Deterministic',)

MUTEX_PARAMS = (
    set(['logits', 'probs']),
    set(['probits', 'probs']),
    set(['rate', 'log_rate']),
    set(['scale', 'scale_tril', 'scale_diag', 'scale_identity_multiplier']),
)

SPECIAL_DISTS = (
    'ConditionalDistribution',
    'ConditionalTransformedDistribution',
    'Distribution',
    'Empirical',
    'Independent',
    'MixtureSameFamily',
    'TransformedDistribution',
)


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
  result['MixtureSameFamily'] = (tfd.MixtureSameFamily, None)
  result['TransformedDistribution'] = (tfd.TransformedDistribution, None)
  return result


# INSTANTIABLE_DISTS is a map from str->(DistClass, params_event_ndims)
INSTANTIABLE_DISTS = instantiable_dists()
del instantiable_dists


# pylint is unable to handle @hps.composite (e.g. complains "No value for
# argument 'batch_shape' in function call"), so disable this lint for the file.

# pylint: disable=no-value-for-parameter


@hps.composite
def valid_slices(draw, batch_shape):
  """Samples a legal (possibly empty) slice for shape batch_shape."""
  # We build up a list of slices in several stages:
  # 1. Choose 0 to batch_rank slices to come before an Ellipsis (...).
  # 2. Decide whether or not to add an Ellipsis; if using, updating the indexing
  #    used (e.g. batch_shape[i]) to identify safe bounds.
  # 3. Choose 0 to [remaining_dims] slices to come last.
  # 4. Decide where to insert between 0 and 4 newaxis slices.
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
  # 4. Decide where to insert between 0 and 4 newaxis slices.
  newaxis_positions = draw(
      hps.lists(hps.integers(min_value=0, max_value=len(slices)), max_size=4))
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


@hps.composite
def broadcasting_params(draw,
                        dist_name,
                        batch_shape,
                        event_dim=None,
                        enable_vars=False):
  """Draws a dict of parameters which should yield the given batch shape."""
  _, params_event_ndims = INSTANTIABLE_DISTS[dist_name]

  def _constraint(param):
    return constraint_for(dist_name, param)

  return draw(
      tfp_hps.broadcasting_params(
          batch_shape,
          event_dim=event_dim,
          enable_vars=enable_vars,
          params_event_ndims=params_event_ndims,
          constraint_fn_for=_constraint,
          mutex_params=MUTEX_PARAMS))


@hps.composite
def independents(draw, batch_shape=None, event_dim=None, enable_vars=False):
  reinterpreted_batch_ndims = draw(hps.integers(min_value=0, max_value=2))
  if batch_shape is None:
    batch_shape = draw(
        tfp_hps.batch_shapes(min_ndims=reinterpreted_batch_ndims))
  else:  # This independent adds some batch dims to its underlying distribution.
    batch_shape = tensorshape_util.concatenate(
        batch_shape,
        draw(
            tfp_hps.batch_shapes(
                min_ndims=reinterpreted_batch_ndims,
                max_ndims=reinterpreted_batch_ndims)))
  underlying, batch_shape = draw(
      distributions(
          batch_shape=batch_shape,
          event_dim=event_dim,
          enable_vars=enable_vars,
          eligibility_filter=lambda name: name != 'Independent'))
  logging.info(
      'underlying distribution: %s; parameters used: %s', underlying,
      [k for k, v in six.iteritems(underlying.parameters) if v is not None])
  return (tfd.Independent(
      underlying,
      reinterpreted_batch_ndims=reinterpreted_batch_ndims,
      validate_args=True),
          batch_shape[:len(batch_shape) - reinterpreted_batch_ndims])


@hps.composite
def transformed_distributions(draw,
                              batch_shape=None,
                              event_dim=None,
                              enable_vars=False):
  bijector = draw(bijector_hps.unconstrained_bijectors())
  logging.info('TD bijector: %s', bijector)
  if batch_shape is None:
    batch_shape = draw(tfp_hps.batch_shapes())
  underlying_batch_shape = batch_shape
  batch_shape_arg = None
  if draw(hps.booleans()):
    # Use batch_shape overrides.
    underlying_batch_shape = tf.TensorShape([])  # scalar underlying batch
    batch_shape_arg = batch_shape
  underlyings = distributions(
      batch_shape=underlying_batch_shape,
      event_dim=event_dim,
      enable_vars=enable_vars).map(
          lambda dist_and_batch_shape: dist_and_batch_shape[0]).filter(
              bijector_hps.distribution_filter_for(bijector))
  to_transform = draw(underlyings)
  logging.info(
      'TD underlying distribution: %s; parameters used: %s', to_transform,
      [k for k, v in six.iteritems(to_transform.parameters) if v is not None])
  return (tfd.TransformedDistribution(
      bijector=bijector,
      distribution=to_transform,
      batch_shape=batch_shape_arg,
      validate_args=True), batch_shape)


@hps.composite
def mixtures_same_family(draw,
                         batch_shape=None,
                         event_dim=None,
                         enable_vars=False):
  if batch_shape is None:
    # Ensure the components dist has at least one batch dim (a component dim).
    batch_shape = draw(tfp_hps.batch_shapes(min_ndims=1, min_lastdimsize=2))
  else:  # This mixture adds a batch dim to its underlying components dist.
    batch_shape = tensorshape_util.concatenate(
        batch_shape,
        draw(tfp_hps.batch_shapes(min_ndims=1, max_ndims=1, min_lastdimsize=2)))

  component_dist, _ = draw(
      distributions(
          batch_shape=batch_shape,
          event_dim=event_dim,
          enable_vars=enable_vars,
          eligibility_filter=lambda name: name != 'MixtureSameFamily'))
  logging.info(
      'component distribution: %s; parameters used: %s', component_dist,
      [k for k, v in six.iteritems(component_dist.parameters) if v is not None])
  # scalar or same-shaped categorical?
  mixture_batch_shape = draw(
      hps.one_of(hps.just(batch_shape[:-1]), hps.just(tf.TensorShape([]))))
  mixture_dist, _ = draw(distributions(
      dist_name='Categorical',
      batch_shape=mixture_batch_shape,
      event_dim=tensorshape_util.as_list(batch_shape)[-1]))
  logging.info(
      'mixture distribution: %s; parameters used: %s', mixture_dist,
      [k for k, v in six.iteritems(mixture_dist.parameters) if v is not None])
  return (tfd.MixtureSameFamily(
      components_distribution=component_dist,
      mixture_distribution=mixture_dist,
      validate_args=True), batch_shape[:-1])


def assert_shapes_unchanged(target_shaped_dict, possibly_bcast_dict):
  for param, target_param_val in six.iteritems(target_shaped_dict):
    np.testing.assert_array_equal(
        tensorshape_util.as_list(target_param_val.shape),
        tensorshape_util.as_list(possibly_bcast_dict[param].shape))


@hps.composite
def distributions(draw,
                  dist_name=None,
                  batch_shape=None,
                  event_dim=None,
                  enable_vars=False,
                  eligibility_filter=lambda name: True):
  """Samples one a set of supported distributions."""
  if dist_name is None:

    dist_name = draw(
        hps.one_of(
            map(hps.just,
                [k for k in INSTANTIABLE_DISTS.keys() if eligibility_filter(k)])
            ))

  dist_cls, _ = INSTANTIABLE_DISTS[dist_name]
  if dist_name == 'Independent':
    return draw(independents(batch_shape, event_dim, enable_vars))
  if dist_name == 'MixtureSameFamily':
    return draw(mixtures_same_family(batch_shape, event_dim, enable_vars))
  if dist_name == 'TransformedDistribution':
    return draw(transformed_distributions(batch_shape, event_dim, enable_vars))

  if batch_shape is None:
    batch_shape = draw(tfp_hps.batch_shapes())

  params_kwargs = draw(
      broadcasting_params(
          dist_name, batch_shape, event_dim=event_dim, enable_vars=enable_vars))
  params_constrained = constraint_for(dist_name)(params_kwargs)
  assert_shapes_unchanged(params_kwargs, params_constrained)
  params_constrained['validate_args'] = True
  return dist_cls(**params_constrained), batch_shape


def maybe_seed(seed):
  return tf1.set_random_seed(seed) if tf.executing_eagerly() else seed


def get_event_dim(dist):
  for param, val in dist.parameters.items():
    if (param in dist._params_event_ndims() and val is not None and
        dist._params_event_ndims()[param]):
      return tf.compat.dimension_value(val.shape[-1])


@test_util.run_all_in_graph_and_eager_modes
class DistributionParamsAreVarsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters((dname,) for dname in TF2_FRIENDLY_DISTS)
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testDistribution(self, dist_name, data):
    if tf.executing_eagerly() != (FLAGS.tf_mode == 'eager'):
      return
    tf1.set_random_seed(
        data.draw(
            hpnp.arrays(dtype=np.int64, shape=[]).filter(lambda x: x != 0)))
    dist, batch_shape = data.draw(
        distributions(dist_name=dist_name, enable_vars=True))
    batch_shape2 = data.draw(tfp_hps.broadcast_compatible_shape(batch_shape))
    dist2, _ = data.draw(
        distributions(
            dist_name=dist_name,
            batch_shape=batch_shape2,
            event_dim=get_event_dim(dist),
            enable_vars=True))
    del batch_shape
    logging.info(
        'distribution: %s; parameters used: %s', dist,
        [k for k, v in six.iteritems(dist.parameters) if v is not None])
    self.evaluate([var.initializer for var in dist.variables])
    for k, v in six.iteritems(dist.parameters):
      if not tensor_util.is_mutable(v):
        continue
      try:
        self.assertIs(getattr(dist, k), v)
      except AssertionError as e:
        raise AssertionError(
            'No attr found for parameter {} of distribution {}: \n{}'.format(
                k, dist_name, e))

    for stat in data.draw(
        hps.sets(
            hps.one_of(
                map(hps.just, [
                    'covariance', 'entropy', 'mean', 'mode', 'stddev',
                    'variance'
                ])),
            min_size=3,
            max_size=3)):
      logging.info('%s.%s', dist_name, stat)
      try:
        with tfp_hps.assert_no_excessive_var_usage(
            'statistic `{}` of `{}`'.format(stat, dist)):
          getattr(dist, stat)()

      except NotImplementedError:
        pass

    with tf.GradientTape() as tape:
      with tfp_hps.assert_no_excessive_var_usage(
          'method `sample` of `{}`'.format(dist)):
        sample = dist.sample()
    if dist.reparameterization_type == tfd.FULLY_REPARAMETERIZED:
      grads = tape.gradient(sample, dist.variables)
      for grad, var in zip(grads, dist.variables):
        var_name = var.name.rstrip('_0123456789:')
        if var_name in NO_SAMPLE_PARAM_GRADS.get(dist_name, ()):
          continue
        if grad is None:
          raise AssertionError(
              'Missing sample -> {} grad for distribution {}'.format(
                  var_name, dist_name))

    # Turn off validations, since log_prob can choke on dist's own samples.
    # Also, to relax conversion counts for KL (might do >2 w/ validate_args).
    dist = dist.copy(validate_args=False)
    dist2 = dist2.copy(validate_args=False)

    try:
      for d1, d2 in (dist, dist2), (dist2, dist):
        with tf.GradientTape() as tape:
          with tfp_hps.assert_no_excessive_var_usage(
              '`kl_divergence` of (`{}` (vars {}), `{}` (vars {}))'.format(
                  d1, d1.variables, d2, d2.variables),
              max_permissible=1):  # No validation => 1 convert per var.
            kl = d1.kl_divergence(d2)
        wrt_vars = list(d1.variables) + list(d2.variables)
        grads = tape.gradient(kl, wrt_vars)
        for grad, var in zip(grads, wrt_vars):
          if grad is None and dist_name not in NO_KL_PARAM_GRADS:
            raise AssertionError('Missing KL({} || {}) -> {} grad:\n'
                                 '{} vars: {}\n{} vars: {}'.format(
                                     d1, d2, var, d1, d1.variables, d2,
                                     d2.variables))
    except NotImplementedError:
      pass

    if dist_name not in NO_LOG_PROB_PARAM_GRADS:
      with tf.GradientTape() as tape:
        lp = dist.log_prob(tf.stop_gradient(sample))
      grads = tape.gradient(lp, dist.variables)
      for grad, var in zip(grads, dist.variables):
        if grad is None:
          raise AssertionError(
              'Missing log_prob -> {} grad for distribution {}'.format(
                  var, dist_name))

    for evaluative in data.draw(
        hps.sets(
            hps.one_of(
                map(hps.just, [
                    'log_prob', 'prob', 'log_cdf', 'cdf',
                    'log_survival_function', 'survival_function'
                ])),
            min_size=3,
            max_size=3)):
      logging.info('%s.%s', dist_name, evaluative)
      try:
        with tfp_hps.assert_no_excessive_var_usage(
            'evaluative `{}` of `{}`'.format(evaluative, dist),
            max_permissible=1):  # No validation => 1 convert
          getattr(dist, evaluative)(sample)
      except NotImplementedError:
        pass


@test_util.run_all_in_graph_and_eager_modes
class DistributionSlicingTest(tf.test.TestCase):

  def _test_slicing(self, data, dist, batch_shape):
    slices = data.draw(valid_slices(batch_shape))
    slice_str = 'dist[{}]'.format(', '.join(stringify_slices(slices)))
    logging.info('slice used: %s', slice_str)
    # Make sure the slice string appears in Hypothesis' attempted example log,
    # by drawing and discarding it.
    data.draw(hps.just(slice_str))
    if not slices:  # Nothing further to check.
      return
    sliced_zeros = np.zeros(batch_shape)[slices]
    sliced_dist = dist[slices]
    self.assertAllEqual(sliced_zeros.shape, sliced_dist.batch_shape)

    try:
      seed = data.draw(
          hpnp.arrays(dtype=np.int64, shape=[]).filter(lambda x: x != 0))
      samples = self.evaluate(dist.sample(seed=maybe_seed(seed)))

      if not sliced_zeros.size:
        # TODO(b/128924708): Fix distributions that fail on degenerate empty
        #     shapes, e.g. Multinomial, DirichletMultinomial, ...
        return

      sliced_samples = self.evaluate(sliced_dist.sample(seed=maybe_seed(seed)))
    except NotImplementedError as e:
      raise
    except tf.errors.UnimplementedError as e:
      if 'Unhandled input dimensions' in str(e) or 'rank not in' in str(e):
        # Some cases can fail with 'Unhandled input dimensions \d+' or
        # 'inputs rank not in [0,6]: \d+'
        return
      raise

    # Come up with the slices for samples (which must also include event dims).
    sample_slices = (
        tuple(slices) if isinstance(slices, collections.Sequence) else
        (slices,))
    if Ellipsis not in sample_slices:
      sample_slices += (Ellipsis,)
    sample_slices += tuple([slice(None)] *
                           tensorshape_util.rank(dist.event_shape))

    # Report sub-sliced samples (on which we compare log_prob) to hypothesis.
    data.draw(hps.just(samples[sample_slices]))
    self.assertAllEqual(samples[sample_slices].shape, sliced_samples.shape)
    try:
      try:
        lp = self.evaluate(dist.log_prob(samples))
      except tf.errors.InvalidArgumentError:
        # TODO(b/129271256): d.log_prob(d.sample()) should not fail
        #     validate_args checks.
        # We only tolerate this case for the non-sliced dist.
        return
      sliced_lp = self.evaluate(sliced_dist.log_prob(samples[sample_slices]))
    except tf.errors.UnimplementedError as e:
      if 'Unhandled input dimensions' in str(e) or 'rank not in' in str(e):
        # Some cases can fail with 'Unhandled input dimensions \d+' or
        # 'inputs rank not in [0,6]: \d+'
        return
      raise
    # TODO(b/128708201): Better numerics for Geometric/Beta?
    # Eigen can return quite different results for packet vs non-packet ops.
    # To work around this, we use a much larger rtol for the last 3
    # (assuming packet size 4) elements.
    packetized_lp = lp[slices].reshape(-1)[:-3]
    packetized_sliced_lp = sliced_lp.reshape(-1)[:-3]
    rtol = (0.1 if any(
        x in dist.name for x in ('Geometric', 'Beta', 'Dirichlet')) else 0.02)
    self.assertAllClose(packetized_lp, packetized_sliced_lp, rtol=rtol)
    possibly_nonpacket_lp = lp[slices].reshape(-1)[-3:]
    possibly_nonpacket_sliced_lp = sliced_lp.reshape(-1)[-3:]
    rtol = 0.4
    self.assertAllClose(
        possibly_nonpacket_lp, possibly_nonpacket_sliced_lp, rtol=rtol)

  def _run_test(self, data):
    tf1.set_random_seed(
        data.draw(
            hpnp.arrays(dtype=np.int64, shape=[]).filter(lambda x: x != 0)))
    dist, batch_shape = data.draw(distributions())
    logging.info(
        'distribution: %s; parameters used: %s', dist,
        [k for k, v in six.iteritems(dist.parameters) if v is not None])
    self.assertAllEqual(batch_shape, dist.batch_shape)

    with self.assertRaisesRegexp(TypeError, 'not iterable'):
      iter(dist)  # __getitem__ magically makes an object iterable.

    self._test_slicing(data, dist, batch_shape)

    # TODO(bjp): Enable sampling and log_prob checks. Currently, too many errors
    #     from out-of-domain samples.
    # self.evaluate(dist.log_prob(dist.sample()))

  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testDistributions(self, data):
    if tf.executing_eagerly() != (FLAGS.tf_mode == 'eager'): return
    self._run_test(data)


# Functions used to constrain randomly sampled parameter ndarrays.
# TODO(b/128518790): Eliminate / minimize the fudge factors in here.


def sigmoid_plus_eps(eps=1e-6):
  return lambda x: tf.sigmoid(x) * (1 - eps) + eps


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
        input_tensor=new_high, axis=reduced_leading_axes)
  reduce_dims = [
      d for d in range(tensorshape_util.rank(high.shape))
      if high.shape[d] < new_high.shape[d]
  ]
  if reduce_dims:
    new_high = tf.math.reduce_max(
        input_tensor=new_high, axis=reduce_dims, keepdims=True)
  return new_high


def fix_finite_discrete(d):
  size = d.get('probs', d.get('logits', None)).shape[-1]
  return dict(d, outcomes=tf.linspace(-1.0, 1.0, size))


def fix_lkj(d):
  return dict(d, concentration=d['concentration'] + 1, dimension=3)


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
    'Chi2WithAbsDf.df':
        tfp_hps.softplus_plus_eps(1),  # does floor(abs(x)) for some reason
    'InverseGaussian.loc':
        tfp_hps.softplus_plus_eps(),
    'VonMisesFisher.mean_direction':  # max ndims is 5
        lambda x: tf.math.l2_normalize(tf.math.sigmoid(x[..., :5]) + 1e-6, -1),
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
    'Geometric.logits':  # TODO(b/128410109): re-enable down to -50
        lambda x: tf.maximum(x, -16.),  # works around the bug
    'Geometric.probs':
        sigmoid_plus_eps(),
    'Binomial.probs':
        tf.sigmoid,
    'NegativeBinomial.probs':
        tf.sigmoid,
    'Bernoulli.probs':
        tf.sigmoid,
    'ProbitBernoulli.probs':
        tf.sigmoid,
    'RelaxedBernoulli.probs':
        tf.sigmoid,
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
    'MultivariateNormalDiagWithSoftplusScale.scale_diag':
        lambda x: tf.maximum(x, -87.),  # softplus(-87) ~= 1e-38
    'scale_identity_multiplier':
        tfp_hps.softplus_plus_eps(),
    'scale_tril':
        tfp_hps.lower_tril_positive_definite,
    'temperature':
        tfp_hps.softplus_plus_eps(),
    'total_count':
        lambda x: tf.floor(tf.sigmoid(x / 100) * 100) + 1,
    'Bernoulli':
        lambda d: dict(d, dtype=tf.float32),
    'LKJ':
        fix_lkj,
    'Triangular':
        fix_triangular,
    'TruncatedNormal':
        lambda d: dict(d, high=ensure_high_gt_low(d['low'], d['high'])),
    'Uniform':
        lambda d: dict(d, high=ensure_high_gt_low(d['low'], d['high'])),
    'Wishart':
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


if __name__ == '__main__':
  tf.test.main()
