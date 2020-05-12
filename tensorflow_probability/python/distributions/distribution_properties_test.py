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

from absl.testing import parameterized
import hypothesis as hp
from hypothesis import strategies as hps
import numpy as np
import six
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions import hypothesis_testlib as dhps
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


TF2_FRIENDLY_DISTS = (
    'Bernoulli',
    'Beta',
    'BetaBinomial',
    'Binomial',
    'Chi',
    'Chi2',
    'CholeskyLKJ',
    'Categorical',
    'Cauchy',
    'ContinuousBernoulli',
    'Deterministic',
    'Dirichlet',
    'DirichletMultinomial',
    'DoublesidedMaxwell',
    'Empirical',
    'Exponential',
    'FiniteDiscrete',
    'Gamma',
    'GammaGamma',
    'GeneralizedPareto',
    'Geometric',
    'Gumbel',
    'HalfCauchy',
    'HalfNormal',
    'HalfStudentT',
    'Horseshoe',
    'InverseGamma',
    'InverseGaussian',
    'JohnsonSU',
    'Kumaraswamy',
    'Laplace',
    'LKJ',
    'LogNormal',
    'Logistic',
    'Normal',
    'Multinomial',
    'NegativeBinomial',
    'OneHotCategorical',
    'OrderedLogistic',
    'Pareto',
    'PERT',
    'PlackettLuce',
    'Poisson',
    # 'PoissonLogNormalQuadratureCompound' TODO(b/137956955): Add support
    # for hypothesis testing
    'ProbitBernoulli',
    'RelaxedBernoulli',
    'ExpRelaxedOneHotCategorical',
    # 'SinhArcsinh' TODO(b/137956955): Add support for hypothesis testing
    'StudentT',
    'Triangular',
    'TruncatedNormal',
    'Uniform',
    'VonMises',
    'VonMisesFisher',
    'WishartTriL',
    'Zipf',
)

NO_SAMPLE_PARAM_GRADS = {
    'Deterministic': ('atol', 'rtol'),
}
NO_LOG_PROB_PARAM_GRADS = ('Deterministic', 'Empirical')
NO_KL_PARAM_GRADS = ('Deterministic',)

# Batch slicing requires implementing `_params_event_ndims`.  Generic
# instantiation (per `instantiable_base_dists`, below) also requires
# `_params_event_ndims`, but some special distributions can be instantiated
# without that.  Of those, this variable lists the ones that do not support
# batch slicing.
INSTANTIABLE_BUT_NOT_SLICABLE = (
    'BatchReshape',
)

EXTRA_TENSOR_CONVERSION_DISTS = {
    'RelaxedBernoulli': 1,
    'WishartTriL': 3,  # not concretizing linear operator scale
    'Chi': 2,  # subclasses `Chi2`, runs redundant checks on `df` parameter
}


# TODO(b/130815467) All distributions should be auto-vectorizeable.
# The lists below contain distributions from INSTANTIABLE_BASE_DISTS that are
# blacklisted by the autovectorization tests. Since not all distributions are
# in INSTANTIABLE_BASE_DISTS, these should not be taken as exhaustive.
SAMPLE_AUTOVECTORIZATION_IS_BROKEN = [
    'DirichletMultinomial',  # No converter for StatelessWhile
    'Gamma',  # "Incompatible shapes" error. (b/150712618).
    'Multinomial',  # No converter for StatelessWhile
    'PlackettLuce',  # No converter for TopKV2
    'TruncatedNormal',  # No converter for ParameterizedTruncatedNormal
]

LOGPROB_AUTOVECTORIZATION_IS_BROKEN = [
    'StudentT',  # Numerical problem: b/149785284
    'HalfStudentT',  # Numerical problem: b/149785284
    'TruncatedNormal',  # Numerical problem: b/150811273
    'VonMisesFisher',  # No converter for CheckNumerics
    'Wishart',  # Actually works, but disabled because log_prob of sample is
                # ill-conditioned for reasons unrelated to pfor.
    'WishartTriL',  # Same as Wishart.
]

EVENT_SPACE_BIJECTOR_IS_BROKEN = [
    'InverseGamma',  # TODO(b/143090143): Enable this when the bug is fixed.
                     # (Reciprocal(Softplus(x)) -> inf for small x)
]


# Vectorization can rewrite computations in ways that (apparently) lead to
# minor floating-point inconsistency.
# TODO(b/142827327): Bring tolerance down to 0 for all distributions.
VECTORIZED_LOGPROB_ATOL = collections.defaultdict(lambda: 1e-6)
VECTORIZED_LOGPROB_ATOL.update({
    'CholeskyLKJ': 1e-4,
    'LKJ': 1e-3,
    'BetaBinomial': 1e-5,
})

VECTORIZED_LOGPROB_RTOL = collections.defaultdict(lambda: 1e-6)
VECTORIZED_LOGPROB_RTOL.update({
    'NegativeBinomial': 1e-5,
})


def extra_tensor_conversions_allowed(dist):
  """Returns number of extra tensor conversions allowed for the input dist."""
  extra_conversions = EXTRA_TENSOR_CONVERSION_DISTS.get(type(dist).__name__)
  if extra_conversions:
    return extra_conversions
  if isinstance(dist, tfd.TransformedDistribution):
    return 1
  if isinstance(dist, tfd.BatchReshape):
    # One for the batch_shape_tensor needed by _call_reshape_input_output.
    # One to cover inability to turn off validate_args for the base
    # distribution (b/143297494).
    return 2
  return 0


@test_util.test_all_tf_execution_regimes
class DistributionParamsAreVarsTest(test_util.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': dname, 'dist_name': dname}
      for dname in TF2_FRIENDLY_DISTS)
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testDistribution(self, dist_name, data):
    seed = test_util.test_seed()
    # Explicitly draw event_dim here to avoid relying on _params_event_ndims
    # later, so this test can support distributions that do not implement the
    # slicing protocol.
    event_dim = data.draw(hps.integers(min_value=2, max_value=6))
    dist = data.draw(dhps.distributions(
        dist_name=dist_name, event_dim=event_dim, enable_vars=True))
    batch_shape = dist.batch_shape
    batch_shape2 = data.draw(tfp_hps.broadcast_compatible_shape(batch_shape))
    dist2 = data.draw(
        dhps.distributions(
            dist_name=dist_name,
            batch_shape=batch_shape2,
            event_dim=event_dim,
            enable_vars=True))
    self.evaluate([var.initializer for var in dist.variables])

    # Check that the distribution passes Variables through to the accessor
    # properties (without converting them to Tensor or anything like that).
    for k, v in six.iteritems(dist.parameters):
      if not tensor_util.is_ref(v):
        continue
      self.assertIs(getattr(dist, k), v)

    # Check that standard statistics do not read distribution parameters more
    # than twice (once in the stat itself and up to once in any validation
    # assertions).
    max_permissible = 2 + extra_tensor_conversions_allowed(dist)
    for stat in sorted(data.draw(
        hps.sets(
            hps.one_of(
                map(hps.just, [
                    'covariance', 'entropy', 'mean', 'mode', 'stddev',
                    'variance'
                ])),
            min_size=3,
            max_size=3))):
      hp.note('Testing excessive var usage in {}.{}'.format(dist_name, stat))
      try:
        with tfp_hps.assert_no_excessive_var_usage(
            'statistic `{}` of `{}`'.format(stat, dist),
            max_permissible=max_permissible):
          getattr(dist, stat)()

      except NotImplementedError:
        pass

    # Check that `sample` doesn't read distribution parameters more than twice,
    # and that it produces non-None gradients (if the distribution is fully
    # reparameterized).
    with tf.GradientTape() as tape:
      # TDs do bijector assertions twice (once by distribution.sample, and once
      # by bijector.forward).
      max_permissible = 2 + extra_tensor_conversions_allowed(dist)
      with tfp_hps.assert_no_excessive_var_usage(
          'method `sample` of `{}`'.format(dist),
          max_permissible=max_permissible):
        sample = dist.sample(seed=seed)
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

    # Turn off validations, since TODO(b/129271256) log_prob can choke on dist's
    # own samples.  Also, to relax conversion counts for KL (might do >2 w/
    # validate_args).
    dist = dist.copy(validate_args=False)
    dist2 = dist2.copy(validate_args=False)

    # Test that KL divergence reads distribution parameters at most once, and
    # that is produces non-None gradients.
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
            raise AssertionError('Missing KL({} || {}) -> {} grad:\n'  # pylint: disable=duplicate-string-formatting-argument
                                 '{} vars: {}\n{} vars: {}'.format(
                                     d1, d2, var, d1, d1.variables, d2,
                                     d2.variables))
    except NotImplementedError:
      pass

    # Test that log_prob produces non-None gradients, except for distributions
    # on the NO_LOG_PROB_PARAM_GRADS blacklist.
    if dist_name not in NO_LOG_PROB_PARAM_GRADS:
      with tf.GradientTape() as tape:
        lp = dist.log_prob(tf.stop_gradient(sample))
      grads = tape.gradient(lp, dist.variables)
      for grad, var in zip(grads, dist.variables):
        if grad is None:
          raise AssertionError(
              'Missing log_prob -> {} grad for distribution {}'.format(
                  var, dist_name))

    # Test that all forms of probability evaluation avoid reading distribution
    # parameters more than once.
    for evaluative in sorted(data.draw(
        hps.sets(
            hps.one_of(
                map(hps.just, [
                    'log_prob', 'prob', 'log_cdf', 'cdf',
                    'log_survival_function', 'survival_function'
                ])),
            min_size=3,
            max_size=3))):
      hp.note('Testing excessive var usage in {}.{}'.format(
          dist_name, evaluative))
      try:
        # No validation => 1 convert. But for TD we allow 2:
        # dist.log_prob(bijector.inverse(samp)) + bijector.ildj(samp)
        max_permissible = 2 + extra_tensor_conversions_allowed(dist)
        with tfp_hps.assert_no_excessive_var_usage(
            'evaluative `{}` of `{}`'.format(evaluative, dist),
            max_permissible=max_permissible):
          getattr(dist, evaluative)(sample)
      except NotImplementedError:
        pass


@test_util.test_all_tf_execution_regimes
class ReproducibilityTest(test_util.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': dname, 'dist_name': dname}
      for dname in sorted(list(dhps.INSTANTIABLE_BASE_DISTS.keys()) +
                          list(dhps.INSTANTIABLE_META_DISTS)))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testDistribution(self, dist_name, data):
    dist = data.draw(dhps.distributions(dist_name=dist_name, enable_vars=False))
    seed = test_util.test_seed()
    with tfp_hps.no_tf_rank_errors():
      s1 = self.evaluate(dist.sample(50, seed=seed))
    if tf.executing_eagerly():
      tf.random.set_seed(seed)
    with tfp_hps.no_tf_rank_errors():
      s2 = self.evaluate(dist.sample(50, seed=seed))
    self.assertAllEqual(s1, s2)


@test_util.test_all_tf_execution_regimes
class EventSpaceBijectorsTest(test_util.TestCase):

  def check_bad_loc_scale(self, dist):
    if hasattr(dist, 'loc') and hasattr(dist, 'scale'):
      try:
        loc_ = tf.convert_to_tensor(dist.loc)
        scale_ = tf.convert_to_tensor(dist.scale)
      except (ValueError, TypeError):
        # If they're not Tensor-convertible, don't try to check them.  This is
        # the case, in, for example, multivariate normal, where the scale is a
        # `LinearOperator`.
        return
      loc, scale = self.evaluate([loc_, scale_])
      hp.assume(np.all(np.abs(loc / scale) < 1e7))

  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testDistribution(self, data):
    enable_vars = data.draw(hps.booleans())

    # TODO(b/146572907): Fix `enable_vars` for metadistributions.
    broken_dists = EVENT_SPACE_BIJECTOR_IS_BROKEN
    if enable_vars:
      broken_dists.extend(dhps.INSTANTIABLE_META_DISTS)

    dist = data.draw(
        dhps.distributions(
            enable_vars=enable_vars,
            eligibility_filter=(lambda name: name not in broken_dists)))
    self.evaluate([var.initializer for var in dist.variables])
    self.check_bad_loc_scale(dist)

    event_space_bijector = dist._experimental_default_event_space_bijector()
    if event_space_bijector is None:
      return

    total_sample_shape = tensorshape_util.concatenate(
        # Draw a sample shape
        data.draw(tfp_hps.shapes()),
        # Draw a shape that broadcasts with `[batch_shape, inverse_event_shape]`
        # where `inverse_event_shape` is the event shape in the bijector's
        # domain. This is the shape of `y` in R**n, such that
        # x = event_space_bijector(y) has the event shape of the distribution.
        data.draw(tfp_hps.broadcasting_shapes(
            tensorshape_util.concatenate(
                dist.batch_shape,
                event_space_bijector.inverse_event_shape(
                    dist.event_shape)), n=1))[0])

    y = data.draw(
        tfp_hps.constrained_tensors(
            tfp_hps.identity_fn, total_sample_shape.as_list()))
    x = event_space_bijector(y)
    with tf.control_dependencies(dist._sample_control_dependencies(x)):
      self.evaluate(tf.identity(x))


def _all_shapes(thing):
  if isinstance(thing, (tfd.Distribution, tfb.Bijector)):
    # pylint: disable=g-complex-comprehension
    answer = [s for _, param in thing.parameters.items()
              for s in _all_shapes(param)]
    if isinstance(thing, tfd.TransformedDistribution):
      answer = [thing.batch_shape + s for s in answer]
    if isinstance(thing, tfd.Distribution):
      answer += [thing.batch_shape + thing.event_shape]
    if isinstance(thing, tfd.MixtureSameFamily):
      num_components = thing.mixture_distribution.logits_parameter().shape[-1]
      answer += [thing.batch_shape + [num_components] + thing.event_shape]
    return answer
  elif tf.is_tensor(thing):
    return [thing.shape]
  else:
    # Assume the thing is some Python constant like a string or a boolean
    return []


def _all_ok(thing, one_ok):
  hp.note('Testing packetization of {}.'.format(thing))
  for s in _all_shapes(thing):
    if not one_ok(s):
      return False
  return True


def _all_packetized(thing):
  def one_ok(shape):
    ans = tf.TensorShape(shape).num_elements() > 1
    for dim in tf.TensorShape(shape):
      ans &= dim % 4 == 0
    if ans:
      hp.note('Presuming shape {} is packetized'.format(shape))
    else:
      hp.note('Not presuming shape {} is packetized'.format(shape))
    return ans
  return _all_ok(thing, one_ok)


def _all_non_packetized(thing):
  def one_ok(shape):
    ans = tf.TensorShape(shape).num_elements() < 4
    if ans:
      hp.note('Presuming shape {} is non-packetized'.format(shape))
    else:
      hp.note('Not presuming shape {} is non-packetized'.format(shape))
    return ans
  return _all_ok(thing, one_ok)


@test_util.test_all_tf_execution_regimes
class DistributionSlicingTest(test_util.TestCase):

  def _test_slicing(self, data, dist):
    strm = test_util.test_seed_stream()
    batch_shape = dist.batch_shape
    slices = data.draw(dhps.valid_slices(batch_shape))
    slice_str = 'dist[{}]'.format(', '.join(dhps.stringify_slices(
        slices)))
    # Make sure the slice string appears in Hypothesis' attempted example log
    hp.note('Using slice ' + slice_str)
    if not slices:  # Nothing further to check.
      return
    sliced_zeros = np.zeros(batch_shape)[slices]
    sliced_dist = dist[slices]
    hp.note('Using sliced distribution {}.'.format(sliced_dist))

    # Check that slicing modifies batch shape as expected.
    self.assertAllEqual(sliced_zeros.shape, sliced_dist.batch_shape)

    if not sliced_zeros.size:
      # TODO(b/128924708): Fix distributions that fail on degenerate empty
      #     shapes, e.g. Multinomial, DirichletMultinomial, ...
      return

    # Check that sampling of sliced distributions executes.
    with tfp_hps.no_tf_rank_errors():
      samples = self.evaluate(dist.sample(seed=strm()))
      sliced_dist_samples = self.evaluate(sliced_dist.sample(seed=strm()))

    # Come up with the slices for samples (which must also include event dims).
    sample_slices = (
        tuple(slices) if isinstance(slices, collections.Sequence) else
        (slices,))
    if Ellipsis not in sample_slices:
      sample_slices += (Ellipsis,)
    sample_slices += tuple([slice(None)] *
                           tensorshape_util.rank(dist.event_shape))

    sliced_samples = samples[sample_slices]

    # Report sub-sliced samples (on which we compare log_prob) to hypothesis.
    hp.note('Sample(s) for testing log_prob ' + str(sliced_samples))

    # Check that sampling a sliced distribution produces the same shape as
    # slicing the samples from the original.
    self.assertAllEqual(sliced_samples.shape, sliced_dist_samples.shape)

    # Check that a sliced distribution can compute the log_prob of its own
    # samples (up to numerical validation errors).
    with tfp_hps.no_tf_rank_errors():
      try:
        lp = self.evaluate(dist.log_prob(samples))
      except tf.errors.InvalidArgumentError:
        # TODO(b/129271256): d.log_prob(d.sample()) should not fail
        #     validate_args checks.
        # We only tolerate this case for the non-sliced dist.
        return
      sliced_lp = self.evaluate(sliced_dist.log_prob(sliced_samples))

    # Check that the sliced dist's log_prob agrees with slicing the original's
    # log_prob.

    # This `hp.assume` is suppressing array sizes that cause the sliced and
    # non-sliced distribution to follow different Eigen code paths.  Those
    # different code paths lead to arbitrarily large variations in the results
    # at parameter settings that Hypothesis is all too good at finding.  Since
    # the purpose of this test is just to check that we got slicing right, those
    # discrepancies are a distraction.
    # TODO(b/140229057): Remove this `hp.assume`, if and when Eigen's numerics
    # become index-independent.
    all_packetized = (
        _all_packetized(dist) and _all_packetized(sliced_dist) and
        _all_packetized(samples) and _all_packetized(sliced_samples))
    hp.note('Packetization check {}'.format(all_packetized))
    all_non_packetized = (
        _all_non_packetized(dist) and _all_non_packetized(sliced_dist) and
        _all_non_packetized(samples) and _all_non_packetized(sliced_samples))
    hp.note('Non-packetization check {}'.format(all_non_packetized))
    hp.assume(all_packetized or all_non_packetized)

    self.assertAllClose(lp[slices], sliced_lp, atol=1e-5, rtol=1e-5)

  def _run_test(self, data):
    def ok(name):
      return name not in INSTANTIABLE_BUT_NOT_SLICABLE
    dist = data.draw(dhps.distributions(enable_vars=False,
                                        eligibility_filter=ok))

    # Check that all distributions still register as non-iterable despite
    # defining __getitem__.  (Because __getitem__ magically makes an object
    # iterable for some reason.)
    with self.assertRaisesRegexp(TypeError, 'not iterable'):
      iter(dist)

    # Test slicing
    self._test_slicing(data, dist)

    # TODO(bjp): Enable sampling and log_prob checks. Currently, too many errors
    #     from out-of-domain samples.
    # self.evaluate(dist.log_prob(dist.sample(seed=test_util.test_seed())))

  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testDistributions(self, data):
    self._run_test(data)

  def disabled_testFailureCase(self):  # pylint: disable=invalid-name
    # TODO(b/140229057): This test should pass.
    dist = tfd.Chi(df=np.float32(27.744131))
    dist = tfd.TransformedDistribution(
        bijector=tfb.NormalCDF(), distribution=dist, batch_shape=[4])
    dist = tfb.Expm1()(dist)
    samps = 1.7182817 + tf.zeros_like(dist.sample(seed=test_util.test_seed()))
    self.assertAllClose(dist.log_prob(samps)[0], dist[0].log_prob(samps[0]))


# TODO(b/150161911): reconcile graph- and eager-mode handling of denormal floats
# so that we can re-enable eager mode tests.
@test_util.test_graph_mode_only
class DistributionsWorkWithAutoVectorizationTest(test_util.TestCase):

  def _test_vectorization(self, dist_name, dist):
    seed = test_util.test_seed()

    num_samples = 3
    if dist_name in SAMPLE_AUTOVECTORIZATION_IS_BROKEN:
      sample = self.evaluate(dist.sample(num_samples, seed=seed))
    else:
      sample = self.evaluate(tf.vectorized_map(
          lambda i: dist.sample(seed=seed),
          tf.range(num_samples),
          fallback_to_while_loop=False))
    hp.note('Drew samples {}'.format(sample))

    if dist_name not in LOGPROB_AUTOVECTORIZATION_IS_BROKEN:
      pfor_lp = tf.vectorized_map(
          dist.log_prob,
          tf.convert_to_tensor(sample),
          fallback_to_while_loop=False)
      batch_lp = dist.log_prob(sample)
      pfor_lp_, batch_lp_ = self.evaluate((pfor_lp, batch_lp))
      self.assertAllClose(pfor_lp_, batch_lp_,
                          atol=VECTORIZED_LOGPROB_ATOL[dist_name],
                          rtol=VECTORIZED_LOGPROB_RTOL[dist_name])

  @parameterized.named_parameters(
      {'testcase_name': dname, 'dist_name': dname}
      for dname in sorted(list(dhps.INSTANTIABLE_BASE_DISTS.keys())))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testVmap(self, dist_name, data):
    dist = data.draw(dhps.distributions(
        dist_name=dist_name, enable_vars=False,
        validate_args=False))  # TODO(b/142826246): Enable validate_args.
    self._test_vectorization(dist_name, dist)


if __name__ == '__main__':
  # Hypothesis often finds numerical near misses.  Debugging them is much aided
  # by seeing all the digits of every floating point number, instead of the
  # usual default of truncating the printed representation to 8 digits.
  np.set_printoptions(floatmode='unique', precision=None)
  tf.test.main()
