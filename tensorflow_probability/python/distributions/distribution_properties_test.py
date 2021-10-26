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
"""Property-based testing for TFP distributions.

These are general distribution properties, like reproducible sampling; tested by
default on TF2 graph and eager mode only.

Tests pertaining to numerics are in numerical_properties_test.py.

Tests of compatibility with TF platform features are in
platform_compatibility_test.py.

Tests of compatibility with JAX transformations are in
jax_transformation_test.py.
"""

import collections

from absl.testing import parameterized
import hypothesis as hp
from hypothesis import strategies as hps
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.distributions import hypothesis_testlib as dhps
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels import hypothesis_testlib as kernel_hps


STATISTIC_CONSISTENT_SHAPES_TEST_BLOCK_LIST = (
)


# Batch slicing requires implementing `_params_event_ndims`.  Generic
# instantiation (per `instantiable_base_dists`, below) also requires
# `_params_event_ndims`, but some special distributions can be instantiated
# without that.  Of those, this variable lists the ones that do not support
# batch slicing.
INSTANTIABLE_BUT_NOT_SLICABLE = (
    'BatchBroadcast',
    'BatchReshape',
    'Sample'  # TODO(b/204210361)
)


EVENT_SPACE_BIJECTOR_IS_BROKEN = [
    'InverseGamma',  # TODO(b/143090143): Enable this when the bug is fixed.
                     # (Reciprocal(Softplus(x)) -> inf for small x)
    'PowerSpherical',  # TODO(b/182609813): Enable when we have a proper
                       # event space bijector
    'SphericalUniform',  # TODO(b/182609813): Enable when we have a proper
                         # event space bijector
    'VonMisesFisher',  # TODO(b/182609813): Enable when we have a proper
                       # event space bijector
]

SLICING_LOGPROB_ATOL = collections.defaultdict(lambda: 1e-5)
SLICING_LOGPROB_ATOL.update({
    'NormalInverseGaussian': 3e-4,  # b/193076242
    'Weibull': 3e-5,
})

SLICING_LOGPROB_RTOL = collections.defaultdict(lambda: 1e-5)
SLICING_LOGPROB_RTOL.update({
    'NormalInverseGaussian': 5e-3,  # b/193076242
    'Sample': 1e-3,
    'Slicing': 6e-5,
    'Weibull': 3e-5,
})


@test_util.test_all_tf_execution_regimes
class StatisticConsistentShapesTest(test_util.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': dname, 'dist_name': dname}
      for dname in sorted(list(dhps.INSTANTIABLE_BASE_DISTS.keys())
                          + list(dhps.INSTANTIABLE_META_DISTS)))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testDistribution(self, dist_name, data):
    if dist_name in STATISTIC_CONSISTENT_SHAPES_TEST_BLOCK_LIST:
      self.skipTest('{} is blocked'.format(dist_name))
    def eligibility_filter(name):
      return name not in STATISTIC_CONSISTENT_SHAPES_TEST_BLOCK_LIST
    dist = data.draw(dhps.distributions(
        dist_name=dist_name, eligibility_filter=eligibility_filter,
        enable_vars=False, validate_args=False))
    hp.note('Trying distribution {}'.format(
        self.evaluate_dict(dist.parameters)))

    for statistic in ['mean', 'mode', 'stddev', 'variance']:
      hp.note('Testing {}.{}'.format(dist_name, statistic))
      self.check_statistic(
          dist, statistic,
          dist.batch_shape + dist.event_shape,
          tf.concat([dist.batch_shape_tensor(),
                     dist.event_shape_tensor()], axis=0))
    hp.note('Testing {}.covariance'.format(dist_name))
    self.check_statistic(
        dist, 'covariance',
        dist.batch_shape + dist.event_shape + dist.event_shape,
        tf.concat([dist.batch_shape_tensor(),
                   dist.event_shape_tensor(),
                   dist.event_shape_tensor()], axis=0))
    hp.note('Testing {}.entropy'.format(dist_name))
    self.check_statistic(
        dist, 'entropy', dist.batch_shape, dist.batch_shape_tensor())

  def check_statistic(
      self, dist, statistic, expected_static_shape, expected_dynamic_shape):
    try:
      with tfp_hps.no_tf_rank_errors():
        result = getattr(dist, statistic)()
      msg = 'Shape {} of {} not compatible with expected {}.'.format(
          result.shape, statistic, expected_static_shape)
      self.assertTrue(
          expected_static_shape.is_compatible_with(result.shape), msg)
      self.assertAllEqual(self.evaluate(expected_dynamic_shape),
                          self.evaluate(tf.shape(result)))
    except NotImplementedError:
      pass


def _constrained_zeros_fn(shape, dtype, constraint_fn):
  """Generates dummy parameters initialized to a valid default value."""
  return hps.just(constraint_fn(tf.fill(shape, tf.cast(0., dtype))))


MLE_AT_CONSTRAINT_BOUNDARY = [
    'Uniform'
]


@test_util.test_all_tf_execution_regimes
class FittingTest(test_util.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': dname, 'dist_name': dname}
      for dname in sorted(list(dhps.INSTANTIABLE_BASE_DISTS.keys())))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testDistribution(self, dist_name, data):
    dist = data.draw(dhps.base_distributions(
        dist_name=dist_name,
        enable_vars=False,
        # Unregularized MLEs can be numerically problematic, e.g., empirical
        # (co)variances can be singular. To avoid such numerical issues, we
        # sanity-check the MLE only for a fixed sample with assumed-sane
        # parameter values (zeros constrained to the parameter support).
        param_strategy_fn=_constrained_zeros_fn,
        batch_shape=data.draw(
            tfp_hps.shapes(min_ndims=0, max_ndims=2, max_side=5))))
    x, lp = self.evaluate(dist.experimental_sample_and_log_prob(
        10, seed=test_util.test_seed(sampler_type='stateless')))

    try:
      parameters = self.evaluate(type(dist)._maximum_likelihood_parameters(x))
    except NotImplementedError:
      self.skipTest('Fitting not implemented.')

    flat_params = tf.nest.flatten(parameters)
    lp_fn = lambda *flat_params: type(dist)(  # pylint: disable=g-long-lambda
        validate_args=True,
        **tf.nest.pack_sequence_as(parameters, flat_params)).log_prob(x)
    lp_mle, grads = self.evaluate(
        tfp_math.value_and_gradient(lp_fn, flat_params))

    # Likelihood of MLE params should be higher than of the original params.
    self.assertAllGreaterEqual(
        tf.reduce_sum(lp_mle, axis=0) - tf.reduce_sum(lp, axis=0),
        -1e-4)

    if dist_name not in MLE_AT_CONSTRAINT_BOUNDARY:
      # MLE parameters should be a critical point of the log prob.
      for g in grads:
        if np.any(np.isnan(g)):
          # Skip parameters with undefined or unstable gradients (e.g.,
          # Categorical `num_classes`).
          continue
        self.assertAllClose(tf.zeros_like(g), g, atol=1e-2)


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

    seed = test_util.test_seed(sampler_type='stateless')
    with tfp_hps.no_tf_rank_errors():
      s1 = self.evaluate(dist.sample(50, seed=seed))
      s2 = self.evaluate(dist.sample(50, seed=seed))
    self.assertAllEqual(s1, s2)


@test_util.test_all_tf_execution_regimes
class SampleAndLogProbTest(test_util.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': dname, 'dist_name': dname}
      for dname in sorted(list(dhps.INSTANTIABLE_BASE_DISTS.keys()) +
                          list(dhps.INSTANTIABLE_META_DISTS)))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testDistribution(self, dist_name, data):
    dist = data.draw(dhps.distributions(dist_name=dist_name, enable_vars=False,
                                        validate_args=False))
    seed = test_util.test_seed(sampler_type='stateless')
    sample_shape = [2, 1]
    with tfp_hps.no_tf_rank_errors(), kernel_hps.no_pd_errors():
      s1, lp1 = dist.experimental_sample_and_log_prob(sample_shape, seed=seed)
      s2 = dist.sample(sample_shape, seed=seed)
      self.assertAllClose(s1, s2, atol=1e-4)

      # Sanity-check the log prob. The actual values may differ arbitrarily (if
      # the `sample_and_log_prob` implementation is more stable) or be NaN, but
      # they should at least have the same shape.
      lp2 = dist.log_prob(s1)
      self.assertAllEqual(lp1.shape, lp2.shape)


@test_util.test_all_tf_execution_regimes
class EventSpaceBijectorsTest(test_util.TestCase, dhps.TestCase):

  def check_event_space_bijector_constrains(self, dist, data):
    event_space_bijector = dist.experimental_default_event_space_bijector()
    if event_space_bijector is None:
      return

    # Draw a sample shape
    sample_shape = data.draw(tfp_hps.shapes())
    inv_event_shape = event_space_bijector.inverse_event_shape(
        tensorshape_util.concatenate(dist.batch_shape, dist.event_shape))

    # Draw a shape that broadcasts with `[batch_shape, inverse_event_shape]`
    # where `inverse_event_shape` is the event shape in the bijector's
    # domain. This is the shape of `y` in R**n, such that
    # x = event_space_bijector(y) has the event shape of the distribution.

    # TODO(b/174778703): Actually draw broadcast compatible shapes.
    batch_inv_event_compat_shape = inv_event_shape
    # batch_inv_event_compat_shape = data.draw(
    #     tfp_hps.broadcast_compatible_shape(inv_event_shape))
    # batch_inv_event_compat_shape = tensorshape_util.concatenate(
    #     (1,) * (len(inv_event_shape) - len(batch_inv_event_compat_shape)),
    #     batch_inv_event_compat_shape)

    total_sample_shape = tensorshape_util.concatenate(
        sample_shape, batch_inv_event_compat_shape)
    # full_sample_batch_event_shape = tensorshape_util.concatenate(
    #     sample_shape, inv_event_shape)

    y = data.draw(
        tfp_hps.constrained_tensors(
            tfp_hps.identity_fn, total_sample_shape.as_list()))
    hp.note('Trying to constrain inputs {}'.format(y))
    with tfp_hps.no_tf_rank_errors():
      x = event_space_bijector(y)
      hp.note('Got constrained samples {}'.format(x))
      with tf.control_dependencies(dist._sample_control_dependencies(x)):
        self.evaluate(tensor_util.identity_as_tensor(x))

      # TODO(b/158874412): Verify DoF changing default bijectors.
      # y_bc = tf.broadcast_to(y, full_sample_batch_event_shape)
      # x_bc = event_space_bijector(y_bc)
      # self.assertAllClose(x, x_bc)
      # fldj = event_space_bijector.forward_log_det_jacobian(y)
      # fldj_bc = event_space_bijector.forward_log_det_jacobian(y_bc)
      # self.assertAllClose(fldj, fldj_bc)

  @parameterized.named_parameters(
      {'testcase_name': dname, 'dist_name': dname}
      for dname in sorted(list(set(dhps.INSTANTIABLE_BASE_DISTS.keys()) -
                               set(EVENT_SPACE_BIJECTOR_IS_BROKEN))))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testDistributionWithVars(self, dist_name, data):
    dist = data.draw(dhps.base_distributions(
        dist_name=dist_name, enable_vars=True))
    self.evaluate([var.initializer for var in dist.variables])
    self.assume_loc_scale_ok(dist)
    self.check_event_space_bijector_constrains(dist, data)

  # TODO(b/146572907): Fix `enable_vars` for metadistributions and
  # fold these two tests into one.
  @parameterized.named_parameters(
      {'testcase_name': dname, 'dist_name': dname}
      for dname in sorted(list(set(dhps.INSTANTIABLE_META_DISTS) -
                               set(EVENT_SPACE_BIJECTOR_IS_BROKEN))))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testDistributionNoVars(self, dist_name, data):
    def ok(name):
      return name not in EVENT_SPACE_BIJECTOR_IS_BROKEN
    dist = data.draw(dhps.distributions(
        dist_name=dist_name, enable_vars=False,
        eligibility_filter=ok))
    self.assume_loc_scale_ok(dist)
    self.check_event_space_bijector_constrains(dist, data)


@test_util.test_all_tf_execution_regimes
class ParameterPropertiesTest(test_util.TestCase):

  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testCanConstructAndSampleDistribution(self, data):

    # TODO(b/169874884): Implement `width` parameters to work around the need
    # for a high > low` joint constraint.
    # NormalInverseGaussian needs |skewness| < tailweight
    high_gt_low_constraint_dists = ('Bates', 'NormalInverseGaussian', 'PERT',
                                    'Triangular', 'TruncatedCauchy',
                                    'TruncatedNormal', 'Uniform')
    not_annotated_dists = ('Empirical|event_ndims=0', 'Empirical|event_ndims=1',
                           'Empirical|event_ndims=2', 'FiniteDiscrete',
                           # cov_perturb_factor is not annotated since its shape
                           # could be a vector or a matrix.
                           'MultivariateNormalDiagPlusLowRankCovariance',
                           'MultivariateStudentTLinearOperator',
                           'PoissonLogNormalQuadratureCompound',
                           'StoppingRatioLogistic',)
    non_trainable_dists = (
        high_gt_low_constraint_dists + not_annotated_dists +
        dhps.INSTANTIABLE_META_DISTS)

    non_trainable_tensor_params = (
        'atol',
        'rtol',
        'eigenvectors',  # TODO(b/171872834): DeterminantalPointProcess
        'total_count',
        'num_samples',
        'df',  # Can't represent constraint that Wishart df > dimension.
        'mean_direction')  # TODO(b/118492439): Add `UnitVector` bijector.
    non_trainable_non_tensor_params = (
        'batch_shape',  # SphericalUniform, at least, has explicit batch shape
        'dimension',
        'dtype')

    dist = data.draw(
        dhps.distributions(
            eligibility_filter=(lambda name: name not in non_trainable_dists)))
    sample_shape = tuple(
        self.evaluate(
            tf.concat([dist.batch_shape_tensor(),
                       dist.event_shape_tensor()],
                      axis=0)))

    params = type(dist).parameter_properties(num_classes=2)
    params64 = type(dist).parameter_properties(dtype=tf.float64, num_classes=2)

    new_parameters = {}
    seeds = {k: s for (k, s) in zip(
        params.keys(),
        samplers.split_seed(test_util.test_seed(), n=len(params)))}
    for param_name, param in params.items():
      if param_name in non_trainable_tensor_params:
        new_parameters[param_name] = dist.parameters[param_name]
      elif param.is_preferred:
        b = param.default_constraining_bijector_fn()
        unconstrained_shape = (
            b.inverse_event_shape_tensor(
                param.shape_fn(sample_shape=sample_shape)))
        unconstrained_param = samplers.normal(
            unconstrained_shape, seed=seeds[param_name])
        new_parameters[param_name] = b.forward(unconstrained_param)

        # Check that passing a float64 `eps` works with float64 parameters.
        b_float64 = params64[param_name].default_constraining_bijector_fn()
        b_float64(tf.cast(unconstrained_param, tf.float64))

    # Copy over any non-Tensor parameters.
    new_parameters.update({
        k: v
        for (k, v) in dist.parameters.items()
        if k in non_trainable_non_tensor_params
    })

    # Sanity check that we got valid parameters.
    new_parameters['validate_args'] = True
    new_dist = type(dist)(**new_parameters)
    x = self.evaluate(new_dist.sample(seed=test_util.test_seed()))
    self.assertEqual(sample_shape, x.shape)

    # Valid parameters should give non-nan samples.
    self.assertAllFalse(np.isnan(x))

  @parameterized.named_parameters(
      {'testcase_name': dname, 'dist_name': dname}
      for dname in sorted(list(dhps.INSTANTIABLE_BASE_DISTS.keys()) +
                          list(dhps.INSTANTIABLE_META_DISTS)))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testInferredBatchShapeMatchesTrueBatchShape(self, dist_name, data):
    with tfp_hps.no_cholesky_decomposition_errors():
      dist = data.draw(
          dhps.distributions(dist_name=dist_name, validate_args=False))
      with tfp_hps.no_tf_rank_errors():
        lp = dist.log_prob(dist.sample(seed=test_util.test_seed()))

    self.assertAllEqual(dist.batch_shape_tensor(), tf.shape(lp))
    self.assertAllEqual(dist.batch_shape, tf.shape(lp))


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


DISTS_OK_TO_SLICE = (set(list(dhps.INSTANTIABLE_BASE_DISTS.keys()) +
                         list(dhps.INSTANTIABLE_META_DISTS)) -
                     set(INSTANTIABLE_BUT_NOT_SLICABLE))


@test_util.test_all_tf_execution_regimes
class DistributionSlicingTest(test_util.TestCase):

  def _test_slicing(self, data, dist_name, dist):
    strm = test_util.test_seed_stream()
    batch_shape = dist.batch_shape
    slices = data.draw(tfp_hps.valid_slices(batch_shape))
    slice_str = 'dist[{}]'.format(', '.join(tfp_hps.stringify_slices(
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

    # Check that the sliced dist's log_prob agrees with slicing the original's
    # log_prob.
    # First, we make sure that the original sample we have passes the
    # original distribution's validations.  We break the bijector cache here
    # because slicing will break it later too.
    with tfp_hps.no_tf_rank_errors():
      try:
        lp = self.evaluate(dist.log_prob(
            samples + tf.constant(0, dtype=samples.dtype)))
      except tf.errors.InvalidArgumentError:
        # TODO(b/129271256): d.log_prob(d.sample()) should not fail
        #     validate_args checks.
        # `return` here passes the example.  If we `hp.assume(False)`
        # instead, that would demand from Hypothesis that it find many
        # examples where this check (and the next one) passes;
        # empirically, it seems to complain that that's too hard.
        return

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

    # Actually evaluate and test the sliced log_prob
    with tfp_hps.no_tf_rank_errors():
      sliced_lp = self.evaluate(sliced_dist.log_prob(sliced_samples))

    self.assertAllClose(lp[slices], sliced_lp,
                        atol=SLICING_LOGPROB_ATOL[dist_name],
                        rtol=SLICING_LOGPROB_RTOL[dist_name])

  @parameterized.named_parameters(
      {'testcase_name': dname, 'dist_name': dname}
      for dname in sorted(DISTS_OK_TO_SLICE))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testDistributions(self, dist_name, data):
    def ok(name):
      return name not in INSTANTIABLE_BUT_NOT_SLICABLE

    dist = data.draw(dhps.distributions(dist_name=dist_name,
                                        enable_vars=False,
                                        eligibility_filter=ok))

    # Check that all distributions still register as non-iterable despite
    # defining __getitem__.  (Because __getitem__ magically makes an object
    # iterable for some reason.)
    with self.assertRaisesRegexp(TypeError, 'not iterable'):
      iter(dist)

    # Test slicing
    self._test_slicing(data, dist_name, dist)

    # TODO(bjp): Enable sampling and log_prob checks. Currently, too many errors
    #     from out-of-domain samples.
    # self.evaluate(dist.log_prob(dist.sample(seed=test_util.test_seed())))

  def disabled_testFailureCase(self):  # pylint: disable=invalid-name
    # TODO(b/140229057): This test should pass.
    dist = tfd.Chi(df=np.float32(27.744131) * np.ones((4,)).astype(np.float32))
    dist = tfd.TransformedDistribution(
        bijector=tfb.NormalCDF(), distribution=dist)
    dist = tfb.Expm1()(dist)
    samps = 1.7182817 + tf.zeros_like(dist.sample(seed=test_util.test_seed()))
    self.assertAllClose(dist.log_prob(samps)[0], dist[0].log_prob(samps[0]))


# Don't decorate with test_util.test_all_tf_execution_regimes, since we're
# explicitly mixing modes.
class TestMixingGraphAndEagerModes(test_util.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': dname, 'dist_name': dname}
      for dname in  sorted(list(dhps.INSTANTIABLE_BASE_DISTS.keys()) +
                           list(dhps.INSTANTIABLE_META_DISTS))
  )
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testSampleEagerCreatedDistributionInGraphMode(self, dist_name, data):
    if not tf.executing_eagerly():
      self.skipTest('Only test mixed eager/graph behavior in eager tests.')
    # Create in eager mode.
    dist = data.draw(dhps.distributions(dist_name=dist_name, enable_vars=False))

    @tf.function
    def f():
      dist.sample()
    f()


if __name__ == '__main__':
  # Hypothesis often finds numerical near misses.  Debugging them is much aided
  # by seeing all the digits of every floating point number, instead of the
  # usual default of truncating the printed representation to 8 digits.
  np.set_printoptions(floatmode='unique', precision=None)
  test_util.main()
