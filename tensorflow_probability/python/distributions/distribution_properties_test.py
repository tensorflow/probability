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

Tests of compatibility with TF platform features are in
platform_compatibility_test.py.

Tests of compatibility with JAX transformations are in
jax_transformation_test.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import parameterized
import hypothesis as hp
from hypothesis import strategies as hps
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions import hypothesis_testlib as dhps
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


# Batch slicing requires implementing `_params_event_ndims`.  Generic
# instantiation (per `instantiable_base_dists`, below) also requires
# `_params_event_ndims`, but some special distributions can be instantiated
# without that.  Of those, this variable lists the ones that do not support
# batch slicing.
INSTANTIABLE_BUT_NOT_SLICABLE = (
    'BatchReshape',
    'Mixture',
    'QuantizedDistribution',
)


EVENT_SPACE_BIJECTOR_IS_BROKEN = [
    'InverseGamma',  # TODO(b/143090143): Enable this when the bug is fixed.
                     # (Reciprocal(Softplus(x)) -> inf for small x)
]

SLICING_LOGPROB_ATOL = collections.defaultdict(lambda: 1e-5)
SLICING_LOGPROB_ATOL.update({
    'Weibull': 3e-5,
})

SLICING_LOGPROB_RTOL = collections.defaultdict(lambda: 1e-5)
SLICING_LOGPROB_RTOL.update({
    'Weibull': 3e-5,
})


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
    if hasattr(dist, 'distribution'):
      # BatchReshape, Independent, TransformedDistribution, and
      # QuantizedDistribution
      self.check_bad_loc_scale(dist.distribution)
    if isinstance(dist, tfd.MixtureSameFamily):
      self.check_bad_loc_scale(dist.mixture_distribution)
      self.check_bad_loc_scale(dist.components_distribution)
    if isinstance(dist, tfd.Mixture):
      self.check_bad_loc_scale(dist.cat)
      self.check_bad_loc_scale(dist.components)
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

  def check_event_space_bijector_constrains(self, dist, data):
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
    self.check_bad_loc_scale(dist)
    self.check_event_space_bijector_constrains(dist, data)

  # TODO(b/146572907): Fix `enable_vars` for metadistributions and
  # fold these two tests into one.
  @parameterized.named_parameters(
      {'testcase_name': dname, 'dist_name': dname}
      for dname in sorted(list(dhps.INSTANTIABLE_META_DISTS)))
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testDistributionNoVars(self, dist_name, data):
    def ok(name):
      return name not in EVENT_SPACE_BIJECTOR_IS_BROKEN
    dist = data.draw(dhps.distributions(
        dist_name=dist_name, enable_vars=True,
        eligibility_filter=ok))
    self.evaluate([var.initializer for var in dist.variables])
    self.check_bad_loc_scale(dist)
    self.check_event_space_bijector_constrains(dist, data)


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


if __name__ == '__main__':
  # Hypothesis often finds numerical near misses.  Debugging them is much aided
  # by seeing all the digits of every floating point number, instead of the
  # usual default of truncating the printed representation to 8 digits.
  np.set_printoptions(floatmode='unique', precision=None)
  tf.test.main()
