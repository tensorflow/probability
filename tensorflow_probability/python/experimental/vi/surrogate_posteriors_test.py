# Copyright 2019 The TensorFlow Probability Authors.
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
"""Tests for surrogate posteriors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import test_util

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

tfb = tfp.bijectors
tfd = tfp.distributions


class _SurrogatePosterior(object):
  """Common methods for testing ADVI surrogate posteriors."""

  def _test_shapes(self, surrogate_posterior, batch_shape, event_shape, seed):

    posterior_sample = surrogate_posterior.sample(seed=seed)

    # Test that the posterior has the specified event shape(s).
    posterior_event_shape = self.evaluate(
        surrogate_posterior.event_shape_tensor())
    event_shape = nest.map_structure_up_to(
        surrogate_posterior.event_shape_tensor(),
        ps.convert_to_shape_tensor,
        event_shape,
        check_types=False)
    self.assertAllEqualNested(event_shape, posterior_event_shape)

    # Test that all sample Tensors have the expected shapes.
    sample_shape = nest.map_structure(
        lambda s: ps.concat([batch_shape, s], axis=0), event_shape)
    self.assertAllEqualNested(
        nest.map_structure(ps.shape, posterior_sample),
        sample_shape)

    # Test that the posterior components have the specified batch shape, which
    # is also the shape of `log_prob` of a sample.
    self.assertAllAssertsNested(
        lambda s: self.assertAllEqual(s, batch_shape),
        self.evaluate(surrogate_posterior.batch_shape_tensor()))
    self.assertAllEqual(
        batch_shape, ps.shape(surrogate_posterior.log_prob(posterior_sample)))

  def _test_gradients(self, surrogate_posterior, seed):

    # Test that gradients are available wrt the variational parameters.
    self.assertNotEmpty(surrogate_posterior.trainable_variables)

    posterior_sample = self.evaluate(surrogate_posterior.sample(seed=seed))
    with tf.GradientTape() as tape:
      posterior_logprob = surrogate_posterior.log_prob(posterior_sample)
    grad = tape.gradient(posterior_logprob,
                         surrogate_posterior.trainable_variables)
    self.assertTrue(all(g is not None for g in grad))

  def _test_dtype(self, surrogate_posterior, dtype, seed):
    self.assertAllAssertsNested(
        lambda t: self.assertDTypeEqual(t, dtype),
        surrogate_posterior.sample(seed=seed))

  def _test_fitting(self, model, surrogate_posterior):

    # Fit model.
    y = [0.2, 0.5, 0.3, 0.7]
    losses = tfp.vi.fit_surrogate_posterior(
        lambda rate, concentration: model.log_prob((rate, concentration, y)),
        surrogate_posterior,
        num_steps=5,  # Don't optimize to completion.
        optimizer=tf.optimizers.Adam(0.1),
        sample_size=10)

    # Compute posterior statistics.
    with tf.control_dependencies([losses]):
      posterior_samples = surrogate_posterior.sample(100)
      posterior_mean = [tf.reduce_mean(x) for x in posterior_samples]
      posterior_stddev = [tf.math.reduce_std(x) for x in posterior_samples]

    self.evaluate(tf1.global_variables_initializer())
    self.evaluate([losses, posterior_mean, posterior_stddev])

  def _make_gamma_model(self):
    Root = tfd.JointDistributionCoroutine.Root  # pylint: disable=invalid-name

    def model_fn():
      concentration = yield Root(tfd.Exponential(1.))
      rate = yield Root(tfd.Exponential(1.))
      y = yield tfd.Sample(  # pylint: disable=unused-variable
          tfd.Gamma(concentration=concentration, rate=rate),
          sample_shape=4)
    return tfd.JointDistributionCoroutine(model_fn)


@test_util.test_all_tf_execution_regimes
class FactoredSurrogatePosterior(test_util.TestCase, _SurrogatePosterior):

  @parameterized.named_parameters(
      {'testcase_name': 'TensorEvent',
       'event_shape': tf.TensorShape([4]),
       'bijector': tfb.Sigmoid(),
       'dtype': np.float64,
       'is_static': True},
      {'testcase_name': 'ListEvent',
       'event_shape': [tf.TensorShape([3]),
                       tf.TensorShape([]),
                       tf.TensorShape([2, 2])],
       'bijector': [tfb.Softplus(), None, tfb.FillTriangular()],
       'dtype': np.float32,
       'is_static': False},
      {'testcase_name': 'DictEvent',
       'event_shape': {'x': tf.TensorShape([1]), 'y': tf.TensorShape([])},
       'bijector': None,
       'dtype': np.float64,
       'is_static': True},
      {'testcase_name': 'NestedEvent',
       'event_shape': {'x': [tf.TensorShape([1]), tf.TensorShape([1, 2])],
                       'y': tf.TensorShape([])},
       'bijector': {
           'x': [tfb.Identity(), tfb.Softplus()], 'y': tfb.Sigmoid()},
       'dtype': np.float32,
       'is_static': True},
  )
  def test_specifying_event_shape(
      self, event_shape, bijector, dtype, is_static):
    seed = test_util.test_seed_stream()
    surrogate_posterior = (
        tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape=tf.nest.map_structure(
                lambda s: self.maybe_static(  # pylint: disable=g-long-lambda
                    np.array(s, dtype=np.int32), is_static=is_static),
                event_shape),
            bijector=bijector,
            dtype=dtype,
            seed=seed(),
            validate_args=True))
    self.evaluate([v.initializer
                   for v in surrogate_posterior.trainable_variables])

    self._test_shapes(
        surrogate_posterior, batch_shape=[], event_shape=event_shape,
        seed=seed())
    self._test_gradients(surrogate_posterior, seed=seed())
    self._test_dtype(surrogate_posterior, dtype, seed())

  @parameterized.named_parameters(
      {'testcase_name': 'TensorEvent',
       'event_shape': [4],
       'initial_loc': np.array([[[0.9, 0.1, 0.5, 0.7]]]),
       'batch_shape': [1, 1],
       'bijector': tfb.Sigmoid(),
       'dtype': np.float32,
       'is_static': False},
      {'testcase_name': 'ListEvent',
       'event_shape': [[3], [], [2, 2]],
       'initial_loc': [np.array([0.1, 7., 3.]),
                       0.1,
                       np.array([[1., 0], [-4., 2.]])],
       'batch_shape': [],
       'bijector': [tfb.Softplus(), None, tfb.FillTriangular()],
       'dtype': np.float64,
       'is_static': True},
      {'testcase_name': 'DictEvent',
       'event_shape': {'x': [2], 'y': []},
       'initial_loc': {'x': np.array([[0.9, 1.2]]),
                       'y': np.array([-4.1])},
       'batch_shape': [1],
       'bijector': None,
       'dtype': np.float32,
       'is_static': False},
      {'testcase_name': 'ExplicitBatchShape',
       'event_shape': [[3], [4]],
       'initial_loc': [0., 0.],
       'batch_shape': [5, 1],
       'bijector': None,
       'dtype': np.float32,
       'is_static': False},
  )
  def test_specifying_initial_loc(
      self, event_shape, initial_loc, batch_shape, bijector, dtype, is_static):
    initial_loc = tf.nest.map_structure(
        lambda s: self.maybe_static(  # pylint: disable=g-long-lambda
            np.array(s, dtype=dtype), is_static=is_static),
        initial_loc)

    if bijector is not None:
      initial_unconstrained_loc = tf.nest.map_structure(
          lambda x, b: x if b is None else b.inverse(x),
          initial_loc, bijector)
    else:
      initial_unconstrained_loc = initial_loc

    surrogate_posterior = (
        tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape=event_shape,
            initial_parameters=tf.nest.map_structure(
                lambda x: {'loc': x, 'scale': 1e-6},
                initial_unconstrained_loc),
            batch_shape=batch_shape,
            dtype=dtype,
            bijector=bijector,
            validate_args=True))
    self.evaluate([v.initializer
                   for v in surrogate_posterior.trainable_variables])

    seed = test_util.test_seed_stream()
    self._test_shapes(
        surrogate_posterior, batch_shape=batch_shape,
        event_shape=event_shape, seed=seed())
    self._test_gradients(surrogate_posterior, seed=seed())
    self._test_dtype(surrogate_posterior, dtype, seed())

    # Check that the sampled values are close to the initial locs.
    posterior_sample_ = self.evaluate(surrogate_posterior.sample(seed=seed()))
    self.assertAllCloseNested(
        tf.nest.map_structure(lambda x, y: tf.broadcast_to(x, ps.shape(y)),
                              initial_loc, posterior_sample_),
        posterior_sample_,
        atol=1e-4)

  @parameterized.named_parameters(
      {'testcase_name': 'TensorEventAllDeterministic',
       'event_shape': [4],
       'base_distribution_cls': tfd.Deterministic},
      {'testcase_name': 'ListEventSingleDeterministic',
       'event_shape': [[3], [], [2, 2]],
       'base_distribution_cls': tfd.Deterministic},
      {'testcase_name': 'ListEventDeterministicNormalStudentT',
       'event_shape': [[3], [], [2, 2]],
       'base_distribution_cls': [tfd.Deterministic, tfd.Normal, tfd.StudentT]},
  )
  def test_specifying_distribution_type(
      self, event_shape, base_distribution_cls):
    surrogate_posterior = (
        tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape=event_shape,
            base_distribution_cls=base_distribution_cls,
            validate_args=True))

    # Test that the surrogate uses the expected distribution types.
    if tf.nest.is_nested(surrogate_posterior.event_shape):
      ds, _ = surrogate_posterior.sample_distributions()
    else:
      ds = [surrogate_posterior]
    for cls, d in zip(
        nest_util.broadcast_structure(ds, base_distribution_cls), ds):
      while isinstance(d, tfd.Independent):
        d = d.distribution
      self.assertIsInstance(d, cls)

  def test_that_gamma_fitting_example_runs(self):
    model = self._make_gamma_model()
    surrogate_posterior = (
        tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape=model.event_shape_tensor()[:-1],
            bijector=[tfb.Softplus(), tfb.Softplus()]))
    self._test_fitting(model, surrogate_posterior)

  def test_multipart_bijector(self):
    dist = tfd.JointDistributionNamed({
        'a': tfd.Exponential(1.),
        'b': tfd.Normal(0., 1.),
        'c': lambda b, a: tfd.Sample(tfd.Normal(b, a), sample_shape=[5])})

    surrogate_posterior = (
        tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape=dist.event_shape,
            bijector=(
                dist.experimental_default_event_space_bijector()),
            validate_args=True))
    self.evaluate([v.initializer
                   for v in surrogate_posterior.trainable_variables])

    # Test that the posterior has the specified event shape(s).
    self.assertAllEqualNested(
        self.evaluate(dist.event_shape_tensor()),
        self.evaluate(surrogate_posterior.event_shape_tensor()))

    posterior_sample_ = self.evaluate(surrogate_posterior.sample(
        seed=test_util.test_seed()))
    posterior_logprob_ = self.evaluate(
        surrogate_posterior.log_prob(posterior_sample_))

    # Test that all sample Tensors have the expected shapes.
    check_shape = lambda s, x: self.assertAllEqual(s, x.shape)
    self.assertAllAssertsNested(
        check_shape, dist.event_shape, posterior_sample_)

    # Test that samples are finite and not NaN.
    self.assertAllAssertsNested(self.assertAllFinite, posterior_sample_)

    # Test that logprob is scalar, finite, and not NaN.
    self.assertEmpty(posterior_logprob_.shape)
    self.assertAllFinite(posterior_logprob_)


@test_util.test_all_tf_execution_regimes
class AffineSurrogatePosterior(test_util.TestCase, _SurrogatePosterior):

  @parameterized.named_parameters(
      {'testcase_name': 'TensorEvent',
       'dist_classes': tfd.Normal,
       'event_shape': [4],
       'operators': [tf.linalg.LinearOperatorLowerTriangular],
       'initial_loc': np.array([[[0.9, 0.1, 0.5, 0.7]]]),
       'implicit_batch_shape': [1, 1],
       'bijector': tfb.Sigmoid(),
       'dtype': np.float32,
       'is_static': False},
      {'testcase_name': 'ListEvent',
       'dist_classes': [tfd.Laplace, tfd.Normal, tfd.Logistic],
       'event_shape': [[3], [], [2, 2]],
       'operators': [tf.linalg.LinearOperatorDiag,
                     tf.linalg.LinearOperatorDiag,
                     tf.linalg.LinearOperatorLowerTriangular],
       'initial_loc': [np.array([0.1, 7., 3.]),
                       0.1,
                       np.array([[1., 0], [-4., 2.]])],
       'implicit_batch_shape': [],
       'bijector': [tfb.FillTriangular(), None, tfb.Softplus()],
       'dtype': np.float64,
       'is_static': True},
      {'testcase_name': 'DictEvent',
       'dist_classes': {'x': tfd.Logistic, 'y': tfd.Normal},
       'event_shape': {'x': [2], 'y': []},
       'operators': 'tril',
       'initial_loc': {'x': np.array([[0.9, 1.2]]),
                       'y': np.array([-4.1])},
       'implicit_batch_shape': [1],
       'bijector': None,
       'dtype': np.float32,
       'is_static': False},
  )
  def test_constrained_affine_from_distributions(
      self, dist_classes, event_shape, operators, initial_loc,
      implicit_batch_shape, bijector, dtype, is_static):
    if not tf.executing_eagerly() and not is_static:
      self.skipTest('tfb.Reshape requires statically known shapes in graph'
                    ' mode.')
    # pylint: disable=g-long-lambda
    initial_loc = tf.nest.map_structure(
        lambda s: self.maybe_static(np.array(s, dtype=dtype),
                                    is_static=is_static),
        initial_loc)
    distributions = nest.map_structure_up_to(
        dist_classes,
        lambda d, loc, s: tfd.Independent(
            d(loc=loc, scale=1.),
            reinterpreted_batch_ndims=ps.rank_from_shape(s)),
        dist_classes, initial_loc, event_shape)
    # pylint: enable=g-long-lambda
    surrogate_posterior = (
        tfp.experimental.vi.
        build_affine_surrogate_posterior_from_base_distribution(
            distributions,
            operators=operators,
            bijector=bijector,
            validate_args=True))

    event_shape = nest.map_structure(
        lambda d: d.event_shape_tensor(), distributions)
    if bijector is not None:
      event_shape = nest.map_structure(
          lambda b, s: s if b is None else b.forward_event_shape_tensor(s),
          bijector, event_shape)

    self.evaluate(
        [v.initializer for v in surrogate_posterior.trainable_variables])

    seed = test_util.test_seed_stream()
    self._test_shapes(
        surrogate_posterior, batch_shape=implicit_batch_shape,
        event_shape=event_shape, seed=seed())
    self._test_gradients(surrogate_posterior, seed=seed())
    self._test_dtype(surrogate_posterior, dtype, seed())

  @parameterized.named_parameters(
      {'testcase_name': 'TensorEvent',
       'event_shape': tf.TensorShape([3]),
       'operators': [tf.linalg.LinearOperatorDiag],
       'batch_shape': (),
       'bijector': tfb.Exp(),
       'dtype': np.float32,
       'is_static': True},
      {'testcase_name': 'ListEvent',
       'event_shape': [tf.TensorShape([3]),
                       tf.TensorShape([]),
                       tf.TensorShape([2, 2])],
       'operators': 'diag',
       'batch_shape': (),
       'bijector': [tfb.Softplus(), None, tfb.FillTriangular()],
       'dtype': np.float32,
       'is_static': False},
      {'testcase_name': 'DictEvent',
       'event_shape': {'x': tf.TensorShape([3]), 'y': tf.TensorShape([])},
       'operators': [[tf.linalg.LinearOperatorDiag],
                     [tf.linalg.LinearOperatorFullMatrix,
                      tf.linalg.LinearOperatorLowerTriangular]],
       'batch_shape': (),
       'bijector': None,
       'dtype': np.float64,
       'is_static': True},
      {'testcase_name': 'BatchShape',
       'event_shape': [tf.TensorShape([3]), tf.TensorShape([])],
       'operators': 'tril',
       'batch_shape': (2, 1),
       'bijector': [tfb.Softplus(), None,],
       'dtype': np.float32,
       'is_static': True},
      {'testcase_name': 'DynamicBatchShape',
       'event_shape': [tf.TensorShape([3]), tf.TensorShape([])],
       'operators': 'tril',
       'batch_shape': (2, 1),
       'bijector': [tfb.Softplus(), None,],
       'dtype': np.float32,
       'is_static': False},
  )
  def test_constrained_affine_from_event_shape(
      self, event_shape, operators, bijector, batch_shape, dtype, is_static):
    if not is_static:
      event_shape = tf.nest.map_structure(
          lambda s: tf1.placeholder_with_default(  # pylint: disable=g-long-lambda
              np.array(s, dtype=np.int32),
              # Reshape bijector needs to know event_ndims statically.
              shape=[len(s)]),
          event_shape)

    surrogate_posterior = (
        tfp.experimental.vi.
        build_affine_surrogate_posterior(
            event_shape=event_shape,
            operators=operators,
            bijector=bijector,
            batch_shape=self.maybe_static(np.int32(batch_shape),
                                          is_static=is_static),
            dtype=dtype,
            validate_args=True))

    self.evaluate(
        [v.initializer for v in surrogate_posterior.trainable_variables])

    seed = test_util.test_seed_stream()
    self._test_shapes(
        surrogate_posterior, batch_shape=batch_shape, event_shape=event_shape,
        seed=seed())
    self._test_gradients(surrogate_posterior, seed=seed())
    self._test_dtype(surrogate_posterior, dtype, seed())

  def test_constrained_affine_from_joint_inputs(self):
    base_distribution = tfd.JointDistributionSequential(
        [tfd.Sample(tfd.Normal(0., 1.), sample_shape=[3, 2]),
         tfd.Logistic(0., 1.),
         tfd.Sample(tfd.Normal(0., 1.), sample_shape=[4])],
        validate_args=True)
    operator = tf.linalg.LinearOperatorBlockDiag(
        operators=[
            tf.linalg.LinearOperatorScaledIdentity(
                6, multiplier=tf.Variable(1.)),
            tf.linalg.LinearOperatorDiag(tf.Variable([1.])),
            tf.linalg.LinearOperatorDiag(tf.Variable(tf.ones([4])))],
        is_non_singular=True)
    bijector = tfb.JointMap([tfb.Exp(), tfb.Identity(), tfb.Softplus()],
                            validate_args=True)
    surrogate_posterior = (
        tfp.experimental.vi.
        build_affine_surrogate_posterior_from_base_distribution(
            base_distribution,
            operators=operator,
            bijector=bijector,
            validate_args=True))
    posterior_sample = surrogate_posterior.sample(seed=test_util.test_seed())
    posterior_logprob = surrogate_posterior.log_prob(posterior_sample)
    self.evaluate(
        [v.initializer for v in surrogate_posterior.trainable_variables])
    sample_, log_prob_ = self.evaluate([posterior_sample, posterior_logprob])
    self.assertEmpty(ps.shape(log_prob_))
    self.assertAllFinite(log_prob_)
    self.assertAllEqualNested(
        self.evaluate(base_distribution.event_shape_tensor()),
        nest.map_structure(ps.shape, sample_))

  def test_mixed_dtypes_raises(self):
    base_distribution = {'a': tfd.Normal(tf.zeros([], dtype=tf.float32), 1.),
                         'b': tfd.Logistic(tf.zeros([], dtype=tf.float64), 1.)}
    operators = [tf.linalg.LinearOperatorDiag] * 2
    with self.assertRaisesRegexp(NotImplementedError, 'mixed dtype'):
      (tfp.experimental.vi.
       build_affine_surrogate_posterior_from_base_distribution(
           base_distribution, operators=operators, validate_args=True))

  def test_deterministic_initialization_from_seed(self):
    initial_samples = []
    for _ in range(2):
      surrogate_posterior = (
          tfp.experimental.vi.build_affine_surrogate_posterior(
              event_shape={'x': tf.TensorShape([3]), 'y': tf.TensorShape([])},
              operators='tril',
              bijector=None,
              dtype=tf.float32,
              seed=test_util.test_seed(sampler_type='stateless'),
              validate_args=True))
      self.evaluate(
          [v.initializer for v in surrogate_posterior.trainable_variables])
      initial_samples.append(
          surrogate_posterior.sample(
              [5], seed=test_util.test_seed(sampler_type='stateless')))
    self.assertAllEqualNested(initial_samples[0], initial_samples[1])

  def test_that_gamma_fitting_example_runs(self):
    model = self._make_gamma_model()
    surrogate_posterior = (
        tfp.experimental.
        vi.build_affine_surrogate_posterior(
            model.event_shape_tensor()[:-1],
            operators=[
                [tf.linalg.LinearOperatorLowerTriangular],
                [None, tf.linalg.LinearOperatorDiag]],
            bijector=[tfb.Softplus(), tfb.Softplus()],
            base_distribution=tfd.Logistic,
            validate_args=True))
    self._test_fitting(model, surrogate_posterior)


@test_util.test_all_tf_execution_regimes
class SplitFlowSurrogatePosterior(
    test_util.TestCase, _SurrogatePosterior):

  @parameterized.named_parameters(
      {'testcase_name': 'TensorEvent',
       'event_shape': [6],
       'constraining_bijector': tfb.Softplus(),
       'batch_shape': [2],
       'dtype': np.float32,
       'is_static': True},
      {'testcase_name': 'ListEvent',
       'event_shape': [tf.TensorShape([3]),
                       tf.TensorShape([]),
                       tf.TensorShape([2, 2])],
       'constraining_bijector': [tfb.Softplus(), None, tfb.FillTriangular()],
       'batch_shape': tf.TensorShape([2, 2]),
       'dtype': np.float32,
       'is_static': False},
      {'testcase_name': 'DictEvent',
       'event_shape': {'x': tf.TensorShape([3]), 'y': tf.TensorShape([])},
       'constraining_bijector': None,
       'batch_shape': tf.TensorShape([]),
       'dtype': np.float64,
       'is_static': True},
  )
  def test_shapes_and_gradients(
      self, event_shape, constraining_bijector, batch_shape, dtype, is_static):
    if not tf.executing_eagerly() and not is_static:
      self.skipTest('tfb.Reshape requires statically known shapes in graph'
                    ' mode.')
    net = tfb.AutoregressiveNetwork(2, hidden_units=[4, 4], dtype=dtype)
    maf = tfb.MaskedAutoregressiveFlow(
        shift_and_log_scale_fn=net, validate_args=True)

    seed = test_util.test_seed_stream()
    surrogate_posterior = (
        tfp.experimental.vi.build_split_flow_surrogate_posterior(
            event_shape=tf.nest.map_structure(
                lambda s: self.maybe_static(  # pylint: disable=g-long-lambda
                    np.array(s, dtype=np.int32), is_static=is_static),
                event_shape),
            constraining_bijector=constraining_bijector,
            batch_shape=batch_shape,
            trainable_bijector=maf,
            dtype=dtype,
            validate_args=True))

    # Add an op to the graph so tf.Variables get created in graph mode.
    _ = surrogate_posterior.sample(seed=seed())
    self.evaluate(
        [v.initializer for v in surrogate_posterior.trainable_variables])

    self._test_shapes(
        surrogate_posterior, batch_shape=batch_shape, event_shape=event_shape,
        seed=seed())
    self._test_gradients(surrogate_posterior, seed=seed())
    self._test_dtype(surrogate_posterior, dtype, seed())

if __name__ == '__main__':
  test_util.main()
