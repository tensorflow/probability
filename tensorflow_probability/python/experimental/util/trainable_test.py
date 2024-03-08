# Copyright 2021 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
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
"""Tests for trainable distributions and bijectors."""

from absl.testing import parameterized

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import fill_scale_tril
from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.bijectors import scale
from tensorflow_probability.python.bijectors import scale_matvec_tril
from tensorflow_probability.python.bijectors import softplus
from tensorflow_probability.python.distributions import dirichlet
from tensorflow_probability.python.distributions import gamma
from tensorflow_probability.python.distributions import geometric
from tensorflow_probability.python.distributions import kumaraswamy
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import poisson
from tensorflow_probability.python.distributions import wishart
from tensorflow_probability.python.experimental.util import trainable
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.internal import tf_keras
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.math.minimize import minimize
from tensorflow_probability.python.math.minimize import minimize_stateless

JAX_MODE = False


@test_util.test_graph_and_eager_modes
class TestStatefulMakeTrainable(test_util.TestCase):

  @parameterized.named_parameters(
      ('Normal', normal.Normal, []), ('Gamma', gamma.Gamma, [2, 3]),
      ('Dirichlet', dirichlet.Dirichlet, [5]),
      ('MultivariateNormalTriL', mvn_tril.MultivariateNormalTriL, [4]),
      ('Kumaraswamy', kumaraswamy.Kumaraswamy, [2, 1]),
      ('Poisson', poisson.Poisson, []),
      ('Geometric', geometric.Geometric, [1, 1]))
  @test_util.jax_disable_variable_test
  def test_trainable_distributions(self, cls, batch_and_event_shape):
    distribution = trainable.make_trainable(
        cls,
        batch_and_event_shape=batch_and_event_shape,
        seed=test_util.test_seed(),
        validate_args=True)
    self.evaluate([v.initializer for v in distribution.trainable_variables])

    x = self.evaluate(distribution.sample())
    self.assertAllEqual(x.shape, batch_and_event_shape)

    # Verify expected number of trainable variables.
    self.assertLen(distribution.trainable_variables,
                   len([k for k, p in cls.parameter_properties().items()
                        if p.is_preferred]))

    # Verify gradients to all parameters.
    with tf.GradientTape() as tape:
      lp = distribution.log_prob(x)
    grad = tape.gradient(lp, distribution.trainable_variables)
    self.assertAllNotNone(grad)

  @parameterized.named_parameters(
      ('Scale', scale.Scale, [2]),
      ('ScaleMatvecTriL', scale_matvec_tril.ScaleMatvecTriL, [4]),
      ('FillScaleTriL', fill_scale_tril.FillScaleTriL, [2, 6]),
      ('Softplus', softplus.Softplus, [2]), ('Identity', identity.Identity, []))
  @test_util.jax_disable_variable_test
  def test_trainable_bijectors(self, cls, batch_and_event_shape):
    bijector = trainable.make_trainable(
        cls,
        batch_and_event_shape=batch_and_event_shape,
        seed=test_util.test_seed(),
        validate_args=True)
    if bijector.trainable_variables:
      self.evaluate([v.initializer for v in bijector.trainable_variables])

    # Verify expected number of trainable variables.
    self.assertLen(bijector.trainable_variables,
                   len([k for k, p in cls.parameter_properties().items()
                        if p.is_tensor and p.is_preferred]))

    # Verify gradients to all parameters.
    x = self.evaluate(samplers.normal(batch_and_event_shape,
                                      seed=test_util.test_seed()))
    with tf.GradientTape() as tape:
      y = bijector.forward(x)
    grad = tape.gradient(y, bijector.trainable_variables)
    self.assertAllNotNone(grad)

    # Verify that the round trip doesn't broadcast, i.e., that it preserves
    # batch_and_event_shape.
    self.assertAllCloseNested(
        x,
        bijector.inverse(tf.identity(y)),  # Disable bijector cache.
        atol=1e-2)

  @test_util.jax_disable_variable_test
  def test_can_specify_initial_values(self):
    distribution = trainable.make_trainable(
        normal.Normal,
        initial_parameters={'scale': 1e-4},
        batch_and_event_shape=[3],
        seed=test_util.test_seed(),
        validate_args=True)
    self.evaluate([v.initializer for v in distribution.trainable_variables])
    self.assertAllClose(
        # Convert to tensor since we can't eval a TransformedVariable directly.
        tf.convert_to_tensor(distribution.scale),
        [1e-4, 1e-4, 1e-4])

  @test_util.jax_disable_variable_test
  def test_can_specify_callable_initializer(self):

    def uniform_initializer(_, shape, dtype, seed,
                            constraining_bijector):
      return constraining_bijector.forward(
          tf.random.stateless_uniform(
              constraining_bijector.inverse_event_shape_tensor(shape),
              dtype=dtype, seed=seed))

    distribution = trainable.make_trainable(
        normal.Normal,
        initial_parameters=uniform_initializer,
        batch_and_event_shape=[3, 4],
        seed=test_util.test_seed(),
        validate_args=True)
    for v in distribution.trainable_variables:
      self.evaluate(v.initializer)
      # Unconstrained variables should be uniform in [0, 1].
      self.assertAllGreater(v, 0.)
      self.assertAllLess(v, 1.)

  @test_util.jax_disable_variable_test
  def test_initialization_is_deterministic_with_seed(self):
    seed = test_util.test_seed(sampler_type='stateless')
    distribution1 = trainable.make_trainable(
        normal.Normal, seed=seed, validate_args=True)
    self.evaluate([v.initializer for v in distribution1.trainable_variables])

    distribution2 = trainable.make_trainable(
        normal.Normal, seed=seed, validate_args=True)
    self.evaluate([v.initializer for v in distribution2.trainable_variables])

    self.assertAllEqual(distribution1.mean(), distribution2.mean())
    self.assertAllEqual(distribution1.stddev(), distribution2.stddev())

  @test_util.jax_disable_variable_test
  def test_can_specify_parameter_dtype(self):
    distribution = trainable.make_trainable(
        normal.Normal,
        initial_parameters={'loc': 17.},
        parameter_dtype=tf.float64,
        seed=test_util.test_seed(sampler_type='stateless'),
        validate_args=True)
    self.evaluate([v.initializer for v in distribution.trainable_variables])
    self.assertEqual(distribution.loc.dtype, tf.float64)
    self.assertEqual(distribution.scale.dtype, tf.float64)
    self.assertEqual(distribution.sample().dtype, tf.float64)

  @test_util.jax_disable_variable_test
  def test_can_specify_fixed_values(self):
    distribution = trainable.make_trainable(
        wishart.WishartTriL,
        batch_and_event_shape=[2, 2],
        seed=test_util.test_seed(sampler_type='stateless'),
        validate_args=True,
        df=3)
    self.evaluate([v.initializer for v in distribution.trainable_variables])
    self.assertAllClose(distribution.df, 3.)
    self.assertLen(distribution.trainable_variables, 1)
    self.assertAllEqual(distribution.sample().shape, [2, 2])

  @test_util.jax_disable_variable_test
  def test_docstring(self):
    self.assertContainsExactSubsequence(trainable.make_trainable.__doc__,
                                        'lambda: -model.log_prob(samples)')

  @test_util.jax_disable_variable_test
  def test_docstring_example_normal(self):
    samples = [4.57, 6.37, 5.93, 7.98, 2.03, 3.59, 8.55, 3.45, 5.06, 6.44]
    model = trainable.make_trainable(
        normal.Normal, seed=test_util.test_seed(sampler_type='stateless'))
    losses = minimize(
        lambda: -model.log_prob(samples),
        optimizer=tf_keras.optimizers.Adam(0.1),
        num_steps=200)
    self.evaluate(tf1.global_variables_initializer())
    self.evaluate(losses)
    self.assertAllClose(tf.reduce_mean(samples), model.mean(), atol=2.0)
    self.assertAllClose(tf.math.reduce_std(samples), model.stddev(), atol=2.0)


@test_util.test_graph_and_eager_modes
class TestStatelessTrainableDistributionsAndBijectors(test_util.TestCase):

  @parameterized.named_parameters(
      ('Normal', normal.Normal, []), ('Gamma', gamma.Gamma, [2, 3]),
      ('Dirichlet', dirichlet.Dirichlet, [5]),
      ('MultivariateNormalTriL', mvn_tril.MultivariateNormalTriL, [4]),
      ('Kumaraswamy', kumaraswamy.Kumaraswamy, [2, 1]),
      ('Poisson', poisson.Poisson, []),
      ('Geometric', geometric.Geometric, [1, 1]))
  def test_trainable_distributions(self, cls, batch_and_event_shape):
    init_fn, apply_fn = trainable.make_trainable_stateless(
        cls, batch_and_event_shape=batch_and_event_shape, validate_args=True)

    raw_parameters = init_fn(seed=test_util.test_seed())
    distribution = apply_fn(raw_parameters)
    x = self.evaluate(distribution.sample(seed=test_util.test_seed()))
    self.assertAllEqual(x.shape, batch_and_event_shape)

    # Verify expected number of trainable variables.
    self.assertLen(
        raw_parameters,
        len([k for k, p in distribution.parameter_properties().items()
             if p.is_preferred]))
    # Verify gradients to all parameters.
    _, grad = gradient.value_and_gradient(
        lambda params: apply_fn(params).log_prob(x), [raw_parameters])
    self.assertAllNotNone(grad)

  @parameterized.named_parameters(
      ('Scale', scale.Scale, [2]),
      ('ScaleMatvecTriL', scale_matvec_tril.ScaleMatvecTriL, [4]),
      ('FillScaleTriL', fill_scale_tril.FillScaleTriL, [2, 6]),
      ('Softplus', softplus.Softplus, [2]), ('Identity', identity.Identity, []))
  def test_trainable_bijectors(self, cls, batch_and_event_shape):
    init_fn, apply_fn = trainable.make_trainable_stateless(
        cls, batch_and_event_shape=batch_and_event_shape, validate_args=True)

    # Verify expected number of trainable variables.
    raw_parameters = init_fn(seed=test_util.test_seed())
    bijector = apply_fn(raw_parameters)
    self.assertLen(raw_parameters, len(
        [k for k, p in bijector.parameter_properties().items()
         if p.is_tensor and p.is_preferred]))
    # Verify gradients to all parameters.
    x = self.evaluate(samplers.normal(batch_and_event_shape,
                                      seed=test_util.test_seed()))
    y, grad = gradient.value_and_gradient(
        lambda params: apply_fn(params).forward(x), [raw_parameters])
    self.assertAllNotNone(grad)

    # Verify that the round trip doesn't broadcast, i.e., that it preserves
    # batch_and_event_shape.
    self.assertAllCloseNested(
        x,
        bijector.inverse(tf.identity(y)),  # Disable bijector cache.
        atol=1e-2)

  def test_can_specify_initial_values(self):
    init_fn, apply_fn = trainable.make_trainable_stateless(
        normal.Normal,
        initial_parameters={'scale': 1e-4},
        batch_and_event_shape=[3],
        validate_args=True)
    raw_parameters = init_fn(seed=test_util.test_seed())
    self.assertAllClose(apply_fn(raw_parameters).scale,
                        [1e-4, 1e-4, 1e-4])

  def test_can_specify_callable_initializer(self):
    def uniform_initializer(_, shape, dtype, seed, constraining_bijector):
      return constraining_bijector.forward(
          tf.random.stateless_uniform(
              constraining_bijector.inverse_event_shape_tensor(shape),
              dtype=dtype, seed=seed))

    init_fn, _ = trainable.make_trainable_stateless(
        normal.Normal,
        initial_parameters=uniform_initializer,
        batch_and_event_shape=[3, 4],
        validate_args=True)
    raw_parameters = init_fn(test_util.test_seed())
    for v in tf.nest.flatten(raw_parameters):
      # Unconstrained parameters should be uniform in [0, 1].
      self.assertAllGreater(v, 0.)
      self.assertAllLess(v, 1.)

  def test_initialization_is_deterministic_with_seed(self):
    seed = test_util.test_seed(sampler_type='stateless')
    init_fn, _ = trainable.make_trainable_stateless(
        normal.Normal, validate_args=True)
    result1 = init_fn(seed=seed)
    seed = test_util.clone_seed(seed)
    result2 = init_fn(seed=seed)
    self.assertAllCloseNested(result1, result2)

  def test_can_specify_parameter_dtype(self):
    init_fn, apply_fn = trainable.make_trainable_stateless(
        normal.Normal,
        initial_parameters={'loc': 17.},
        parameter_dtype=tf.float64,
        validate_args=True)
    distribution = apply_fn(init_fn(seed=test_util.test_seed()))
    self.assertEqual(distribution.loc.dtype, tf.float64)
    self.assertEqual(distribution.scale.dtype, tf.float64)
    self.assertEqual(distribution.sample(seed=test_util.test_seed()).dtype,
                     tf.float64)

  def test_can_specify_fixed_values(self):
    init_fn, apply_fn = trainable.make_trainable_stateless(
        wishart.WishartTriL,
        batch_and_event_shape=[2, 2],
        validate_args=True,
        df=3)
    raw_parameters = init_fn(seed=test_util.test_seed())
    self.assertLen(raw_parameters, 1)
    distribution = apply_fn(raw_parameters)
    self.assertAllClose(distribution.df, 3.)
    self.assertAllEqual(distribution.sample(seed=test_util.test_seed()).shape,
                        [2, 2])

  def test_dynamic_shape(self):
    batch_and_event_shape = tf1.placeholder_with_default(
        [4, 3, 2], shape=None)
    init_fn, apply_fn = trainable.make_trainable_stateless(
        normal.Normal,
        batch_and_event_shape=batch_and_event_shape,
        validate_args=True)
    distribution = apply_fn(init_fn(seed=test_util.test_seed()))
    x = self.evaluate(distribution.sample(seed=test_util.test_seed()))
    self.assertAllEqual(x.shape, batch_and_event_shape)

  def test_docstring(self):
    self.assertContainsExactSubsequence(
        trainable.make_trainable_stateless.__doc__,
        'lambda *params: -apply_fn(params).log_prob(samples)')

  def test_docstring_example_normal(self):
    if not JAX_MODE:
      self.skipTest('Stateless minimization requires optax.')
    import optax  # pylint: disable=g-import-not-at-top

    samples = [4.57, 6.37, 5.93, 7.98, 2.03, 3.59, 8.55, 3.45, 5.06, 6.44]
    init_fn, apply_fn = trainable.make_trainable_stateless(normal.Normal)
    final_params, losses = minimize_stateless(
        lambda *params: -apply_fn(params).log_prob(samples),
        init=init_fn(seed=test_util.test_seed(sampler_type='stateless')),
        optimizer=optax.adam(0.1),
        num_steps=200)
    model = apply_fn(final_params)
    self.evaluate(losses)
    self.assertAllClose(tf.reduce_mean(samples), model.mean(), atol=2.0)
    self.assertAllClose(tf.math.reduce_std(samples), model.stddev(), atol=2.0)


if __name__ == '__main__':
  test_util.main()
