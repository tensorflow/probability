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
"""Tests for composite tensor conversion routines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports
import numpy as np
import six
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.experimental.composite_tensor import _registry as clsid_registry
from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

tfb = tfp.bijectors
tfd = tfp.distributions


def normal_composite(*args, **kwargs):
  return tfp.experimental.as_composite(tfd.Normal(*args, **kwargs))


def sigmoid_normal_composite(*args, **kwargs):
  return tfp.experimental.as_composite(tfb.Sigmoid()(tfd.Normal(
      *args, **kwargs)))


def onehot_cat_composite(*args, **kwargs):
  return tfp.experimental.as_composite(tfd.OneHotCategorical(*args, **kwargs))


@test_util.run_all_in_graph_and_eager_modes
class CompositeTensorTest(tfp_test_util.TestCase):

  def test_basics(self):
    dist = normal_composite(0, 1, validate_args=True)
    flat = tf.nest.flatten(dist, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(dist, flat, expand_composites=True)
    self.evaluate(unflat.sample())
    self.evaluate(unflat.log_prob(.5))

    d2 = normal_composite(
        loc=dist.sample(seed=tfp_test_util.test_seed()),
        scale=1,
        validate_args=True)
    tf.nest.assert_same_structure(dist, d2, expand_composites=True)

  def test_basics_var(self):
    loc = tf.Variable(0.)
    self.evaluate(loc.initializer)
    dist = normal_composite(loc, 1, validate_args=True)
    self.evaluate([v.initializer for v in dist.variables])
    flat = tf.nest.flatten(dist, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(dist, flat, expand_composites=True)
    self.evaluate(unflat.sample())
    self.evaluate(unflat.log_prob(.5))

  def test_basics_mutex_params(self):
    var = tf.Variable([.9, .1])
    self.evaluate(var.initializer)
    dist = onehot_cat_composite(logits=var, validate_args=True)

    flat = tf.nest.flatten(dist, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(dist, flat, expand_composites=True)
    self.evaluate(unflat.sample())

    d2 = onehot_cat_composite(logits=tf.Variable([.5, .5]), validate_args=True)
    tf.nest.assert_same_structure(dist, d2, expand_composites=True)

    d3 = onehot_cat_composite(probs=var, validate_args=True)

    # pylint: disable=g-error-prone-assert-raises
    with self.assertRaisesRegexp(ValueError,
                                 'Incompatible CompositeTensor TypeSpecs'):
      tf.nest.assert_same_structure(dist, d3, expand_composites=True)
    # pylint: enable=g-error-prone-assert-raises

  def test_basics_assertfails(self):
    dist = normal_composite(0., 1., validate_args=True)
    flat = tf.nest.flatten(dist, expand_composites=True)
    flat[1] = tf.constant(2)
    with self.assertRaises(TypeError):
      unflat = tf.nest.pack_sequence_as(dist, flat, expand_composites=True)
      self.evaluate(unflat.log_prob(.5))

  def test_tf_function(self):

    @tf.function
    def make_dist():
      return normal_composite(
          0., tf.random.uniform([]) + .1, validate_args=True)

    self.evaluate(make_dist().sample())
    self.evaluate(make_dist().log_prob(.25))

    @tf.function
    def take_dist(d):
      return d.sample(), d.log_prob(.25)

    dist = normal_composite(0, 1, validate_args=True)
    self.evaluate(take_dist(dist))

  def test_while_loop(self):
    d_init = normal_composite(loc=0, scale=1, validate_args=True)
    d_final, = tf.while_loop(
        cond=lambda _: True,
        body=lambda d: [  # pylint: disable=g-long-lambda
            normal_composite(
                loc=d.sample(seed=tfp_test_util.test_seed()),
                scale=1,
                validate_args=True)
        ],
        loop_vars=[d_init],
        maximum_iterations=20,
        parallel_iterations=1)
    self.evaluate(d_final.sample())
    self.evaluate(d_final.log_prob(.25))

  def test_export_import(self):
    path = self.create_tempdir().full_path

    class Model(tf.Module):

      def __init__(self):
        self.loc = tf.Variable([0., 1.])
        self.scale_adj = tf.Variable([0.], shape=[None])

      @tf.function(input_signature=(normal_composite(0, [1, 2])._type_spec,))
      def make_dist(self, d):
        return normal_composite(
            tf.convert_to_tensor(self.loc),
            self.scale_adj + d.prob(.1),
            validate_args=True)

    m1 = Model()
    self.evaluate([v.initializer for v in m1.variables])
    self.evaluate(m1.loc.assign(m1.loc + 1.))

    tf.saved_model.save(m1, os.path.join(path, 'saved_model1'))
    m2 = tf.saved_model.load(os.path.join(path, 'saved_model1'))
    self.evaluate([v.initializer for v in (m2.loc, m2.scale_adj)])
    d = normal_composite(.3, [.5, .9])
    self.evaluate(m2.make_dist(d).sample())
    self.evaluate(m2.loc.assign(m2.loc + 2))
    self.evaluate(m2.make_dist(d).sample())

    self.evaluate(m2.scale_adj.assign([1., 2., 3.]))
    tf.saved_model.save(m2, os.path.join(path, 'saved_model2'))
    m3 = tf.saved_model.load(os.path.join(path, 'saved_model2'))
    self.evaluate([v.initializer for v in (m3.loc, m3.scale_adj)])
    with self.assertRaisesOpError('compatible shape'):
      self.evaluate(m3.make_dist(d).sample())

  def test_import_uncached_class(self):
    path = self.create_tempdir().full_path

    class Model(tf.Module):

      @tf.function(input_signature=(normal_composite(0, [1, 2])._type_spec,))
      def make_dist(self, d):
        return normal_composite(d.sample(), 1, validate_args=True)

    m1 = Model()
    tf.saved_model.save(m1, os.path.join(path, 'saved_model1'))
    # Eliminate cached classes, forcing auto-regen of class on load.
    clsid_registry.clear()
    m2 = tf.saved_model.load(os.path.join(path, 'saved_model1'))
    d = normal_composite(.3, [.5, .9])
    self.evaluate(m2.make_dist(d).sample())

  def test_import_unrecognized_class(self):
    path = self.create_tempdir().full_path

    class Normal(tfd.Distribution):  # Same name as tfd.Normal, but diff type.

      def __init__(self, loc, scale):
        self.dist = tfd.Normal(loc, scale)
        super(Normal, self).__init__(
            dtype=None, reparameterization_type=None,
            validate_args=False, allow_nan_stats=False)

      def _sample_n(self, n, seed=None, **kwargs):
        return self.dist._sample_n(n, seed=seed, **kwargs)  # pylint: disable=protected-access

    class Model(tf.Module):

      @tf.function(input_signature=())
      def make_dist(self):
        return tfp.experimental.as_composite(Normal(0, 1))

    m1 = Model()
    tf.saved_model.save(m1, os.path.join(path, 'saved_model1'))
    # Eliminate cached classes, forcing breakage on load.
    clsid_registry.clear()
    with self.assertRaisesRegexp(
        ValueError, r'For user-defined.*decorated.*register_composite'):
      tf.saved_model.load(os.path.join(path, 'saved_model1'))

    tfp.experimental.as_composite(Normal(0, 1))
    # Now warmed-up, loading should work.
    m2 = tf.saved_model.load(os.path.join(path, 'saved_model1'))
    self.evaluate(m2.make_dist().sample())

    # Eliminate cached classes again, but now register Normal as if it had been
    # decorated from the beginning.
    clsid_registry.clear()
    self.assertEqual(Normal, tfp.experimental.register_composite(Normal))

    # Loading should work again.
    m3 = tf.saved_model.load(os.path.join(path, 'saved_model1'))
    self.evaluate(m3.make_dist().sample())

  def test_sigmoid_normal(self):
    if six.PY2:
      self.skipTest(
          'PY3-only test because we do not support the callable argument '
          'kwargs_split_fn of TransformedDistribution in PY2.')
    sn = tfb.Sigmoid()(tfd.Normal(0, 1))
    dist = tfp.experimental.as_composite(sn)
    flat = tf.nest.flatten(dist, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(dist, flat, expand_composites=True)
    self.evaluate(unflat.sample())
    self.evaluate(unflat.log_prob(.5))

  def test_sigmoid_normal_save_load(self):
    if six.PY2:
      self.skipTest(
          'PY3-only test because we do not support the callable argument '
          'kwargs_split_fn of TransformedDistribution in PY2.')
    path = self.create_tempdir().full_path

    class Model(tf.Module):

      @tf.function(
          input_signature=(sigmoid_normal_composite(loc=0,
                                                    scale=[1,
                                                           2])._type_spec,))
      def make_dist(self, d):
        return sigmoid_normal_composite(d.sample(), 1, validate_args=True)

    m1 = Model()
    tf.saved_model.save(m1, os.path.join(path, 'saved_model1'))
    m2 = tf.saved_model.load(os.path.join(path, 'saved_model1'))
    d = sigmoid_normal_composite(.3, [.5, .9])
    self.evaluate(m2.make_dist(d).sample())

  def test_sigmoid_normal_with_params(self):
    if six.PY2:
      self.skipTest(
          'PY3-only test because we do not support the callable argument '
          'kwargs_split_fn of TransformedDistribution in PY2.')
    sn = tfb.Sigmoid(
        low=[2.0, 3.0], high=[4.0, 5.0])(
            tfd.Normal([6.0, 7.0], 1))
    dist = tfp.experimental.as_composite(sn)
    flat = tf.nest.flatten(dist, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(dist, flat, expand_composites=True)
    self.evaluate(unflat.sample())
    self.evaluate(unflat.log_prob(.5))

  def test_finite_discrete(self):
    outcomes = tf.Variable([1., 2., 4.])
    self.evaluate(outcomes.initializer)
    fd = tfd.FiniteDiscrete(outcomes, logits=tf.math.log([0.1, 0.4, 0.3]))
    log_prob_before = self.evaluate(fd.log_prob(2.))
    dist = tfp.experimental.as_composite(fd)
    flat = tf.nest.flatten(dist, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(dist, flat, expand_composites=True)
    self.evaluate(unflat.sample())
    log_prob_after = self.evaluate(unflat.log_prob(2.))
    self.assertEqual(log_prob_before, log_prob_after)

  def test_multivariate_normal_linear_operator(self):
    linop = tf.linalg.LinearOperatorIdentity(2)
    d = tfd.MultivariateNormalLinearOperator(scale=linop)
    sample = [-2.0, 3.0]
    log_prob_before = self.evaluate(d.log_prob(sample))
    dist = tfp.experimental.as_composite(d)
    flat = tf.nest.flatten(dist, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(dist, flat, expand_composites=True)
    self.evaluate(unflat.sample())
    log_prob_after = self.evaluate(unflat.log_prob(sample))
    self.assertAllEqual(log_prob_before, log_prob_after)

  def test_multivariate_normal_linear_operator_diag(self):
    linop = tf.linalg.LinearOperatorDiag([5.0, -6.0])
    d = tfd.MultivariateNormalLinearOperator(scale=linop)
    sample = [-2.0, 3.0]
    log_prob_before = self.evaluate(d.log_prob(sample))
    dist = tfp.experimental.as_composite(d)
    flat = tf.nest.flatten(dist, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(dist, flat, expand_composites=True)
    self.evaluate(unflat.sample())
    log_prob_after = self.evaluate(unflat.log_prob(sample))
    self.assertAllEqual(log_prob_before, log_prob_after)

  def test_multivariate_normal_low_rank_update(self):
    diag_operator = tf.linalg.LinearOperatorDiag([1., 2., 3.],
                                                 is_non_singular=True,
                                                 is_self_adjoint=True,
                                                 is_positive_definite=True)
    operator = tf.linalg.LinearOperatorLowRankUpdate(
        base_operator=diag_operator,
        u=[[1., 2.], [-1., 3.], [0., 0.]],
        diag_update=[11., 12.],
        v=[[1., 2.], [-1., 3.], [10., 10.]])
    d = tfd.MultivariateNormalLinearOperator(scale=operator)
    sample = [-2.0, 3.0, -4.0]
    log_prob_before = self.evaluate(d.log_prob(sample))
    dist = tfp.experimental.as_composite(d)
    flat = tf.nest.flatten(dist, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(dist, flat, expand_composites=True)
    self.evaluate(unflat.sample())
    log_prob_after = self.evaluate(unflat.log_prob(sample))
    self.assertAllEqual(log_prob_before, log_prob_after)

  def test_multivariate_normal_linear_operator_inversion(self):
    operator = tf.linalg.LinearOperatorFullMatrix([[1., -2.], [-3., 4.]])
    operator_inv = tf.linalg.LinearOperatorInversion(operator)
    d = tfd.MultivariateNormalLinearOperator(scale=operator_inv)
    sample = [-2.0, 3.0]
    log_prob_before = self.evaluate(d.log_prob(sample))
    dist = tfp.experimental.as_composite(d)
    flat = tf.nest.flatten(dist, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(dist, flat, expand_composites=True)
    self.evaluate(unflat.sample())
    log_prob_after = self.evaluate(unflat.log_prob(sample))
    self.assertAllEqual(log_prob_before, log_prob_after)

  def test_multivariate_normal_tril(self):
    mu = [1., 2, 3]
    cov = [[0.36, 0.12, 0.06], [0.12, 0.29, -0.13], [0.06, -0.13, 0.26]]
    scale = tf.linalg.cholesky(cov)
    d = tfd.MultivariateNormalTriL(loc=mu, scale_tril=scale)
    sample = [-2.0, 3.0, -4.0]
    log_prob_before = self.evaluate(d.log_prob(sample))
    dist = tfp.experimental.as_composite(d)
    flat = tf.nest.flatten(dist, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(dist, flat, expand_composites=True)
    self.evaluate(unflat.sample())
    log_prob_after = self.evaluate(unflat.log_prob(sample))
    self.assertAllEqual(log_prob_before, log_prob_after)

  def test_independent(self):
    fd = tfd.Independent(
        distribution=tfd.Normal(loc=[-1., 1], scale=[0.1, 0.5]),
        reinterpreted_batch_ndims=1)
    sample = [-2.0, 3.0]
    log_prob_before = self.evaluate(fd.log_prob(sample))
    dist = tfp.experimental.as_composite(fd)
    flat = tf.nest.flatten(dist, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(dist, flat, expand_composites=True)
    self.evaluate(unflat.sample())
    log_prob_after = self.evaluate(unflat.log_prob(sample))
    self.assertEqual(log_prob_before, log_prob_after)

  def test_shift_bijector(self):
    d = tfd.Normal([0., 1.], [2., 3.])
    bij = tfb.Shift(4.)
    td = tfd.TransformedDistribution(distribution=d, bijector=bij)
    sample = [-2.0, 3.0]
    log_prob_before = self.evaluate(td.log_prob(sample))
    dist = tfp.experimental.as_composite(td)
    flat = tf.nest.flatten(dist, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(dist, flat, expand_composites=True)
    self.evaluate(unflat.sample())
    log_prob_after = self.evaluate(unflat.log_prob(sample))
    self.assertAllEqual(log_prob_before, log_prob_after)

  def test_chain_bijector(self):
    d = tfd.Normal([1., 2.], [3., 4.])
    bij = tfb.Chain([tfb.Shift(5.), tfb.Scale(6.)])
    td = tfd.TransformedDistribution(distribution=d, bijector=bij)
    sample = [[7., 8.], [9., -1.]]
    log_prob_before = self.evaluate(td.log_prob(sample))
    dist = tfp.experimental.as_composite(td)
    flat = tf.nest.flatten(dist, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(dist, flat, expand_composites=True)
    self.evaluate(unflat.sample())
    log_prob_after = self.evaluate(unflat.log_prob(sample))
    self.assertAllEqual(log_prob_before, log_prob_after)

  def test_keras_layers(self):
    # 10-vector to 5-vector
    def layer_helper(x):
      loc = tf.split(x, 2, axis=-1)[0]
      scale = tf.math.exp(tf.split(x, 2, axis=-1)[1])
      d = tfd.Normal(loc, scale)
      cd = tfp.experimental.as_composite(d)
      return cd

    model1 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10),
        tf.keras.layers.Lambda(layer_helper),
        tf.keras.layers.Lambda(tfd.Distribution.mean),
        tf.keras.layers.Dense(10),
    ])
    model2 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10),
        tfp.layers.DistributionLambda(layer_helper),
        tf.keras.layers.Dense(10),
    ])

    model1.compile(optimizer='sgd', loss='mean_squared_error')
    model2.compile(optimizer='sgd', loss='mean_squared_error')
    xs = np.expand_dims(np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]), axis=-1)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
    model1.fit(xs, ys, epochs=500)
    model2.fit(xs, ys, epochs=500, steps_per_epoch=5)
    x_test = np.expand_dims(np.array([10.0]), axis=-1)
    model1.predict(x_test, steps=1)
    model2.predict(x_test, steps=1)

  def test_transformed_distribution(self):
    fd = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0., scale=1.),
        bijector=tfb.Exp())
    sample = 2.
    log_prob_before = self.evaluate(fd.log_prob(sample))
    dist = tfp.experimental.as_composite(fd)
    flat = tf.nest.flatten(dist, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(dist, flat, expand_composites=True)
    self.evaluate(unflat.sample())
    log_prob_after = self.evaluate(unflat.log_prob(sample))
    self.assertEqual(log_prob_before, log_prob_after)

  def test_multi_calls(self):

    class TrivialMetaDist(tfd.Distribution):

      def __init__(self, dist):
        self.dist = dist
        super(TrivialMetaDist, self).__init__(
            dtype=None, reparameterization_type=None,
            validate_args=False, allow_nan_stats=False)

    n = tfd.Normal(0, 1)
    d = TrivialMetaDist(n)
    d1 = tfp.experimental.as_composite(d)
    d2 = tfp.experimental.as_composite(d1)
    self.assertIsNot(d, d1)
    self.assertIs(d1, d2)

  def test_basics_mixture_same_family(self):
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[0.3, 0.7]),
        components_distribution=tfd.Normal(
            loc=[-1., 1],
            scale=[0.1, 0.5]))
    dist = tfp.experimental.as_composite(gm)
    flat = tf.nest.flatten(dist, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(dist, flat, expand_composites=True)
    self.evaluate(unflat.sample())
    self.evaluate(unflat.log_prob(.5))

  def test_already_composite_tensor(self):
    b = tfb.Scale(2.)
    b2 = tfp.experimental.as_composite(b)
    self.assertIsInstance(b, tf.__internal__.CompositeTensor)
    self.assertIs(b, b2)


if __name__ == '__main__':
  tf.test.main()
