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
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.experimental.composite_tensor import _registry as clsid_registry
from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

tfb = tfp.bijectors
tfd = tfp.distributions


def normal_composite(*args, **kwargs):
  return tfp.experimental.as_composite(tfd.Normal(*args, **kwargs))


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

    d2 = onehot_cat_composite(logits=[.5, .5], validate_args=True)
    tf.nest.assert_same_structure(dist, d2, expand_composites=True)

    d3 = onehot_cat_composite(probs=var, validate_args=True)

    # pylint: disable=g-error-prone-assert-raises
    with self.assertRaisesRegexp(ValueError,
                                 'Incompatible CompositeTensor TypeSpecs'):
      tf.nest.assert_same_structure(dist, d3, expand_composites=True)
    # pylint: enable=g-error-prone-assert-raises

  def test_basics_assertfails(self):
    dist = normal_composite(0, 1, validate_args=True)
    flat = tf.nest.flatten(dist, expand_composites=True)
    flat[1] = tf.constant(-1.)
    with self.assertRaisesOpError('`scale` must be positive'):
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
        self.scale_adj = tf.Variable(0.)

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

    self.evaluate(m2.scale_adj.assign(-30.))
    tf.saved_model.save(m2, os.path.join(path, 'saved_model2'))
    m3 = tf.saved_model.load(os.path.join(path, 'saved_model2'))
    self.evaluate([v.initializer for v in (m3.loc, m3.scale_adj)])
    with self.assertRaisesOpError('Argument `scale` must be positive'):
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

    class Normal(tfd.Normal):  # Note, same name as tfd.Normal, but diff type.
      pass

    class Model(tf.Module):

      @tf.function(input_signature=())
      def make_dist(self):
        return tfp.experimental.as_composite(Normal(0, 1))

    m1 = Model()
    tf.saved_model.save(m1, os.path.join(path, 'saved_model1'))
    # Eliminate cached classes, forcing breakage on load.
    clsid_registry.clear()
    with self.assertRaisesRegexp(
        ValueError, r'For non-builtin.*call `as_composite` before'):
      tf.saved_model.load(os.path.join(path, 'saved_model1'))

    tfp.experimental.as_composite(Normal(0, 1))
    # Now warmed-up, loading should work.
    m2 = tf.saved_model.load(os.path.join(path, 'saved_model1'))
    self.evaluate(m2.make_dist().sample())

  def test_not_implemented(self):
    with self.assertRaisesRegexp(NotImplementedError,
                                 r'Unable.*sigmoidNormal.*file an issue'):
      tfp.experimental.as_composite(tfb.Sigmoid()(tfd.Normal(0, 1)))

    outcomes = tf.Variable([1., 2., 4.])
    self.evaluate(outcomes.initializer)
    # FiniteDiscrete does not include `outcomes` in the _params_event_ndims, so
    # it doesn't become part of the Tensor part of the CompositeTensor.
    with self.assertRaisesRegexp(NotImplementedError,
                                 r'FiniteDiscrete.*(Unable to serialize.)'):
      tfp.experimental.as_composite(
          tfd.FiniteDiscrete(outcomes, logits=tf.math.log([0.1, 0.4, 0.3])))

  def test_multi_calls(self):
    d = tfd.Normal(0, 1)
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


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
