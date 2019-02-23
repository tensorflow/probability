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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# Dependency imports
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

tfk = tf.keras
tfkl = tf.keras.layers
tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers


def _logit_avg_expit(t):
  """Computes `logit(mean(expit(t)))` in a numerically stable manner."""
  log_avg_prob = (
      tf.reduce_logsumexp(input_tensor=-tf.nn.softplus(-t), axis=0) -
      tf.math.log(tf.cast(tf.shape(input=t)[0], t.dtype)))
  return log_avg_prob - tf.math.log1p(-tf.exp(log_avg_prob))


def _vec_pad(x, value=0):
  """Prepends a column of zeros to a matrix."""
  paddings = tf.concat(
      [tf.zeros([tf.rank(x) - 1, 2], dtype=tf.int32), [[1, 0]]], axis=0)
  return tf.pad(tensor=x, paddings=paddings, constant_values=value)


@test_util.run_all_in_graph_and_eager_modes
class EndToEndTest(tf.test.TestCase):
  """Test tfp.layers work in all three Keras APIs.

  For end-to-end tests we fit a Variational Autoencoder (VAE) because this
  requires chaining two Keras models, an encoder and decoder. Chaining two
  models is important because making a `Distribution` as output by a Keras model
  the input of another Keras model--and concurrently fitting both--is the
  primary value-add of using the `tfp.layers.DistributionLambda`. Otherwise,
  under many circumstances you can directly return a Distribution from a Keras
  layer, as long as the Distribution base class has a tensor conversion function
  registered via `tf.register_tensor_conversion_function`.

  Fundamentally, there are three ways to be Keras models:
  1. `tf.keras.Sequential`
  2. Functional API
  3. Subclass `tf.keras.Model`.

  Its important to have end-to-end tests for all three, because #1 and #2 call
  `__call__` and `call` differently. (#3's call pattern depends on user
  implementation details, but in general ends up being either #1 or #2.)
  """

  def setUp(self):
    self.encoded_size = 2
    self.input_shape = [2, 2, 1]
    self.train_size = 100
    self.test_size = 100
    self.x = np.random.rand(
        self.train_size, *self.input_shape).astype(np.float32)
    self.x_test = np.random.rand(
        self.test_size, *self.input_shape).astype(np.float32)

  # TODO(b/120307671): Once this bug is resolved, use
  # `activity_regularizer=tfpl.KLDivergenceRegularizer` instead of
  # `KLDivergenceAddLoss`.

  def test_keras_sequential_api(self):
    """Test `DistributionLambda`s are composable via Keras `Sequential` API."""

    encoder_model = tfk.Sequential([
        tfkl.InputLayer(input_shape=self.input_shape),
        tfkl.Flatten(),
        tfkl.Dense(10, activation='relu'),
        tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(self.encoded_size)),
        tfpl.MultivariateNormalTriL(self.encoded_size),
        tfpl.KLDivergenceAddLoss(
            tfd.Independent(tfd.Normal(loc=[0., 0], scale=1),
                            reinterpreted_batch_ndims=1),
            weight=self.train_size),
    ])

    decoder_model = tfk.Sequential([
        tfkl.InputLayer(input_shape=[self.encoded_size]),
        tfkl.Dense(10, activation='relu'),
        tfkl.Dense(tfpl.IndependentBernoulli.params_size(self.input_shape)),
        tfpl.IndependentBernoulli(self.input_shape, tfd.Bernoulli.logits),
    ])

    vae_model = tfk.Model(
        inputs=encoder_model.inputs,
        outputs=decoder_model(encoder_model.outputs[0]))
    vae_model.compile(
        optimizer=tf.compat.v1.train.AdamOptimizer(),
        loss=lambda x, rv_x: -rv_x.log_prob(x),
        metrics=[])
    vae_model.fit(self.x, self.x,
                  batch_size=25,
                  epochs=1,
                  verbose=True,
                  validation_data=(self.x_test, self.x_test),
                  shuffle=True)
    yhat = vae_model(tf.convert_to_tensor(value=self.x_test))
    self.assertIsInstance(yhat, tfd.Independent)
    self.assertIsInstance(yhat.distribution, tfd.Bernoulli)

  def test_keras_functional_api(self):
    """Test `DistributionLambda`s are composable via Keras functional API."""

    encoder_model = [
        tfkl.Flatten(),
        tfkl.Dense(10, activation='relu'),
        tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(
            self.encoded_size)),
        tfpl.MultivariateNormalTriL(self.encoded_size),
        tfpl.KLDivergenceAddLoss(
            tfd.Independent(tfd.Normal(loc=[0., 0], scale=1),
                            reinterpreted_batch_ndims=1),
            weight=self.train_size),
    ]

    decoder_model = [
        tfkl.Dense(10, activation='relu'),
        tfkl.Dense(tfpl.IndependentBernoulli.params_size(self.input_shape)),
        tfpl.IndependentBernoulli(self.input_shape, tfd.Bernoulli.logits),
    ]

    images = tfkl.Input(shape=self.input_shape)
    encoded = functools.reduce(lambda x, f: f(x), encoder_model, images)
    decoded = functools.reduce(lambda x, f: f(x), decoder_model, encoded)

    vae_model = tfk.Model(inputs=images, outputs=decoded)
    vae_model.compile(
        optimizer=tf.compat.v1.train.AdamOptimizer(),
        loss=lambda x, rv_x: -rv_x.log_prob(x),
        metrics=[])
    vae_model.fit(self.x, self.x,
                  batch_size=25,
                  epochs=1,
                  verbose=True,
                  validation_data=(self.x_test, self.x_test),
                  shuffle=True)
    yhat = vae_model(tf.convert_to_tensor(value=self.x_test))
    self.assertIsInstance(yhat, tfd.Independent)
    self.assertIsInstance(yhat.distribution, tfd.Bernoulli)

  def test_keras_model_api(self):
    """Test `DistributionLambda`s are composable via Keras `Model` API."""

    class Encoder(tfk.Model):
      """Encoder."""

      def __init__(self, input_shape, encoded_size, train_size):
        super(Encoder, self).__init__()
        self._layers = [
            tfkl.Flatten(),
            tfkl.Dense(10, activation='relu'),
            tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size)),
            tfpl.MultivariateNormalTriL(encoded_size),
            tfpl.KLDivergenceAddLoss(
                tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                                reinterpreted_batch_ndims=1),
                weight=train_size),
        ]

      def call(self, inputs):
        return functools.reduce(lambda x, f: f(x), self._layers, inputs)

    class Decoder(tfk.Model):
      """Decoder."""

      def __init__(self, output_shape):
        super(Decoder, self).__init__()
        self._layers = [
            tfkl.Dense(10, activation='relu'),
            tfkl.Dense(tfpl.IndependentBernoulli.params_size(output_shape)),
            tfpl.IndependentBernoulli(output_shape, tfd.Bernoulli.logits),
        ]

      def call(self, inputs):
        return functools.reduce(lambda x, f: f(x), self._layers, inputs)

    encoder = Encoder(self.input_shape, self.encoded_size, self.train_size)
    decoder = Decoder(self.input_shape)

    images = tfkl.Input(shape=self.input_shape)
    encoded = encoder(images)
    decoded = decoder(encoded)

    vae_model = tfk.Model(inputs=images, outputs=decoded)
    vae_model.compile(
        optimizer=tf.compat.v1.train.AdamOptimizer(),
        loss=lambda x, rv_x: -rv_x.log_prob(x),
        metrics=[])
    vae_model.fit(self.x, self.x,
                  batch_size=25,
                  epochs=1,
                  validation_data=(self.x_test, self.x_test))
    yhat = vae_model(tf.convert_to_tensor(value=self.x_test))
    self.assertIsInstance(yhat, tfd.Independent)
    self.assertIsInstance(yhat.distribution, tfd.Bernoulli)

  def test_keras_sequential_api_multiple_draws(self):
    num_draws = 2

    encoder_model = tfk.Sequential([
        tfkl.InputLayer(input_shape=self.input_shape),
        tfkl.Flatten(),
        tfkl.Dense(10, activation='relu'),
        tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(self.encoded_size)),
        tfpl.MultivariateNormalTriL(self.encoded_size,
                                    lambda s: s.sample(num_draws, seed=42)),
        tfpl.KLDivergenceAddLoss(
            # TODO(b/119756336): Due to eager/graph Jacobian graph caching bug
            # we add here the capability for deferred construction of the prior.
            lambda: tfd.MultivariateNormalDiag(loc=tf.zeros(self.encoded_size)),
            weight=self.train_size),
    ])

    decoder_model = tfk.Sequential([
        tfkl.InputLayer(input_shape=[self.encoded_size]),
        tfkl.Dense(10, activation='relu'),
        tfkl.Dense(tfpl.IndependentBernoulli.params_size(
            self.input_shape)),
        tfkl.Lambda(_logit_avg_expit),  # Same as averaging the Bernoullis.
        tfpl.IndependentBernoulli(self.input_shape, tfd.Bernoulli.logits),
    ])

    vae_model = tfk.Model(
        inputs=encoder_model.inputs,
        outputs=decoder_model(encoder_model.outputs[0]))
    vae_model.compile(
        optimizer=tf.compat.v1.train.AdamOptimizer(),
        loss=lambda x, rv_x: -rv_x.log_prob(x),
        metrics=[])
    vae_model.fit(self.x, self.x,
                  batch_size=25,
                  epochs=1,
                  steps_per_epoch=1,  # Usually `n // batch_size`.
                  validation_data=(self.x_test, self.x_test))
    yhat = vae_model(tf.convert_to_tensor(value=self.x_test))
    self.assertIsInstance(yhat, tfd.Independent)
    self.assertIsInstance(yhat.distribution, tfd.Bernoulli)


@test_util.run_all_in_graph_and_eager_modes
class KLDivergenceAddLoss(tf.test.TestCase):

  def test_approx_kl(self):
    # TODO(b/120320323): Enable this test in eager.
    if tf.executing_eagerly(): return

    event_size = 2
    prior = tfd.MultivariateNormalDiag(loc=tf.zeros(event_size))

    model = tfk.Sequential([
        tfpl.MultivariateNormalTriL(event_size,
                                    lambda s: s.sample(int(1e3), seed=42)),
        tfpl.KLDivergenceAddLoss(prior, test_points_reduce_axis=0),
    ])

    loc = [-1., 1.]
    scale_tril = [[1.1, 0.],
                  [0.2, 1.3]]
    actual_kl = tfd.kl_divergence(
        tfd.MultivariateNormalTriL(loc, scale_tril), prior)

    x = tf.concat(
        [loc, tfb.ScaleTriL().inverse(scale_tril)], axis=0)[tf.newaxis]

    y = model(x)
    self.assertEqual(1, len(model.losses))
    y = model(x)
    self.assertEqual(2, len(model.losses))

    [loc_, scale_tril_, actual_kl_, approx_kl_] = self.evaluate([
        y.loc, y.scale.to_dense(), actual_kl, model.losses[0]])
    self.assertAllClose([loc], loc_, atol=0., rtol=1e-5)
    self.assertAllClose([scale_tril], scale_tril_, atol=0., rtol=1e-5)
    self.assertNear(actual_kl_, approx_kl_, err=0.15)

    model.compile(
        optimizer=tf.compat.v1.train.AdamOptimizer(),
        loss=lambda x, dist: -dist.log_prob(x[0, :event_size]),
        metrics=[])
    model.fit(x, x,
              batch_size=25,
              epochs=1,
              steps_per_epoch=1)  # Usually `n // batch_size`.


@test_util.run_all_in_graph_and_eager_modes
class MultivariateNormalTriLTest(tf.test.TestCase):

  def _check_distribution(self, t, x):
    self.assertIsInstance(x, tfd.MultivariateNormalTriL)
    t_back = tf.concat([
        x.loc, tfb.ScaleTriL().inverse(x.scale.to_dense())], axis=-1)
    self.assertAllClose(*self.evaluate([t, t_back]), atol=1e-6, rtol=1e-5)

  def test_new(self):
    d = 4
    p = tfpl.MultivariateNormalTriL.params_size(d)
    t = tfd.Normal(0, 1).sample([2, 3, p], seed=42)
    x = tfpl.MultivariateNormalTriL.new(t, d, validate_args=True)
    self._check_distribution(t, x)

  def test_layer(self):
    d = 4
    p = tfpl.MultivariateNormalTriL.params_size(d)
    layer = tfpl.MultivariateNormalTriL(d, tfd.Distribution.mean)
    t = tfd.Normal(0, 1).sample([2, 3, p], seed=42)
    x = layer(t)
    self._check_distribution(t, x)

  def test_doc_string(self):
    # Load data.
    n = int(1e3)
    scale_tril = np.array([[1.6180, 0.],
                           [-2.7183, 3.1416]]).astype(np.float32)
    scale_noise = 0.01
    x = self.evaluate(tfd.Normal(loc=0, scale=1).sample([n, 2]))
    eps = tfd.Normal(loc=0, scale=scale_noise).sample([n, 2])
    y = self.evaluate(tf.matmul(x, scale_tril) + eps)
    d = y.shape[-1]

    # To save testing time, let's encode the answer (i.e., _cheat_). Note: in
    # writing this test we verified the correct answer is achieved with random
    # initialization.
    true_kernel = np.pad(scale_tril, [[0, 0], [0, 3]], 'constant')
    true_bias = np.array([0, 0, np.log(scale_noise), 0, np.log(scale_noise)])

    # Create model.
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            tfpl.MultivariateNormalTriL.params_size(d),
            kernel_initializer=lambda s, **_: true_kernel,
            bias_initializer=lambda s, **_: true_bias),
        tfpl.MultivariateNormalTriL(d),
    ])

    # Fit.
    model.compile(
        optimizer=tf.compat.v1.train.AdamOptimizer(),
        loss=lambda y, model: -model.log_prob(y),
        metrics=[])
    batch_size = 100
    model.fit(x, y,
              batch_size=batch_size,
              epochs=1,  # One ping only.
              steps_per_epoch=n // batch_size)
    self.assertAllClose(true_kernel, model.get_weights()[0],
                        atol=1e-2, rtol=1e-3)
    self.assertAllClose(true_bias, model.get_weights()[1],
                        atol=1e-2, rtol=1e-3)


@test_util.run_all_in_graph_and_eager_modes
class OneHotCategoricalTest(tf.test.TestCase):

  def _check_distribution(self, t, x):
    self.assertIsInstance(x, tfd.OneHotCategorical)
    [t_, x_logits_, x_probs_, mean_] = self.evaluate([
        t, x.logits, x.probs, x.mean()])
    self.assertAllClose(t_, x_logits_, atol=1e-6, rtol=1e-5)
    self.assertAllClose(x_probs_, mean_, atol=1e-6, rtol=1e-5)

  def test_new(self):
    d = 4
    p = tfpl.OneHotCategorical.params_size(d)
    t = tfd.Normal(0, 1).sample([2, 3, p], seed=42)
    x = tfpl.OneHotCategorical.new(t, d, validate_args=True)
    self._check_distribution(t, x)

  def test_layer(self):
    d = 4
    p = tfpl.OneHotCategorical.params_size(d)
    layer = tfpl.OneHotCategorical(d, validate_args=True)
    t = tfd.Normal(0, 1).sample([2, 3, p], seed=42)
    x = layer(t)
    self._check_distribution(t, x)

  def test_doc_string(self):
    # Load data.
    n = int(1e4)
    scale_noise = 0.01
    x = self.evaluate(tfd.Normal(loc=0, scale=1).sample([n, 2]))
    eps = tfd.Normal(loc=0, scale=scale_noise).sample([n, 1])
    y = self.evaluate(tfd.OneHotCategorical(
        logits=_vec_pad(
            0.3142 + 1.6180 * x[..., :1] - 2.7183 * x[..., 1:] + eps),
        dtype=tf.float32).sample())
    d = y.shape[-1]

    # Create model.
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(tfpl.OneHotCategorical.params_size(d) - 1),
        tf.keras.layers.Lambda(_vec_pad),
        tfpl.OneHotCategorical(d),
    ])

    # Fit.
    model.compile(
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.5),
        loss=lambda y, model: -model.log_prob(y),
        metrics=[])
    batch_size = 100
    model.fit(x, y,
              batch_size=batch_size,
              epochs=1,
              steps_per_epoch=n // batch_size,
              shuffle=True)
    self.assertAllClose([[1.6180], [-2.7183]], model.get_weights()[0],
                        atol=0, rtol=0.1)


@test_util.run_all_in_graph_and_eager_modes
class CategoricalMixtureOfOneHotCategoricalTest(tf.test.TestCase):

  def _check_distribution(self, t, x):
    self.assertIsInstance(x, tfd.MixtureSameFamily)
    self.assertIsInstance(x.mixture_distribution, tfd.Categorical)
    self.assertIsInstance(x.components_distribution, tfd.OneHotCategorical)
    t_back = tf.concat([
        x.mixture_distribution.logits,
        tf.reshape(x.components_distribution.logits, shape=[2, 3, -1]),
    ], axis=-1)
    [
        t_,
        t_back_,
        x_mean_,
        x_log_mean_,
        sample_mean_,
    ] = self.evaluate([
        t,
        t_back,
        x.mean(),
        x.log_mean(),
        tf.reduce_mean(input_tensor=x.sample(int(10e3), seed=42), axis=0),
    ])
    self.assertAllClose(t_, t_back_, atol=1e-6, rtol=1e-5)
    self.assertAllClose(x_mean_, np.exp(x_log_mean_), atol=1e-6, rtol=1e-5)
    self.assertAllClose(sample_mean_, x_mean_, atol=1e-3, rtol=0.1)

  def test_new(self):
    k = 2  # num components
    d = 4  # event size
    p = tfpl.CategoricalMixtureOfOneHotCategorical.params_size(d, k)
    t = tfd.Normal(0, 1).sample([2, 3, p], seed=42)
    x = tfpl.CategoricalMixtureOfOneHotCategorical.new(
        t, d, k, validate_args=True)
    self._check_distribution(t, x)

  def test_layer(self):
    k = 2  # num components
    d = 4  # event size
    p = tfpl.CategoricalMixtureOfOneHotCategorical.params_size(d, k)
    layer = tfpl.CategoricalMixtureOfOneHotCategorical(
        d, k, validate_args=True)
    t = tfd.Normal(0, 1).sample([2, 3, p], seed=42)
    x = layer(t)
    self._check_distribution(t, x)

  def test_doc_string(self):
    # Load data.
    n = int(1e3)
    scale_noise = 0.01
    x = self.evaluate(tfd.Normal(loc=0, scale=1).sample([n, 2]))
    eps = tfd.Normal(loc=0, scale=scale_noise).sample([n, 1])
    y = self.evaluate(tfd.OneHotCategorical(
        logits=_vec_pad(
            0.3142 + 1.6180 * x[..., :1] - 2.7183 * x[..., 1:] + eps),
        dtype=tf.float32).sample())
    d = y.shape[-1]

    # Create model.
    k = 2
    p = tfpl.CategoricalMixtureOfOneHotCategorical.params_size(d, k)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(p),
        tfpl.CategoricalMixtureOfOneHotCategorical(d, k),
    ])

    # Fit.
    model.compile(
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.5),
        loss=lambda y, model: -model.log_prob(y),
        metrics=[])
    batch_size = 100
    model.fit(x, y,
              batch_size=batch_size,
              epochs=1,
              steps_per_epoch=1,  # Usually `n // batch_size`.
              shuffle=True)

    yhat = model(x)
    self.assertIsInstance(yhat, tfd.MixtureSameFamily)
    self.assertIsInstance(yhat.mixture_distribution, tfd.Categorical)
    self.assertIsInstance(yhat.components_distribution, tfd.OneHotCategorical)
    # TODO(b/120221303): For now we just check that the code executes and we get
    # back a distribution instance. Better would be to change the data
    # generation so the model becomes well-specified (and we can check correctly
    # fitted params). However, not doing this test is not critical since all
    # components are unit-tested. (Ie, what we really want here--but don't
    # strictly need--is another end-to-end test.)


@test_util.run_all_in_graph_and_eager_modes
class _IndependentLayerTest(object):
  """Base class for testing independent distribution layers.

  Instances of subclasses must set:
    self.layer_class: The independent distribution layer class.
    self.dist_class: The underlying `tfd.Distribution` class.
    self.dtype: The data type for the parameters passed to the layer.
    self.use_static_shape: Whether or not test tensor inputs should have
      statically-known shapes.
  """

  def _distribution_to_params(self, distribution, batch_shape):
    """Given a self.layer_class instance, return a tensor of its parameters."""
    raise NotImplementedError

  def _build_tensor(self, ndarray, dtype=None):
    # Enforce parameterized dtype and static/dynamic testing.
    ndarray = np.asarray(ndarray).astype(
        dtype if dtype is not None else self.dtype)
    return tf.compat.v1.placeholder_with_default(
        input=ndarray, shape=ndarray.shape if self.use_static_shape else None)

  def _check_distribution(self, t, x, batch_shape):
    self.assertIsInstance(x, tfd.Independent)
    self.assertIsInstance(x.distribution, self.dist_class)
    t_back = self._distribution_to_params(x.distribution, batch_shape)
    [t_, t_back_] = self.evaluate([t, t_back])
    self.assertAllClose(t_, t_back_, atol=1e-6, rtol=1e-5)

  def test_new(self):
    batch_shape = self._build_tensor([2], dtype=np.int32)
    event_shape = self._build_tensor([2, 1, 2], dtype=np.int32)
    p = self.layer_class.params_size(event_shape)

    low = self._build_tensor(-3.)
    high = self._build_tensor(3.)
    t = tfd.Uniform(low, high).sample(tf.concat([batch_shape, [p]], 0), seed=42)

    x = self.layer_class.new(t, event_shape, validate_args=True)
    self._check_distribution(t, x, batch_shape)

  def test_layer(self):
    batch_shape = self._build_tensor([5, 5], dtype=np.int32)
    p = self.layer_class.params_size()

    low = self._build_tensor(-3.)
    high = self._build_tensor(3.)
    t = tfd.Uniform(low, high).sample(tf.concat([batch_shape, [p]], 0), seed=42)

    layer = self.layer_class(validate_args=True)
    x = layer(t)
    self._check_distribution(t, x, batch_shape)


@test_util.run_all_in_graph_and_eager_modes
class _IndependentBernoulliTest(_IndependentLayerTest):
  layer_class = tfpl.IndependentBernoulli
  dist_class = tfd.Bernoulli

  def _distribution_to_params(self, distribution, batch_shape):
    return tf.reshape(distribution.logits,
                      tf.concat([batch_shape, [-1]], axis=-1))


@test_util.run_all_in_graph_and_eager_modes
class IndependentBernoulliTestDynamicShape(tf.test.TestCase,
                                           _IndependentBernoulliTest):
  dtype = np.float64
  use_static_shape = False


@test_util.run_all_in_graph_and_eager_modes
class IndependentBernoulliTestStaticShape(tf.test.TestCase,
                                          _IndependentBernoulliTest):
  dtype = np.float32
  use_static_shape = True

  def test_doc_string(self):
    # Load data.
    n = int(1e4)
    scale_tril = np.array([[1.6180, 0.],
                           [-2.7183, 3.1416]]).astype(np.float32)
    scale_noise = 0.01
    x = self.evaluate(tfd.Normal(loc=0, scale=1).sample([n, 2]))
    eps = tfd.Normal(loc=0, scale=scale_noise).sample([n, 2])
    y = self.evaluate(tfd.Bernoulli(
        logits=tf.reshape(tf.matmul(x, scale_tril) + eps,
                          shape=[n, 1, 2, 1])).sample())
    event_shape = y.shape[1:]

    # Create model.
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            tfpl.IndependentBernoulli.params_size(event_shape)),
        tfpl.IndependentBernoulli(event_shape),
    ])

    # Fit.
    model.compile(
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.5),
        loss=lambda y, model: -model.log_prob(y),
        metrics=[])
    batch_size = 100
    model.fit(x, y,
              batch_size=batch_size,
              epochs=1,
              steps_per_epoch=n // batch_size,
              shuffle=True)
    self.assertAllClose(scale_tril, model.get_weights()[0],
                        atol=0.15, rtol=0.15)
    self.assertAllClose([0., 0.], model.get_weights()[1],
                        atol=0.15, rtol=0.15)


@test_util.run_all_in_graph_and_eager_modes
class _IndependentLogisticTest(_IndependentLayerTest):
  layer_class = tfpl.IndependentLogistic
  dist_class = tfd.Logistic

  def _distribution_to_params(self, distribution, batch_shape):
    return tf.concat([
        tf.reshape(distribution.loc, tf.concat([batch_shape, [-1]], axis=-1)),
        tfd.softplus_inverse(tf.reshape(
            distribution.scale, tf.concat([batch_shape, [-1]], axis=-1)))
    ], -1)


@test_util.run_all_in_graph_and_eager_modes
class IndependentLogisticTestDynamicShape(tf.test.TestCase,
                                          _IndependentLogisticTest):
  dtype = np.float32
  use_static_shape = False


@test_util.run_all_in_graph_and_eager_modes
class IndependentLogisticTestStaticShape(tf.test.TestCase,
                                         _IndependentLogisticTest):
  dtype = np.float64
  use_static_shape = True

  def test_doc_string(self):
    input_shape = [28, 28, 1]
    encoded_shape = 2
    encoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=input_shape, dtype=self.dtype),
        tfkl.Flatten(),
        tfkl.Dense(10, activation='relu'),
        tfkl.Dense(tfpl.IndependentLogistic.params_size(encoded_shape)),
        tfpl.IndependentLogistic(encoded_shape),
        tfkl.Lambda(lambda x: x + 0.)  # To force conversion to tensor.
    ])

    # Test that we can run the model and get a sample.
    x = np.random.randn(*([1] + input_shape)).astype(self.dtype)
    self.assertEqual((1, 2), encoder.predict_on_batch(x).shape)

    out = encoder(tf.convert_to_tensor(value=x))
    self.assertEqual((1, 2), out.shape)
    self.assertEqual((1, 2), self.evaluate(out).shape)


@test_util.run_all_in_graph_and_eager_modes
class _IndependentNormalTest(_IndependentLayerTest):
  layer_class = tfpl.IndependentNormal
  dist_class = tfd.Normal

  def _distribution_to_params(self, distribution, batch_shape):
    return tf.concat([
        tf.reshape(distribution.loc, tf.concat([batch_shape, [-1]], axis=-1)),
        tfd.softplus_inverse(tf.reshape(
            distribution.scale, tf.concat([batch_shape, [-1]], axis=-1)))
    ], -1)

  def test_keras_sequential_with_unknown_input_size(self):
    input_shape = [28, 28, 1]
    encoded_shape = self._build_tensor([2], dtype=np.int32)
    params_size = tfpl.IndependentNormal.params_size(encoded_shape)

    def reshape(x):
      return tf.reshape(
          x, tf.concat([tf.shape(input=x)[:-1], [-1, params_size]], 0))

    # Test a Sequential model where the input to IndependentNormal does not have
    # a statically-known shape.
    encoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=input_shape, dtype=self.dtype),
        tfkl.Flatten(),
        tfkl.Dense(12, activation='relu'),
        tfkl.Lambda(reshape),
        # When encoded_shape/params_size are placeholders, the input to the
        # IndependentNormal has shape (?, ?, ?) or (1, ?, ?), depending on
        # whether or not encoded_shape's shape is known.
        tfpl.IndependentNormal(encoded_shape),
        tfkl.Lambda(lambda x: x + 0.)  # To force conversion to tensor.
    ])

    x = np.random.randn(*([1] + input_shape)).astype(self.dtype)
    self.assertEqual((1, 3, 2), encoder.predict_on_batch(x).shape)

    out = encoder(tf.convert_to_tensor(value=x))
    if tf.executing_eagerly():
      self.assertEqual((1, 3, 2), out.shape)
    elif self.use_static_shape:
      self.assertEqual([1, None, None], out.shape.as_list())
    self.assertEqual((1, 3, 2), self.evaluate(out).shape)


@test_util.run_all_in_graph_and_eager_modes
class IndependentNormalTestDynamicShape(tf.test.TestCase,
                                        _IndependentNormalTest):
  dtype = np.float32
  use_static_shape = False


@test_util.run_all_in_graph_and_eager_modes
class IndependentNormalTestStaticShape(tf.test.TestCase,
                                       _IndependentNormalTest):
  dtype = np.float64
  use_static_shape = True

  def test_doc_string(self):
    input_shape = [28, 28, 1]
    encoded_shape = 2
    encoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=input_shape, dtype=self.dtype),
        tfkl.Flatten(),
        tfkl.Dense(10, activation='relu'),
        tfkl.Dense(tfpl.IndependentNormal.params_size(encoded_shape)),
        tfpl.IndependentNormal(encoded_shape),
        tfkl.Lambda(lambda x: x + 0.)  # To force conversion to tensor.
    ])

    # Test that we can run the model and get a sample.
    x = np.random.randn(*([1] + input_shape)).astype(self.dtype)
    self.assertEqual((1, 2), encoder.predict_on_batch(x).shape)

    out = encoder(tf.convert_to_tensor(value=x))
    self.assertEqual((1, 2), out.shape)
    self.assertEqual((1, 2), self.evaluate(out).shape)


@test_util.run_all_in_graph_and_eager_modes
class _IndependentPoissonTest(_IndependentLayerTest):
  layer_class = tfpl.IndependentPoisson
  dist_class = tfd.Poisson

  def _distribution_to_params(self, distribution, batch_shape):
    return tf.reshape(distribution.log_rate,
                      tf.concat([batch_shape, [-1]], axis=-1))


@test_util.run_all_in_graph_and_eager_modes
class IndependentPoissonTestDynamicShape(tf.test.TestCase,
                                         _IndependentPoissonTest):
  dtype = np.float32
  use_static_shape = False


@test_util.run_all_in_graph_and_eager_modes
class IndependentPoissonTestStaticShape(tf.test.TestCase,
                                        _IndependentPoissonTest):
  dtype = np.float64
  use_static_shape = True

  def test_doc_string(self):
    # Create example data.
    n = 2000
    d = 4
    x = self.evaluate(tfd.Uniform(low=1., high=10.).sample([n, d], seed=42))
    w = [[0.314], [0.272], [-0.162], [0.058]]
    log_rate = tf.matmul(x, w) - 0.141
    y = self.evaluate(tfd.Poisson(log_rate=log_rate).sample())

    # Poisson regression.
    model = tfk.Sequential([
        tfkl.Dense(tfpl.IndependentPoisson.params_size(1)),
        tfpl.IndependentPoisson(1)
    ])

    # Fit.
    model.compile(
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.05),
        loss=lambda y, model: -model.log_prob(y),
        metrics=[])
    batch_size = 50

    model.fit(x, y,
              batch_size=batch_size,
              epochs=1,
              steps_per_epoch=1,  # Usually `n // batch_size`.
              verbose=True,
              shuffle=True)


@test_util.run_all_in_graph_and_eager_modes
class _MixtureLayerTest(object):
  """Base class for testing mixture (same-family) distribution layers.

  Instances of subclasses must set:
    self.layer_class: The mixture distribution layer class.
    self.dist_class: The underlying component `tfd.Distribution` class.
    self.dtype: The data type for the parameters passed to the layer.
    self.use_static_shape: Whether or not test tensor inputs should have
      statically-known shapes.
  """

  def _distribution_to_params(self, distribution, batch_shape):
    """Given a self.layer_class instance, return a tensor of its parameters."""
    raise NotImplementedError

  def _build_tensor(self, ndarray, dtype=None):
    # Enforce parameterized dtype and static/dynamic testing.
    ndarray = np.asarray(ndarray).astype(
        dtype if dtype is not None else self.dtype)
    return tf.compat.v1.placeholder_with_default(
        input=ndarray, shape=ndarray.shape if self.use_static_shape else None)

  def _check_distribution(self, t, x, batch_shape):
    self.assertIsInstance(x, tfd.MixtureSameFamily)
    self.assertIsInstance(x.mixture_distribution, tfd.Categorical)
    self.assertIsInstance(x.components_distribution, tfd.Independent)
    self.assertIsInstance(x.components_distribution.distribution,
                          self.dist_class)

    t_back = self._distribution_to_params(x, batch_shape)
    [t_, t_back_] = self.evaluate([t, t_back])
    self.assertAllClose(t_, t_back_, atol=1e-6, rtol=1e-5)

  def test_new(self):
    n = self._build_tensor(4, dtype=np.int32)
    event_shape = self._build_tensor(3, dtype=np.int32)
    p = self.layer_class.params_size(n, event_shape)

    batch_shape = self._build_tensor([4, 2], dtype=np.int32)
    low = self._build_tensor(-3.)
    high = self._build_tensor(3.)
    t = tfd.Uniform(low, high).sample(tf.concat([batch_shape, [p]], 0), seed=42)

    x = self.layer_class.new(t, n, event_shape, validate_args=True)
    self._check_distribution(t, x, batch_shape)

  def test_layer(self):
    n = self._build_tensor(3, dtype=np.int32)
    event_shape = self._build_tensor([4, 2], dtype=np.int32)
    p = self.layer_class.params_size(n, event_shape)

    batch_shape = self._build_tensor([7, 3], dtype=np.int32)
    low = self._build_tensor(-3.)
    high = self._build_tensor(3.)
    t = tfd.Uniform(low, high).sample(tf.concat([batch_shape, [p]], 0), seed=42)

    layer = self.layer_class(n, event_shape, validate_args=True)
    x = layer(t)
    self._check_distribution(t, x, batch_shape)


@test_util.run_all_in_graph_and_eager_modes
class _MixtureLogisticTest(_MixtureLayerTest):
  layer_class = tfpl.MixtureLogistic
  dist_class = tfd.Logistic

  def _distribution_to_params(self, distribution, batch_shape):
    """Given a self.layer_class instance, return a tensor of its parameters."""
    params_shape = tf.concat([batch_shape, [-1]], axis=0)
    batch_and_n_shape = tf.concat(
        [tf.shape(input=distribution.mixture_distribution.logits), [-1]],
        axis=0)
    cd = distribution.components_distribution.distribution
    return tf.concat([
        distribution.mixture_distribution.logits,
        tf.reshape(tf.concat([
            tf.reshape(cd.loc, batch_and_n_shape),
            tf.reshape(tfd.softplus_inverse(cd.scale), batch_and_n_shape)
        ], axis=-1), params_shape),
    ], axis=-1)

  def test_doc_string(self):
    # Load data (graph of a cardioid).
    n = 2000
    t = self.evaluate(tfd.Uniform(low=-np.pi, high=np.pi).sample([n, 1]))
    r = 2 * (1 - tf.cos(t))
    x = tf.convert_to_tensor(value=self.evaluate(
        r * tf.sin(t) + tfd.Normal(loc=0., scale=0.1).sample([n, 1])))
    y = tf.convert_to_tensor(value=self.evaluate(
        r * tf.cos(t) + tfd.Normal(loc=0., scale=0.1).sample([n, 1])))

    # Model the distribution of y given x with a Mixture Density Network.
    event_shape = self._build_tensor([1], dtype=np.int32)
    num_components = self._build_tensor(5, dtype=np.int32)
    params_size = tfpl.MixtureNormal.params_size(num_components, event_shape)
    model = tfk.Sequential([
        tfkl.Dense(12, activation='relu'),
        # NOTE: We must hard-code 15 below, instead of using `params_size`,
        # because the first argument to `tfkl.Dense` must be an integer (and
        # not, e.g., a placeholder tensor).
        tfkl.Dense(15, activation=None),
        tfpl.MixtureLogistic(num_components, event_shape),
    ])

    # Fit.
    batch_size = 100
    model.compile(
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.02),
        loss=lambda y, model: -model.log_prob(y))
    model.fit(x, y,
              batch_size=batch_size,
              epochs=1,
              steps_per_epoch=n // batch_size)

    self.assertEqual(15, self.evaluate(tf.convert_to_tensor(value=params_size)))


@test_util.run_all_in_graph_and_eager_modes
class MixtureLogisticTestDynamicShape(tf.test.TestCase,
                                      _MixtureLogisticTest):
  dtype = np.float64
  use_static_shape = False


@test_util.run_all_in_graph_and_eager_modes
class MixtureLogisticTestStaticShape(tf.test.TestCase,
                                     _MixtureLogisticTest):
  dtype = np.float32
  use_static_shape = True


@test_util.run_all_in_graph_and_eager_modes
class _MixtureNormalTest(_MixtureLayerTest):
  layer_class = tfpl.MixtureNormal
  dist_class = tfd.Normal

  def _distribution_to_params(self, distribution, batch_shape):
    """Given a self.layer_class instance, return a tensor of its parameters."""
    params_shape = tf.concat([batch_shape, [-1]], axis=0)
    batch_and_n_shape = tf.concat(
        [tf.shape(input=distribution.mixture_distribution.logits), [-1]],
        axis=0)
    cd = distribution.components_distribution.distribution
    return tf.concat([
        distribution.mixture_distribution.logits,
        tf.reshape(tf.concat([
            tf.reshape(cd.loc, batch_and_n_shape),
            tf.reshape(tfd.softplus_inverse(cd.scale), batch_and_n_shape)
        ], axis=-1), params_shape),
    ], axis=-1)

  def test_doc_string(self):
    # Load data (graph of a cardioid).
    n = 2000
    t = self.evaluate(tfd.Uniform(low=-np.pi, high=np.pi).sample([n, 1]))
    r = 2 * (1 - tf.cos(t))
    x = tf.convert_to_tensor(value=self.evaluate(
        r * tf.sin(t) + tfd.Normal(loc=0., scale=0.1).sample([n, 1])))
    y = tf.convert_to_tensor(value=self.evaluate(
        r * tf.cos(t) + tfd.Normal(loc=0., scale=0.1).sample([n, 1])))

    # Model the distribution of y given x with a Mixture Density Network.
    event_shape = self._build_tensor([1], dtype=np.int32)
    num_components = self._build_tensor(5, dtype=np.int32)
    params_size = tfpl.MixtureNormal.params_size(num_components, event_shape)
    model = tfk.Sequential([
        tfkl.Dense(12, activation='relu'),
        # NOTE: We must hard-code 15 below, instead of using `params_size`,
        # because the first argument to `tfkl.Dense` must be an integer (and
        # not, e.g., a placeholder tensor).
        tfkl.Dense(15, activation=None),
        tfpl.MixtureNormal(num_components, event_shape),
    ])

    # Fit.
    batch_size = 100
    model.compile(
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.02),
        loss=lambda y, model: -model.log_prob(y))
    model.fit(x, y,
              batch_size=batch_size,
              epochs=1,
              steps_per_epoch=n // batch_size)

    self.assertEqual(15, self.evaluate(tf.convert_to_tensor(value=params_size)))


@test_util.run_all_in_graph_and_eager_modes
class MixtureNormalTestDynamicShape(tf.test.TestCase,
                                    _MixtureNormalTest):
  dtype = np.float32
  use_static_shape = False


@test_util.run_all_in_graph_and_eager_modes
class MixtureNormalTestStaticShape(tf.test.TestCase,
                                   _MixtureNormalTest):
  dtype = np.float64
  use_static_shape = True


@test_util.run_all_in_graph_and_eager_modes
class _MixtureSameFamilyTest(object):

  def _build_tensor(self, ndarray, dtype=None):
    # Enforce parameterized dtype and static/dynamic testing.
    ndarray = np.asarray(ndarray).astype(
        dtype if dtype is not None else self.dtype)
    return tf.compat.v1.placeholder_with_default(
        input=ndarray, shape=ndarray.shape if self.use_static_shape else None)

  def _check_distribution(self, t, x, batch_shape):
    self.assertIsInstance(x, tfd.MixtureSameFamily)
    self.assertIsInstance(x.mixture_distribution, tfd.Categorical)
    self.assertIsInstance(x.components_distribution, tfd.MultivariateNormalTriL)

    shape = tf.concat([batch_shape, [-1]], axis=0)
    batch_and_n_shape = tf.concat(
        [tf.shape(input=x.mixture_distribution.logits), [-1]], axis=0)
    cd = x.components_distribution
    scale_tril = tfb.ScaleTriL(diag_shift=np.array(1e-5, self.dtype))
    t_back = tf.concat([
        x.mixture_distribution.logits,
        tf.reshape(tf.concat([
            tf.reshape(cd.loc, batch_and_n_shape),
            tf.reshape(
                scale_tril.inverse(cd.scale.to_dense()),
                batch_and_n_shape),
        ], axis=-1), shape),
    ], axis=-1)
    [t_, t_back_] = self.evaluate([t, t_back])
    self.assertAllClose(t_, t_back_, atol=1e-6, rtol=1e-5)

  def test_new(self):
    n = self._build_tensor(4, dtype=np.int32)
    batch_shape = self._build_tensor([4, 2], dtype=np.int32)
    event_size = self._build_tensor(3, dtype=np.int32)
    low = self._build_tensor(-3.)
    high = self._build_tensor(3.)
    cps = tfpl.MultivariateNormalTriL.params_size(event_size)
    p = tfpl.MixtureSameFamily.params_size(n, cps)

    t = tfd.Uniform(low, high).sample(tf.concat([batch_shape, [p]], 0), seed=42)
    normal = tfpl.MultivariateNormalTriL(event_size, validate_args=True)
    x = tfpl.MixtureSameFamily.new(t, n, normal, validate_args=True)
    self._check_distribution(t, x, batch_shape)

  def test_layer(self):
    n = self._build_tensor(3, dtype=np.int32)
    batch_shape = self._build_tensor([7, 3], dtype=np.int32)
    event_size = self._build_tensor(4, dtype=np.int32)
    low = self._build_tensor(-3.)
    high = self._build_tensor(3.)
    cps = tfpl.MultivariateNormalTriL.params_size(event_size)
    p = tfpl.MixtureSameFamily.params_size(n, cps)

    normal = tfpl.MultivariateNormalTriL(event_size, validate_args=True)
    layer = tfpl.MixtureSameFamily(n, normal, validate_args=True)
    t = tfd.Uniform(low, high).sample(tf.concat([batch_shape, [p]], 0), seed=42)
    x = layer(t)
    self._check_distribution(t, x, batch_shape)

  def test_doc_string(self):
    # Load data (graph of a cardioid).
    n = 2000
    t = self.evaluate(tfd.Uniform(low=-np.pi, high=np.pi).sample([n, 1]))
    r = 2 * (1 - tf.cos(t))
    x = tf.convert_to_tensor(value=self.evaluate(
        r * tf.sin(t) + tfd.Normal(loc=0., scale=0.1).sample([n, 1])))
    y = tf.convert_to_tensor(value=self.evaluate(
        r * tf.cos(t) + tfd.Normal(loc=0., scale=0.1).sample([n, 1])))

    # Model the distribution of y given x with a Mixture Density Network.
    event_shape = self._build_tensor([1], dtype=np.int32)
    num_components = self._build_tensor(5, dtype=np.int32)
    params_size = tfpl.MixtureSameFamily.params_size(
        num_components, tfpl.IndependentNormal.params_size(event_shape))
    model = tfk.Sequential([
        tfkl.Dense(12, activation='relu'),
        # NOTE: We must hard-code 15 below, instead of using `params_size`,
        # because the first argument to `tfkl.Dense` must be an integer (and
        # not, e.g., a placeholder tensor).
        tfkl.Dense(15, activation=None),
        tfpl.MixtureSameFamily(num_components,
                               tfpl.IndependentNormal(event_shape)),
    ])

    # Fit.
    batch_size = 100
    model.compile(
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.02),
        loss=lambda y, model: -model.log_prob(y))
    model.fit(x, y,
              batch_size=batch_size,
              epochs=1,
              steps_per_epoch=1)  # Usually `n // batch_size`.

    self.assertEqual(15, self.evaluate(tf.convert_to_tensor(value=params_size)))


@test_util.run_all_in_graph_and_eager_modes
class MixtureSameFamilyTestDynamicShape(tf.test.TestCase,
                                        _MixtureSameFamilyTest):
  dtype = np.float32
  use_static_shape = False


@test_util.run_all_in_graph_and_eager_modes
class MixtureSameFamilyTestStaticShape(tf.test.TestCase,
                                       _MixtureSameFamilyTest):
  dtype = np.float64
  use_static_shape = True


if __name__ == '__main__':
  tf.test.main()
