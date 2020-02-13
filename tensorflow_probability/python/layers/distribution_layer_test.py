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
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import layers as tfpl
from tensorflow_probability.python.internal import test_util

tfk = tf.keras

tfkl = tf.keras.layers


def _logit_avg_expit(t):
  """Computes `logit(mean(expit(t)))` in a numerically stable manner."""
  log_avg_prob = (
      tf.reduce_logsumexp(-tf.nn.softplus(-t), axis=0) -
      tf.math.log(tf.cast(tf.shape(t)[0], t.dtype)))
  return log_avg_prob - tf.math.log1p(-tf.exp(log_avg_prob))


def _vec_pad(x, value=0):
  """Prepends a column of zeros to a matrix."""
  paddings = tf.concat(
      [tf.zeros([tf.rank(x) - 1, 2], dtype=tf.int32), [[1, 0]]], axis=0)
  return tf.pad(x, paddings=paddings, constant_values=value)


# TODO(b/143642032): Figure out how to solve issues with save/load, so that we
# can decorate all of these tests with @test_util.test_all_tf_execution_regimes
@test_util.test_graph_and_eager_modes
class EndToEndTest(test_util.TestCase):
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
    self.train_size = 10000
    self.test_size = 1000
    self.x = (np.random.rand(
        self.train_size, *self.input_shape) > 0.75).astype(np.float32)
    self.x_test = (np.random.rand(
        self.test_size, *self.input_shape) > 0.75).astype(np.float32)
    super(EndToEndTest, self).setUp()

  def test_keras_sequential_api(self):
    """Test `DistributionLambda`s are composable via Keras `Sequential` API."""

    prior_model = tfk.Sequential([
        tfpl.VariableLayer(shape=[self.encoded_size]),
        tfpl.DistributionLambda(
            lambda t: tfd.Independent(tfd.Normal(loc=t, scale=1),  # pylint: disable=g-long-lambda
                                      reinterpreted_batch_ndims=1)),
    ])

    beta = tf.Variable(0.9, name='beta')  # "beta" as in beta-VAE.

    encoder_model = tfk.Sequential([
        tfkl.InputLayer(input_shape=self.input_shape),
        tfkl.Flatten(),
        tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(self.encoded_size)),
        tfpl.MultivariateNormalTriL(
            self.encoded_size,
            activity_regularizer=tfpl.KLDivergenceRegularizer(
                prior_model, weight=beta)),
    ])

    decoder_model = tfk.Sequential([
        tfkl.InputLayer(input_shape=[self.encoded_size]),
        tfkl.Dense(tfpl.IndependentBernoulli.params_size(self.input_shape)),
        tfpl.IndependentBernoulli(self.input_shape, tfd.Bernoulli.logits),
    ])

    vae_model = tfk.Model(
        inputs=encoder_model.inputs,
        # TODO(b/139437503): remove training=False once cl/263432058 hits
        # nightly.
        outputs=decoder_model(encoder_model.outputs[0], training=False))

    self.assertLen(vae_model.trainable_weights, 4 + 1 + 1)

    def accuracy(x, rv_x):
      rv_x = getattr(rv_x, '_tfp_distribution', rv_x)
      return tf.reduce_mean(
          tf.cast(tf.equal(x, rv_x.mode()), x.dtype),
          axis=tf.range(-rv_x.event_shape.ndims, 0))

    vae_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.5),
        loss=lambda x, rv_x: -rv_x.log_prob(x),
        metrics=[accuracy])

    self.evaluate([v.initializer for v in vae_model.variables])

    vae_model.fit(self.x, self.x,
                  batch_size=25,
                  epochs=1,
                  verbose=True,
                  validation_data=(self.x_test, self.x_test),
                  shuffle=True)
    yhat = vae_model(tf.convert_to_tensor(self.x_test))
    self.assertIsInstance(yhat, tfd.Independent)
    self.assertIsInstance(yhat.distribution, tfd.Bernoulli)

  def test_keras_functional_api(self):
    """Test `DistributionLambda`s are composable via Keras functional API."""

    beta = tf.Variable(  # 0 vars since not trainable.
        0.9, trainable=False, name='beta')  # "beta" as in beta-VAE

    encoder_model = [
        tfkl.Flatten(),
        tfkl.Dense(10, activation='relu'),  # 2 vars
        tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(  # 2 vars
            self.encoded_size)),
        tfpl.MultivariateNormalTriL(self.encoded_size),
        tfpl.KLDivergenceAddLoss(
            tfd.Independent(
                tfd.Normal(
                    loc=tf.Variable([0., 0.]),  # 1 var
                    scale=tfp.util.TransformedVariable(  # 1 var
                        1., bijector=tfb.Exp())),
                reinterpreted_batch_ndims=1),
            weight=beta),
    ]

    decoder_model = [
        tfkl.Dense(10, activation='relu'),  # 2 vars
        tfkl.Dense(tfpl.IndependentBernoulli.params_size(  # 2 vars
            self.input_shape)),
        tfpl.IndependentBernoulli(self.input_shape, tfd.Bernoulli.logits),
    ]

    images = tfkl.Input(shape=self.input_shape)
    encoded = functools.reduce(lambda x, f: f(x), encoder_model, images)
    decoded = functools.reduce(lambda x, f: f(x), decoder_model, encoded)

    vae_model = tfk.Model(inputs=images, outputs=decoded)
    vae_model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=lambda x, rv_x: -rv_x.log_prob(x),
        metrics=[])
    self.assertLen(vae_model.trainable_weights,
                   (2 + 2) + (2 + 2) + (1 + 1) + 0)
    self.evaluate([v.initializer for v in vae_model.variables])
    vae_model.fit(self.x, self.x,
                  batch_size=25,
                  epochs=1,
                  verbose=True,
                  validation_data=(self.x_test, self.x_test),
                  shuffle=True)
    yhat = vae_model(tf.convert_to_tensor(self.x_test))
    self.assertIsInstance(yhat, tfd.Independent)
    self.assertIsInstance(yhat.distribution, tfd.Bernoulli)

  def test_keras_model_api(self):
    """Test `DistributionLambda`s are composable via Keras `Model` API."""

    class Encoder(tfk.Model):
      """Encoder."""

      def __init__(self, input_shape, encoded_size, train_size):
        super(Encoder, self).__init__()
        self._sub_layers = [
            tfkl.Flatten(),
            tfkl.Dense(10, activation='relu'),
            tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size)),
            tfpl.MultivariateNormalTriL(encoded_size),
            tfpl.KLDivergenceAddLoss(
                tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                                reinterpreted_batch_ndims=1),
                weight=0.9),  # "beta" as in beta-VAE.
        ]

      def call(self, inputs):
        return functools.reduce(lambda x, f: f(x), self._sub_layers, inputs)

    class Decoder(tfk.Model):
      """Decoder."""

      def __init__(self, output_shape):
        super(Decoder, self).__init__()
        self._sub_layers = [
            tfkl.Dense(10, activation='relu'),
            tfkl.Dense(tfpl.IndependentBernoulli.params_size(output_shape)),
            tfpl.IndependentBernoulli(output_shape, tfd.Bernoulli.logits),
        ]

      def call(self, inputs):
        return functools.reduce(lambda x, f: f(x), self._sub_layers, inputs)

    encoder = Encoder(self.input_shape, self.encoded_size, self.train_size)
    decoder = Decoder(self.input_shape)

    images = tfkl.Input(shape=self.input_shape)
    encoded = encoder(images)
    decoded = decoder(encoded)

    vae_model = tfk.Model(inputs=images, outputs=decoded)
    vae_model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=lambda x, rv_x: -rv_x.log_prob(x),
        metrics=[])
    vae_model.fit(self.x, self.x,
                  batch_size=25,
                  epochs=1,
                  validation_data=(self.x_test, self.x_test))
    yhat = vae_model(tf.convert_to_tensor(self.x_test))
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
            tfd.MultivariateNormalDiag(
                loc=tf.Variable(tf.zeros([self.encoded_size]))),
            weight=0.9),  # "beta" as in beta-VAE.
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
        optimizer=tf.optimizers.Adam(),
        loss=lambda x, rv_x: -rv_x.log_prob(x),
        metrics=[])
    self.assertLen(encoder_model.trainable_variables, (2 + 2) + 1)
    self.assertLen(decoder_model.trainable_variables, 2 + 2)
    self.assertLen(vae_model.trainable_variables, (2 + 2) + (2 + 2) + 1)
    vae_model.fit(self.x, self.x,
                  batch_size=25,
                  epochs=1,
                  steps_per_epoch=1,  # Usually `n // batch_size`.
                  validation_data=(self.x_test, self.x_test))
    yhat = vae_model(tf.convert_to_tensor(self.x_test))
    self.assertIsInstance(yhat, tfd.Independent)
    self.assertIsInstance(yhat.distribution, tfd.Bernoulli)

  def test_side_variable_is_auto_tracked(self):
    # `s` is the "side variable".
    s = tfp.util.TransformedVariable(1., tfb.Softplus())
    prior = tfd.Normal(tf.Variable(0.), 1.)
    linear_regression = tf.keras.Sequential([
        tf.keras.layers.Dense(1),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(t, s),
            activity_regularizer=tfpl.KLDivergenceRegularizer(prior)),
    ])
    linear_regression.build(tf.TensorShape([1, 3]))
    self.assertLen(linear_regression.trainable_variables, 4)
    self.assertIn(id(s.pretransformed_input),
                  [id(x) for x in linear_regression.trainable_variables])
    self.assertIn(id(prior.loc),
                  [id(x) for x in linear_regression.trainable_variables])


@test_util.test_graph_and_eager_modes
class DistributionLambdaSerializationTest(test_util.TestCase):

  def assertSerializable(self, model, batch_size=1):
    """Assert that a model can be saved/loaded via Keras Model.save/load_model.

    Arguments:
      model: A Keras model that outputs a `tfd.Distribution`.
      batch_size: The batch size to use when checking that the model produces
        the same results as a serialized/deserialized copy.  Default value: 1.
    """
    batch_shape = [batch_size]

    input_shape = batch_shape + model.input.shape[1:].as_list()
    dtype = model.input.dtype.as_numpy_dtype

    model_file = self.create_tempfile()
    model.save(model_file.full_path, save_format='h5')
    model_copy = tfk.models.load_model(model_file.full_path)

    x = np.random.uniform(-3., 3., input_shape).astype(dtype)

    model_x_mean = self.evaluate(model(x).mean())
    self.assertAllEqual(model_x_mean, self.evaluate(model_copy(x).mean()))

    output_shape = model_x_mean.shape
    y = np.random.uniform(0., 1., output_shape).astype(dtype)

    self.assertAllEqual(self.evaluate(model(x).log_prob(y)),
                        self.evaluate(model_copy(x).log_prob(y)))

  def assertExportable(self, model, batch_size=1):
    """Assert a Keras model supports export_saved_model/load_from_saved_model.

    Arguments:
      model: A Keras model with Tensor output.
      batch_size: The batch size to use when checking that the model produces
        the same results as a serialized/deserialized copy.  Default value: 1.
    """
    batch_shape = [batch_size]

    input_shape = batch_shape + model.input.shape[1:].as_list()
    dtype = model.input.dtype.as_numpy_dtype

    model_dir = self.create_tempdir()
    tf1.keras.experimental.export_saved_model(model, model_dir.full_path)
    model_copy = tf1.keras.experimental.load_from_saved_model(
        model_dir.full_path)

    x = np.random.uniform(-3., 3., input_shape).astype(dtype)
    self.assertAllEqual(self.evaluate(model(x)), self.evaluate(model_copy(x)))
    self.assertAllEqual(model.predict(x), model_copy.predict(x))

  def test_serialization(self):
    model = tfk.Sequential([
        tfkl.Dense(2, input_shape=(5,)),
        # pylint: disable=g-long-lambda
        tfpl.DistributionLambda(lambda t: tfd.Normal(
            loc=t[..., 0:1], scale=tf.exp(t[..., 1:2])))
    ])
    self.assertSerializable(model)

    model = tfk.Sequential([
        tfkl.Dense(2, input_shape=(5,)),
        # pylint: disable=g-long-lambda
        tfpl.DistributionLambda(lambda t: tfd.Normal(
            loc=t[..., 0:1], scale=tf.exp(t[..., 1:2]))),
        tfkl.Lambda(lambda d: d.mean() + d.stddev())
    ])
    self.assertExportable(model, batch_size=4)

  @staticmethod
  def _make_distribution(t):
    return tfpl.MixtureSameFamily.new(t, 3, tfpl.IndependentNormal([2]))

  def test_serialization_static_method(self):
    model = tfk.Sequential([
        tfkl.Dense(15, input_shape=(5,)),
        tfpl.DistributionLambda(
            # pylint: disable=unnecessary-lambda
            lambda t: DistributionLambdaSerializationTest._make_distribution(t))
    ])
    model.compile(optimizer='adam', loss='mse')
    # TODO(b/138375951): Re-enable this test.
    # self.assertSerializable(model, batch_size=3)

    model = tfk.Sequential([
        tfkl.Dense(15, input_shape=(5,)),
        tfpl.DistributionLambda(
            DistributionLambdaSerializationTest._make_distribution,
            convert_to_tensor_fn=tfd.Distribution.mean),
        tfkl.Lambda(lambda x: x + 1.)
    ])
    model.compile(optimizer='adam', loss='mse')
    self.assertExportable(model)

  def test_serialization_closure_over_lambdas_tensors_and_numpy_array(self):
    num_components = np.array(3)
    one = tf.convert_to_tensor(1)
    mk_ind_norm = lambda event_shape: tfpl.IndependentNormal(event_shape + one)
    def make_distribution(t):
      return tfpl.MixtureSameFamily.new(
          t, num_components, mk_ind_norm(1))

    model = tfk.Sequential([
        tfkl.Dense(15, input_shape=(5,)),
        tfpl.DistributionLambda(make_distribution)
    ])
    self.assertSerializable(model, batch_size=4)

    model = tfk.Sequential([
        tfkl.Dense(15, input_shape=(5,)),
        # pylint: disable=unnecessary-lambda
        tfpl.DistributionLambda(lambda t: make_distribution(t)),
        tfkl.Lambda(lambda d: d.mean() + d.stddev())
    ])
    self.assertExportable(model, batch_size=2)


@test_util.test_graph_and_eager_modes
class DistributionLambdaVariableCreation(test_util.TestCase):

  def test_variable_creation(self):
    conv1 = tfkl.Convolution2D(filters=1, kernel_size=[1, 3])
    conv2 = tfkl.Convolution2D(filters=1, kernel_size=[2, 1])
    pad1 = tfkl.ZeroPadding2D(padding=((0, 0), (1, 1)))
    pad2 = tfkl.ZeroPadding2D(padding=((1, 0), (0, 0)))

    loc = tfk.Sequential([conv1, pad1])
    scale = tfk.Sequential([conv2, pad2])

    x = tfkl.Input(shape=(3, 3, 1))

    normal = tfpl.DistributionLambda(
        lambda x: tfd.Normal(loc=loc(x), scale=tf.exp(scale(x))))
    normal._loc_net = loc
    normal._scale_net = scale

    model = tfk.Model(x, normal(x))  # pylint: disable=unused-variable
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=lambda x, rv_x: -rv_x.log_prob(x),
        metrics=[])

    x_train = np.random.rand(1000, 3, 3, 1).astype(np.float32)
    x_test = np.random.rand(100, 3, 3, 1).astype(np.float32)
    model.fit(x_train, x_train,
              batch_size=25,
              epochs=5,
              steps_per_epoch=10,
              validation_data=(x_test, x_test))


@test_util.test_graph_and_eager_modes
class KLDivergenceAddLossTest(test_util.TestCase):

  def test_approx_kl(self):
    # TODO(b/120320323): Enable this test in eager.
    if tf.executing_eagerly():
      self.skipTest('KLDivergenceAddLossTest disabled for Eager (b/120320323).')

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
        [loc, tfb.FillScaleTriL().inverse(scale_tril)], axis=0)[tf.newaxis]

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
        optimizer=tf.optimizers.Adam(),
        loss=lambda x, dist: -dist.log_prob(x[0, :event_size]),
        metrics=[])
    model.fit(x, x,
              batch_size=25,
              epochs=1,
              steps_per_epoch=1)  # Usually `n // batch_size`.

  def test_use_exact_kl(self):
    # TODO(b/120320323): Enable this test in eager.
    if tf.executing_eagerly():
      self.skipTest('KLDivergenceAddLossTest disabled for Eager (b/120320323).')

    event_size = 2
    prior = tfd.MultivariateNormalDiag(loc=tf.zeros(event_size))

    # Use a small number of samples because we want to verify that
    # we calculated the exact KL divergence and not the one from sampling.
    model = tfk.Sequential([
        tfpl.MultivariateNormalTriL(event_size,
                                    lambda s: s.sample(3, seed=42)),
        tfpl.KLDivergenceAddLoss(prior, use_exact_kl=True),
    ])

    loc = [-1., 1.]
    scale_tril = [[1.1, 0.],
                  [0.2, 1.3]]
    actual_kl = tfd.kl_divergence(
        tfd.MultivariateNormalTriL(loc, scale_tril), prior)

    x = tf.concat(
        [loc, tfb.FillScaleTriL().inverse(scale_tril)], axis=0)[tf.newaxis]

    y = model(x)
    self.assertEqual(1, len(model.losses))
    y = model(x)
    self.assertEqual(2, len(model.losses))

    [loc_, scale_tril_, actual_kl_, evaluated_kl_] = self.evaluate([
        y.loc, y.scale.to_dense(), actual_kl, model.losses[0]])

    self.assertAllClose([loc], loc_, atol=0., rtol=1e-5)
    self.assertAllClose([scale_tril], scale_tril_, atol=0., rtol=1e-5)
    self.assertNear(actual_kl_, evaluated_kl_, err=1e-5)

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=lambda x, dist: -dist.log_prob(x[0, :event_size]),
        metrics=[])
    model.fit(x, x,
              batch_size=25,
              epochs=1,
              steps_per_epoch=1)  # Usually `n // batch_size`.


@test_util.test_graph_and_eager_modes
class MultivariateNormalTriLTest(test_util.TestCase):

  def _check_distribution(self, t, x):
    self.assertIsInstance(x, tfd.MultivariateNormalTriL)
    t_back = tf.concat([
        x.loc, tfb.FillScaleTriL().inverse(x.scale.to_dense())], axis=-1)
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
        optimizer=tf.optimizers.Adam(),
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


@test_util.test_graph_and_eager_modes
class OneHotCategoricalTest(test_util.TestCase):

  def _check_distribution(self, t, x):
    self.assertIsInstance(x, tfd.OneHotCategorical)
    [t_, x_logits_, x_probs_, mean_] = self.evaluate([
        t, x.logits_parameter(), x.probs_parameter(), x.mean()])
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
        optimizer=tf.optimizers.Adam(learning_rate=0.5),
        loss=lambda y, model: -model.log_prob(y),
        metrics=[])
    batch_size = 100
    model.fit(x, y,
              batch_size=batch_size,
              epochs=1,
              steps_per_epoch=n // batch_size,
              shuffle=True)


@test_util.test_graph_and_eager_modes
class CategoricalMixtureOfOneHotCategoricalTest(test_util.TestCase):

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
        tf.reduce_mean(x.sample(int(10e3), seed=42), axis=0),
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
        optimizer=tf.optimizers.Adam(learning_rate=0.5),
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


@test_util.test_graph_and_eager_modes
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
    return tf1.placeholder_with_default(
        ndarray, shape=ndarray.shape if self.use_static_shape else None)

  def _check_distribution(self, t, x, batch_shape):
    self.assertIsInstance(x, tfd.Independent)
    self.assertIsInstance(x.distribution, self.dist_class)
    self.assertEqual(self.dtype, x.dtype)
    t_back = self._distribution_to_params(x.distribution, batch_shape)
    [t_, t_back_] = self.evaluate([t, t_back])
    self.assertAllClose(t_, t_back_, atol=1e-6, rtol=1e-5)
    self.assertEqual(self.dtype, t_back_.dtype)

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

    layer = self.layer_class(validate_args=True, dtype=self.dtype)
    x = layer(t)
    self._check_distribution(t, x, batch_shape)

  def test_serialization(self):
    event_shape = []
    params_size = self.layer_class.params_size(event_shape)
    batch_shape = [4, 1]

    low = self._build_tensor(-3., dtype=self.dtype)
    high = self._build_tensor(3., dtype=self.dtype)
    x = self.evaluate(tfd.Uniform(low, high).sample(
        batch_shape + [params_size], seed=42))

    model = tfk.Sequential([
        tfkl.Dense(params_size, input_shape=(params_size,), dtype=self.dtype),
        self.layer_class(event_shape, validate_args=True, dtype=self.dtype),
    ])

    model_file = self.create_tempfile()
    model.save(model_file.full_path, save_format='h5')
    model_copy = tfk.models.load_model(model_file.full_path)

    self.assertAllEqual(self.evaluate(model(x).mean()),
                        self.evaluate(model_copy(x).mean()))

    self.assertEqual(self.dtype, model(x).mean().dtype.as_numpy_dtype)

    ones = np.ones([7] + batch_shape + event_shape, dtype=self.dtype)
    self.assertAllEqual(self.evaluate(model(x).log_prob(ones)),
                        self.evaluate(model_copy(x).log_prob(ones)))

  def test_model_export(self):
    event_shape = [3, 2]
    params_size = self.layer_class.params_size(event_shape)
    batch_shape = [4]

    low = self._build_tensor(-3., dtype=self.dtype)
    high = self._build_tensor(3., dtype=self.dtype)
    x = self.evaluate(tfd.Uniform(low, high).sample(
        batch_shape + [params_size], seed=42))

    model = tfk.Sequential([
        tfkl.Dense(params_size, input_shape=(params_size,), dtype=self.dtype),
        self.layer_class(event_shape, validate_args=True,
                         convert_to_tensor_fn='mean', dtype=self.dtype),
        # NOTE: For TensorFlow to be able to serialize the graph (i.e., for
        # serving), the model must output a Tensor and not a Distribution.
        tfkl.Dense(1, dtype=self.dtype),
    ])
    model.compile(optimizer='adam', loss='mse')

    model_dir = self.create_tempdir()
    tf1.keras.experimental.export_saved_model(model, model_dir.full_path)
    model_copy = tf1.keras.experimental.load_from_saved_model(
        model_dir.full_path)

    self.assertAllEqual(self.evaluate(model(x)), self.evaluate(model_copy(x)))
    self.assertEqual(self.dtype, model(x).dtype.as_numpy_dtype)


@test_util.test_graph_and_eager_modes
class _IndependentBernoulliTest(_IndependentLayerTest):
  layer_class = tfpl.IndependentBernoulli
  dist_class = tfd.Bernoulli

  def _distribution_to_params(self, distribution, batch_shape):
    return tf.reshape(distribution.logits,
                      tf.concat([batch_shape, [-1]], axis=-1))


@test_util.test_graph_and_eager_modes
class IndependentBernoulliTestDynamicShape(test_util.TestCase,
                                           _IndependentBernoulliTest):
  dtype = np.float64
  use_static_shape = False


@test_util.test_graph_and_eager_modes
class IndependentBernoulliTestStaticShape(test_util.TestCase,
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
        optimizer=tf.optimizers.Adam(learning_rate=0.5),
        loss=lambda y, model: -model.log_prob(y))
    batch_size = 10000
    model.fit(x, y,
              batch_size=batch_size,
              epochs=100,
              shuffle=True)
    self.assertAllClose(scale_tril, model.get_weights()[0],
                        atol=0.15, rtol=0.15)
    self.assertAllClose([0., 0.], model.get_weights()[1],
                        atol=0.15, rtol=0.15)


@test_util.test_graph_and_eager_modes
class _IndependentLogisticTest(_IndependentLayerTest):
  layer_class = tfpl.IndependentLogistic
  dist_class = tfd.Logistic

  def _distribution_to_params(self, distribution, batch_shape):
    return tf.concat([
        tf.reshape(distribution.loc, tf.concat([batch_shape, [-1]], axis=-1)),
        tfp.math.softplus_inverse(tf.reshape(
            distribution.scale, tf.concat([batch_shape, [-1]], axis=-1)))
    ], -1)


@test_util.test_graph_and_eager_modes
class IndependentLogisticTestDynamicShape(test_util.TestCase,
                                          _IndependentLogisticTest):
  dtype = np.float32
  use_static_shape = False


@test_util.test_graph_and_eager_modes
class IndependentLogisticTestStaticShape(test_util.TestCase,
                                         _IndependentLogisticTest):
  dtype = np.float64
  use_static_shape = True

  def test_doc_string(self):
    input_shape = [28, 28, 1]
    encoded_shape = 2
    encoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=input_shape, dtype=self.dtype),
        tfkl.Flatten(dtype=self.dtype),
        tfkl.Dense(10, activation='relu', dtype=self.dtype),
        tfkl.Dense(tfpl.IndependentLogistic.params_size(encoded_shape),
                   dtype=self.dtype),
        tfpl.IndependentLogistic(encoded_shape, dtype=self.dtype),
        tfkl.Lambda(lambda x: x + 0.,  # To force conversion to tensor.
                    dtype=self.dtype)
    ])

    # Test that we can run the model and get a sample.
    x = np.random.randn(*([1] + input_shape)).astype(self.dtype)
    self.assertEqual((1, 2), encoder.predict_on_batch(x).shape)

    out = encoder(tf.convert_to_tensor(x))
    self.assertEqual((1, 2), out.shape)
    self.assertEqual((1, 2), self.evaluate(out).shape)
    self.assertEqual(self.dtype, out.dtype)


@test_util.test_graph_and_eager_modes
class _IndependentNormalTest(_IndependentLayerTest):
  layer_class = tfpl.IndependentNormal
  dist_class = tfd.Normal

  def _distribution_to_params(self, distribution, batch_shape):
    return tf.concat([
        tf.reshape(distribution.loc, tf.concat([batch_shape, [-1]], axis=-1)),
        tfp.math.softplus_inverse(tf.reshape(
            distribution.scale, tf.concat([batch_shape, [-1]], axis=-1)))
    ], -1)

  def test_keras_sequential_with_unknown_input_size(self):
    input_shape = [28, 28, 1]
    encoded_shape = self._build_tensor([2], dtype=np.int32)
    params_size = tfpl.IndependentNormal.params_size(encoded_shape)

    def reshape(x):
      return tf.reshape(
          x, tf.concat([tf.shape(x)[:-1], [-1, params_size]], 0))

    # Test a Sequential model where the input to IndependentNormal does not have
    # a statically-known shape.
    encoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=input_shape, dtype=self.dtype),
        tfkl.Flatten(dtype=self.dtype),
        tfkl.Dense(12, activation='relu', dtype=self.dtype),
        tfkl.Lambda(reshape, dtype=self.dtype),
        # When encoded_shape/params_size are placeholders, the input to the
        # IndependentNormal has shape (?, ?, ?) or (1, ?, ?), depending on
        # whether or not encoded_shape's shape is known.
        tfpl.IndependentNormal(encoded_shape, dtype=self.dtype),
        tfkl.Lambda(lambda x: x + 0.,  # To force conversion to tensor.
                    dtype=self.dtype)
    ])

    x = np.random.randn(*([1] + input_shape)).astype(self.dtype)
    self.assertEqual((1, 3, 2), encoder.predict_on_batch(x).shape)

    out = encoder(tf.convert_to_tensor(x))
    if tf.executing_eagerly():
      self.assertEqual((1, 3, 2), out.shape)
    elif self.use_static_shape:
      self.assertEqual([1, None, None], out.shape.as_list())
    self.assertEqual((1, 3, 2), self.evaluate(out).shape)
    self.assertEqual(self.dtype, out.dtype)


@test_util.test_graph_and_eager_modes
class IndependentNormalTestDynamicShape(test_util.TestCase,
                                        _IndependentNormalTest):
  dtype = np.float32
  use_static_shape = False


@test_util.test_graph_and_eager_modes
class IndependentNormalTestStaticShape(test_util.TestCase,
                                       _IndependentNormalTest):
  dtype = np.float64
  use_static_shape = True

  def test_doc_string(self):
    input_shape = [28, 28, 1]
    encoded_shape = 2
    encoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=input_shape, dtype=self.dtype),
        tfkl.Flatten(dtype=self.dtype),
        tfkl.Dense(10, activation='relu', dtype=self.dtype),
        tfkl.Dense(tfpl.IndependentNormal.params_size(encoded_shape),
                   dtype=self.dtype),
        tfpl.IndependentNormal(encoded_shape, dtype=self.dtype),
        tfkl.Lambda(lambda x: x + 0.,  # To force conversion to tensor.
                    dtype=self.dtype)
    ])

    # Test that we can run the model and get a sample.
    x = np.random.randn(*([1] + input_shape)).astype(self.dtype)
    self.assertEqual((1, 2), encoder.predict_on_batch(x).shape)

    out = encoder(tf.convert_to_tensor(x))
    self.assertEqual((1, 2), out.shape)
    self.assertEqual((1, 2), self.evaluate(out).shape)
    self.assertEqual(self.dtype, out.dtype)


@test_util.test_graph_and_eager_modes
class _IndependentPoissonTest(_IndependentLayerTest):
  layer_class = tfpl.IndependentPoisson
  dist_class = tfd.Poisson

  def _distribution_to_params(self, distribution, batch_shape):
    return tf.reshape(distribution.log_rate,
                      tf.concat([batch_shape, [-1]], axis=-1))


@test_util.test_graph_and_eager_modes
class IndependentPoissonTestDynamicShape(test_util.TestCase,
                                         _IndependentPoissonTest):
  dtype = np.float32
  use_static_shape = False


@test_util.test_graph_and_eager_modes
class IndependentPoissonTestStaticShape(test_util.TestCase,
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
        tfkl.Dense(tfpl.IndependentPoisson.params_size(1), dtype=self.dtype),
        tfpl.IndependentPoisson(1, dtype=self.dtype)
    ])

    # Fit.
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.05),
        loss=lambda y, model: -model.log_prob(y),
        metrics=[])
    batch_size = 50

    model.fit(x, y,
              batch_size=batch_size,
              epochs=1,
              steps_per_epoch=1,  # Usually `n // batch_size`.
              verbose=True,
              shuffle=True)


@test_util.test_graph_and_eager_modes
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
    return tf1.placeholder_with_default(
        ndarray, shape=ndarray.shape if self.use_static_shape else None)

  def _check_distribution(self, t, x, batch_shape):
    self.assertIsInstance(x, tfd.MixtureSameFamily)
    self.assertIsInstance(x.mixture_distribution, tfd.Categorical)
    self.assertIsInstance(x.components_distribution, tfd.Independent)
    self.assertIsInstance(x.components_distribution.distribution,
                          self.dist_class)
    self.assertEqual(self.dtype, x.dtype)

    t_back = self._distribution_to_params(x, batch_shape)
    [t_, t_back_] = self.evaluate([t, t_back])
    self.assertAllClose(t_, t_back_, atol=1e-6, rtol=1e-5)
    self.assertEqual(self.dtype, t_back_.dtype)

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

    layer = self.layer_class(n, event_shape, validate_args=True,
                             dtype=self.dtype)
    x = layer(t)
    self._check_distribution(t, x, batch_shape)

  def test_serialization(self):
    n = 3
    event_shape = []
    params_size = self.layer_class.params_size(n, event_shape)
    batch_shape = [4, 1]

    low = self._build_tensor(-3., dtype=self.dtype)
    high = self._build_tensor(3., dtype=self.dtype)
    x = self.evaluate(tfd.Uniform(low, high).sample(
        batch_shape + [params_size], seed=42))

    model = tfk.Sequential([
        tfkl.Dense(params_size, input_shape=(params_size,), dtype=self.dtype),
        self.layer_class(n, event_shape, validate_args=True, dtype=self.dtype),
    ])

    model_file = self.create_tempfile()
    model.save(model_file.full_path, save_format='h5')
    model_copy = tfk.models.load_model(model_file.full_path)

    self.assertAllEqual(self.evaluate(model(x).mean()),
                        self.evaluate(model_copy(x).mean()))

    self.assertEqual(self.dtype, model(x).mean().dtype.as_numpy_dtype)

    ones = np.ones([7] + batch_shape + event_shape, dtype=self.dtype)
    self.assertAllEqual(self.evaluate(model(x).log_prob(ones)),
                        self.evaluate(model_copy(x).log_prob(ones)))

  def test_model_export(self):
    n = 5
    event_shape = [3, 2]
    params_size = self.layer_class.params_size(n, event_shape)
    batch_shape = [4]

    low = self._build_tensor(-3., dtype=self.dtype)
    high = self._build_tensor(3., dtype=self.dtype)
    x = self.evaluate(tfd.Uniform(low, high).sample(
        batch_shape + [params_size], seed=42))

    model = tfk.Sequential([
        tfkl.Dense(params_size, input_shape=(params_size,), dtype=self.dtype),
        self.layer_class(n, event_shape, validate_args=True,
                         convert_to_tensor_fn='mean', dtype=self.dtype),
        # NOTE: For TensorFlow to be able to serialize the graph (i.e., for
        # serving), the model must output a Tensor and not a Distribution.
        tfkl.Dense(1, dtype=self.dtype),
    ])
    model.compile(optimizer='adam', loss='mse')

    model_dir = self.create_tempdir()
    tf1.keras.experimental.export_saved_model(model, model_dir.full_path)
    model_copy = tf1.keras.experimental.load_from_saved_model(
        model_dir.full_path)

    self.assertAllEqual(self.evaluate(model(x)), self.evaluate(model_copy(x)))
    self.assertEqual(self.dtype, model(x).dtype.as_numpy_dtype)


@test_util.test_graph_and_eager_modes
class _MixtureLogisticTest(_MixtureLayerTest):
  layer_class = tfpl.MixtureLogistic
  dist_class = tfd.Logistic

  def _distribution_to_params(self, distribution, batch_shape):
    """Given a self.layer_class instance, return a tensor of its parameters."""
    params_shape = tf.concat([batch_shape, [-1]], axis=0)
    batch_and_n_shape = tf.concat(
        [tf.shape(distribution.mixture_distribution.logits), [-1]],
        axis=0)
    cd = distribution.components_distribution.distribution
    return tf.concat([
        distribution.mixture_distribution.logits,
        tf.reshape(tf.concat([
            tf.reshape(cd.loc, batch_and_n_shape),
            tf.reshape(tfp.math.softplus_inverse(cd.scale), batch_and_n_shape)
        ], axis=-1), params_shape),
    ], axis=-1)

  def test_doc_string(self):
    # Load data (graph of a cardioid).
    n = 2000
    t = self.evaluate(tfd.Uniform(low=-np.pi, high=np.pi).sample([n, 1]))
    r = 2 * (1 - tf.cos(t))
    x = tf.convert_to_tensor(self.evaluate(
        r * tf.sin(t) + tfd.Normal(loc=0., scale=0.1).sample([n, 1])))
    y = tf.convert_to_tensor(self.evaluate(
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
        optimizer=tf.optimizers.Adam(learning_rate=0.02),
        loss=lambda y, model: -model.log_prob(y))
    model.fit(x, y,
              batch_size=batch_size,
              epochs=1,
              steps_per_epoch=n // batch_size)

    self.assertEqual(15, self.evaluate(tf.convert_to_tensor(params_size)))


@test_util.test_graph_and_eager_modes
class MixtureLogisticTestDynamicShape(test_util.TestCase,
                                      _MixtureLogisticTest):
  dtype = np.float64
  use_static_shape = False


@test_util.test_graph_and_eager_modes
class MixtureLogisticTestStaticShape(test_util.TestCase,
                                     _MixtureLogisticTest):
  dtype = np.float32
  use_static_shape = True


@test_util.test_graph_and_eager_modes
class _MixtureNormalTest(_MixtureLayerTest):
  layer_class = tfpl.MixtureNormal
  dist_class = tfd.Normal

  def _distribution_to_params(self, distribution, batch_shape):
    """Given a self.layer_class instance, return a tensor of its parameters."""
    params_shape = tf.concat([batch_shape, [-1]], axis=0)
    batch_and_n_shape = tf.concat(
        [tf.shape(distribution.mixture_distribution.logits), [-1]],
        axis=0)
    cd = distribution.components_distribution.distribution
    return tf.concat([
        distribution.mixture_distribution.logits,
        tf.reshape(tf.concat([
            tf.reshape(cd.loc, batch_and_n_shape),
            tf.reshape(tfp.math.softplus_inverse(cd.scale), batch_and_n_shape)
        ], axis=-1), params_shape),
    ], axis=-1)

  def test_doc_string(self):
    # Load data (graph of a cardioid).
    n = 2000
    t = self.evaluate(tfd.Uniform(low=-np.pi, high=np.pi).sample([n, 1]))
    r = 2 * (1 - tf.cos(t))
    x = tf.convert_to_tensor(self.evaluate(
        r * tf.sin(t) + tfd.Normal(loc=0., scale=0.1).sample([n, 1])))
    y = tf.convert_to_tensor(self.evaluate(
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
        optimizer=tf.optimizers.Adam(learning_rate=0.02),
        loss=lambda y, model: -model.log_prob(y))
    model.fit(x, y,
              batch_size=batch_size,
              epochs=1,
              steps_per_epoch=n // batch_size)

    self.assertEqual(15, self.evaluate(tf.convert_to_tensor(params_size)))


@test_util.test_graph_and_eager_modes
class MixtureNormalTestDynamicShape(test_util.TestCase,
                                    _MixtureNormalTest):
  dtype = np.float32
  use_static_shape = False


@test_util.test_graph_and_eager_modes
class MixtureNormalTestStaticShape(test_util.TestCase,
                                   _MixtureNormalTest):
  dtype = np.float64
  use_static_shape = True


@test_util.test_graph_and_eager_modes
class _MixtureSameFamilyTest(object):

  def _build_tensor(self, ndarray, dtype=None):
    # Enforce parameterized dtype and static/dynamic testing.
    ndarray = np.asarray(ndarray).astype(
        dtype if dtype is not None else self.dtype)
    return tf1.placeholder_with_default(
        ndarray, shape=ndarray.shape if self.use_static_shape else None)

  def _check_distribution(self, t, x, batch_shape):
    self.assertIsInstance(x, tfd.MixtureSameFamily)
    self.assertIsInstance(x.mixture_distribution, tfd.Categorical)
    self.assertIsInstance(x.components_distribution, tfd.MultivariateNormalTriL)

    shape = tf.concat([batch_shape, [-1]], axis=0)
    batch_and_n_shape = tf.concat(
        [tf.shape(x.mixture_distribution.logits), [-1]], axis=0)
    cd = x.components_distribution
    scale_tril = tfb.FillScaleTriL(diag_shift=np.array(1e-5, self.dtype))
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
    normal = tfpl.MultivariateNormalTriL(event_size, validate_args=True,
                                         dtype=self.dtype)
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

    normal = tfpl.MultivariateNormalTriL(event_size, validate_args=True,
                                         dtype=self.dtype)
    layer = tfpl.MixtureSameFamily(n, normal, validate_args=True,
                                   dtype=self.dtype)
    t = tfd.Uniform(low, high).sample(tf.concat([batch_shape, [p]], 0), seed=42)
    x = layer(t)
    self._check_distribution(t, x, batch_shape)

  def test_doc_string(self):
    # Load data (graph of a cardioid).
    n = 2000
    t = self.evaluate(tfd.Uniform(low=-np.pi, high=np.pi).sample([n, 1]))
    r = 2 * (1 - tf.cos(t))
    x = tf.convert_to_tensor(self.evaluate(
        r * tf.sin(t) + tfd.Normal(loc=0., scale=0.1).sample([n, 1])))
    y = tf.convert_to_tensor(self.evaluate(
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
        optimizer=tf.optimizers.Adam(learning_rate=0.02),
        loss=lambda y, model: -model.log_prob(y))
    model.fit(x, y,
              batch_size=batch_size,
              epochs=1,
              steps_per_epoch=1)  # Usually `n // batch_size`.

    self.assertEqual(15, self.evaluate(tf.convert_to_tensor(params_size)))


@test_util.test_graph_and_eager_modes
class MixtureSameFamilyTestDynamicShape(test_util.TestCase,
                                        _MixtureSameFamilyTest):
  dtype = np.float32
  use_static_shape = False


@test_util.test_graph_and_eager_modes
class MixtureSameFamilyTestStaticShape(test_util.TestCase,
                                       _MixtureSameFamilyTest):
  dtype = np.float64
  use_static_shape = True


@test_util.test_graph_and_eager_modes
class VariationalGaussianProcessEndToEnd(test_util.TestCase):

  def testEndToEnd(self):
    np.random.seed(43)
    dtype = np.float64

    n = 1000
    w0 = 0.125
    b0 = 5.
    x_range = [-20, 60]

    def s(x):
      g = (x - x_range[0]) / (x_range[1] - x_range[0])
      return 3*(0.25 + g**2.)

    x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
    eps = np.random.randn(n) * s(x)
    y = (w0 * x * (1 + np.sin(x)) + b0) + eps
    x0 = np.linspace(*x_range, num=1000)

    class KernelFn(tf.keras.layers.Layer):

      def __init__(self, **kwargs):
        super(KernelFn, self).__init__(**kwargs)

        self._amplitude = self.add_weight(
            initializer=tf.initializers.constant(.54),
            dtype=dtype,
            name='amplitude')

      def call(self, x):
        return x

      @property
      def kernel(self):
        return tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=tf.nn.softplus(self._amplitude))

    num_inducing_points = 50

    # Add a leading dimension for the event_shape.
    eyes = np.expand_dims(np.eye(num_inducing_points), 0)
    variational_inducing_observations_scale_initializer = (
        tf.initializers.constant(1e-3 * eyes))

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[1], dtype=dtype),
        tf.keras.layers.Dense(1, kernel_initializer='Ones', use_bias=False,
                              activation=None, dtype=dtype),
        tfp.layers.VariationalGaussianProcess(
            num_inducing_points=num_inducing_points,
            kernel_provider=KernelFn(dtype=dtype),
            inducing_index_points_initializer=(
                tf.initializers.constant(
                    np.linspace(*x_range,
                                num=num_inducing_points,
                                dtype=dtype)[..., np.newaxis])),
            variational_inducing_observations_scale_initializer=(
                variational_inducing_observations_scale_initializer)),
    ])

    batch_size = 64
    kl_weight = np.float64(batch_size) / n
    loss = lambda y, d: d.variational_loss(y, kl_weight=kl_weight)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.02),
        loss=loss)

    if not tf.executing_eagerly():
      self.evaluate([v.initializer for v in model.variables])

    # This should have no issues
    model.fit(x, y, epochs=5, batch_size=batch_size, verbose=False)

    vgp = model(x0[..., tf.newaxis])
    num_samples = 7
    samples_ = self.evaluate(vgp.sample(num_samples))
    self.assertAllEqual(samples_.shape, (7, 1000, 1))
    self.assertEqual(dtype, vgp.dtype)


@test_util.test_graph_and_eager_modes
class JointDistributionLayer(test_util.TestCase):

  def test_works(self):
    x = tf.keras.Input(shape=())
    y = tfp.layers.VariableLayer(shape=[2, 4, 3], dtype=tf.float32)(x)
    y = tf.keras.layers.Dense(5, use_bias=False)(y)
    y = tfp.layers.DistributionLambda(
        lambda t: tfd.JointDistributionSequential([  # pylint: disable=g-long-lambda
            tfd.Gamma(t[..., 0], t[..., 1]),
            tfd.Normal(t[..., 2], 1),
            lambda m, s: tfd.Normal(loc=m, scale=s),
        ]),
    )(y)
    m = tf.keras.Model(x, y)
    dist = m(0.)
    self.assertIsInstance(dist, tfd.JointDistributionSequential)
    # Check the weights.
    self.assertEqual(2, len(m.weights))


if __name__ == '__main__':
  tf.test.main()
