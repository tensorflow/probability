"""Builds a Bayesian VGG16 Model"""

import tensorflow as tf
import tensorflow_probability as tfp


def BayesianVGG(input_dim,
                num_classes=10,
                kernel_posterior_mean=-9,
                kernel_posterior_stddev=0.1,
                kernel_posterior_constraint=0.2):
  """
  Args:
    input_dim: A Numpy `array` of IMAGE_SHAPE.
    kernel_posterior_mean: Python `int` number for the initial kernel
      posterior log mean. The smaller the mean the closer is the
      initialization to a deterministic network.
    kernel_posterior_stddev: Python `float` number for the initial kernel
      posterior stddev.
      i.e. log_var ~ N(kernel_posterior_mean, kernel_posterior_stddev)
    kernel_posterior_constraint: Python `float` number for the log value
      to constrain the log variance throughout training.
      i.e. log_var <= log(kernel_posterior_constraint).
  """

  filters = [64, 128, 256, 512, 512]
  kernels = [3, 3, 3, 3, 3]
  strides = [2, 2, 2, 2, 2]

  kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(
      untransformed_scale_initializer=tf.random_normal_initializer(
          mean=kernel_posterior_mean,
          stddev=kernel_posterior_stddev),
      untransformed_scale_constraint=lambda t: tf.clip_by_value(
          t, -1000, tf.log(kernel_posterior_constraint)))

  image = tf.keras.layers.Input(shape=input_dim, dtype='float32')

  x = image
  for i in range(len(kernels)):
    x = _vggconv_block(
        x,
        filters[i],
        kernels[i],
        strides[i],
        kernel_posterior_fn)

  x = tf.keras.layers.Flatten()(x)
  x = tfp.layers.DenseFlipout(
      num_classes,
      kernel_posterior_fn=kernel_posterior_fn)(x)
  model = tf.keras.Model(inputs=image, outputs=x, name='vgg16')
  return model


def _vggconv_block(x, filters, kernel, stride, kernel_posterior_fn):
  out = tfp.layers.Convolution2DFlipout(
      filters,
      kernel,
      padding='same',
      kernel_posterior_fn=kernel_posterior_fn)(x)
  out = tf.keras.layers.BatchNormalization()(out)
  out = tf.keras.layers.Activation('relu')(out)

  out = tfp.layers.Convolution2DFlipout(
      filters,
      kernel,
      padding='same',
      kernel_posterior_fn=kernel_posterior_fn)(out)
  out = tf.keras.layers.BatchNormalization()(out)
  out = tf.keras.layers.Activation('relu')(out)

  out = tf.keras.layers.MaxPooling2D(
      pool_size=(2, 2), strides=stride)(out)
  return out
