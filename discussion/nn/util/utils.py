# Lint as: python2, python3
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
"""Utilitity functions for building neural networks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import sys

import matplotlib.pyplot as plt
import numpy as np

from six.moves import zip
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors.chain import Chain
from tensorflow_probability.python.bijectors.shift import Shift
from tensorflow_probability.python.bijectors.softplus import Softplus
from tensorflow_probability.python.distributions.categorical import Categorical
from tensorflow_probability.python.distributions.independent import Independent
from tensorflow_probability.python.distributions.joint_distribution_sequential import JointDistributionSequential
from tensorflow_probability.python.distributions.mixture_same_family import MixtureSameFamily
from tensorflow_probability.python.distributions.normal import Normal
from tensorflow_probability.python.distributions.sample import Sample
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.util.deferred_tensor import TransformedVariable


__all__ = [
    'display_imgs',
    'expand_dims',
    'flatten_rightmost',
    'make_fit_op',
    'make_kernel_bias',
    'make_kernel_bias_posterior_mvn_diag',
    'make_kernel_bias_prior_spike_and_slab',
    'negloglik',
    'trace',
    'tune_dataset',
    'variables_load',
    'variables_save',
    'variables_summary',
]


def display_imgs(x, title=None, fignum=None):
  """Display images as a grid."""
  if not tf.executing_eagerly():
    raise NotImplementedError('`display_imgs` can only be executed eagerly.')
  def _preprocess(z):
    return np.array(getattr(z, 'numpy', lambda: z)())
  x = _preprocess(x)
  if title is not None:
    title = _preprocess(title)
  x = np.reshape(x, (-1,) + x.shape[-4:])
  nrows, ncols, h, w, c = x.shape
  x = np.reshape(np.transpose(x, [0, 2, 1, 3, 4]), [nrows * h, ncols * w, c])
  plt.ioff()
  subplots_kwargs = dict(
      nrows=1,
      ncols=1,
      figsize=(ncols, max(2, nrows)),
      num=fignum,
      clear=True)
  try:
    fig, axs = plt.subplots(**subplots_kwargs)
  except TypeError:
    subplots_kwargs.pop('clear')
    fig, axs = plt.subplots(**subplots_kwargs)
  axs.imshow(x.squeeze(), interpolation='none', cmap='gray')
  axs.axis('off')
  if title is not None:
    axs.set_title(title)
  fig.tight_layout()
  plt.show()
  plt.ion()
  return fig, axs


def tune_dataset(dataset,
                 batch_size=None,
                 shuffle_size=None,
                 preprocess_fn=None,
                 repeat_count=-1):
  """Sets generally recommended parameters for a `tf.data.Dataset`.

  Args:
    dataset: `tf.data.Dataset`-like instance to be tuned according to this
      functions arguments.
    batch_size: Python `int` representing the number of elements in each
      minibatch.
    shuffle_size: Python `int` representing the number of elements to shuffle
      (at a time).
    preprocess_fn: Python `callable` applied to each item in `dataset`.
    repeat_count: Python `int`, representing the number of times the dataset
      should be repeated. The default behavior (`repeat_count = -1`) is for the
      dataset to be repeated indefinitely. If `repeat_count is None` repeat is
      "off;" note that this is a deviation from `tf.data.Dataset.repeat` which
      interprets `None` as "repeat indefinitely".
      Default value: `-1` (i.e., repeat indefinitely).

  Returns:
    tuned_dataset: `tf.data.Dataset` instance tuned according to this functions
      arguments.

  #### Example

  ```python
  [train_dataset, eval_dataset], datasets_info = tfds.load(
       name='mnist',
       split=['train', 'test'],
       with_info=True,
       as_supervised=True,
       shuffle_files=True)

  def _preprocess(image, label):
    image = tf.cast(image, dtype=tf.int32)
    u = tf.random.uniform(shape=tf.shape(image), maxval=256, dtype=image.dtype)
    image = tf.cast(u < image, dtype=tf.float32)   # Randomly binarize.
    return image, label

  # TODO(b/144500779): Cant use `experimental_compile=True`.
  @tf.function(autograph=False)
  def one_step(iter):
    x, y = next(iter)
    return tf.reduce_mean(x)

  ds = tune_dataset(
      train_dataset,
      batch_size=32,
      shuffle_size=int(datasets_info.splits['train'].num_examples / 7),
      preprocess_fn=_preprocess)
  it = iter(ds)
  [one_step(it)]*3  # Build graph / burn-in.
  %time one_step(it)
  ```

  """
  # https://www.tensorflow.org/guide/data_performance
  # The order of these builder arguments matters.
  # The following has been tuned using the snippet above.
  if preprocess_fn is not None:
    dataset = dataset.map(
        preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.cache()
  if shuffle_size is not None:
    dataset = dataset.shuffle(shuffle_size)
  # Repeat must follow shuffle, else samples could be dropped if we only
  # complete one epoch. E.g.,
  # ds = tf.data.Dataset.range(10).repeat(2).shuffle(5)
  # seen = set(e.numpy() for i, e in enumerate(ds) if i < 10)  # One epoch.
  # assert len(seen) == 10
  if repeat_count is not None:
    dataset = dataset.repeat(repeat_count)
  if batch_size is not None:
    dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset


def negloglik(x, y, model_fn, axis=-1):
  """Negative log-likelihood."""
  return -tf.reduce_mean(model_fn(x).log_prob(y), axis=axis)


@contextlib.contextmanager
def _dummy_context():
  """A context manager which does nothing."""
  yield


def make_fit_op(loss_fn, optimizer, trainable_variables,
                grad_summary_fn=None, tf_function=True, xla_compile=True):
  """One training step.

  Args:
    loss_fn: Python `callable` which returns the pair `loss` (`tf.Tensor`) and
      any other second result such that
      `tf.nest.map_structure(tf.convert_to_tensor, other)` will succeed.
    optimizer: `tf.optimizers.Optimizer`-like instance which has members
      `gradient` and `apply_gradients`.
    trainable_variables: `tf.nest.flatten`-able structure of `tf.Variable`
      instances.
    grad_summary_fn: Python `callable` which takes a `trainable_variables`-like
      structure of `tf.Tensor`s representing the gradient of the result of
      `loss_fn` with respect to `trainable_variables`. For example,
      `lambda grads: tf.nest.map_structure(tf.norm, grads)`.
      Default value: `None` (i.e., no summarization is made).
    tf_function: `bool` representing whether the resulting function should be
      `tf.function` decoreated.
      Default value: `True`.
    xla_compile: `bool` representing whether XLA compilation should be
      performed. (This argument is ignored if the function is executed eagerly.)
      Default value: `True`.

  Returns:
    fit_op: A Python `callable` taking args which are forwarded to `loss_fn` and
      such that when called `trainable_variables` are updated per the logic of
      `optimizer.apply_gradients`.
  """
  maybe_tf_function = (tf.function(autograph=False) if tf_function
                       else _dummy_context())
  @maybe_tf_function
  def fit_op(*args, **kwargs):
    """Performs one gradient descent update to `trainable_variables`."""
    maybe_xla_compile = (tf.xla.experimental.jit_scope(compile_ops=True)
                         if not tf.executing_eagerly() and xla_compile
                         else _dummy_context())
    with maybe_xla_compile:
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tf.nest.map_structure(tape.watch, trainable_variables)
        loss, other = loss_fn(*args, **kwargs)
      grads = tape.gradient(loss, trainable_variables)
      optimizer.apply_gradients(list(zip(
          tf.nest.flatten(grads),
          tf.nest.flatten(trainable_variables))))
      if grad_summary_fn is not None:
        return loss, other, grad_summary_fn(grads)
      return loss, other
  # Note: we can't do `return tf.xla.experimental.compile(fit)` since we can't
  # assume the function arguments are coercible to `tf.Tensor`s.
  return fit_op


def flatten_rightmost(x, ndims=3):
  """Flatten rightmost dims."""
  leftmost_ndims = prefer_static.rank(x) - ndims
  new_shape = prefer_static.pad(
      prefer_static.shape(x)[:leftmost_ndims],
      paddings=[[0, 1]],
      constant_values=-1)
  y = tf.reshape(x, new_shape)
  if x.shape.ndims is not None:
    d = x.shape[leftmost_ndims:]
    d = np.prod(d) if d.is_fully_defined() else None
    y.set_shape(x.shape[:leftmost_ndims].concatenate(d))
  return y


def trace(name=None):
  """Returns a function which prints info related to input."""
  name = '' if name is None else 'name:{:10}  '.format(name)
  def _trace(x):
    """Prints something."""
    if hasattr(x, 'dtype') and hasattr(x, 'shape'):
      print('--- TRACE:  {}shape:{:16}  dtype:{:10}'.format(
          name,
          str(tensorshape_util.as_list(x.shape)),
          dtype_util.name(x.dtype)))
    else:
      print('--- TRACE:  {}value:{}'.format(name, x))
    sys.stdout.flush()
    return x
  return _trace


def expand_dims(x, axis, name=None):
  """Like `tf.expand_dims` but accepts a vector of axes to expand."""
  with tf.name_scope(name or 'expand_dims'):
    x = tf.convert_to_tensor(x, name='x')
    axis = tf.convert_to_tensor(axis, dtype_hint=tf.int32, name='axis')
    nx = prefer_static.rank(x)
    na = prefer_static.size(axis)
    is_neg_axis = axis < 0
    k = prefer_static.reduce_sum(prefer_static.cast(is_neg_axis, axis.dtype))
    axis = prefer_static.where(is_neg_axis, axis + nx, axis)
    axis = prefer_static.sort(axis)
    axis_neg, axis_pos = prefer_static.split(axis, [k, -1])
    idx = prefer_static.argsort(prefer_static.concat([
        axis_pos,
        prefer_static.range(nx),
        axis_neg,
    ], axis=0), stable=True)
    shape = prefer_static.pad(prefer_static.shape(x),
                              paddings=[[na - k, k]],
                              constant_values=1)
    shape = prefer_static.gather(shape, idx)
    return tf.reshape(x, shape)


def variables_save(filename, variables):
  """Saves structure of `tf.Variable`s to `filename`."""
  if not tf.executing_eagerly():
    raise ValueError('Can only `save` while in eager mode.')
  np.savez_compressed(
      filename, *[v.numpy() for v in tf.nest.flatten(variables)])


def variables_load(filename, variables):
  """Assigns values to structure of `tf.Variable`s from `filename`."""
  with np.load(filename) as data:
    vars_ = tf.nest.flatten(variables)
    if len(vars_) != len(data):
      raise ValueError(
          'File "{}" has incorrect number of variables '
          '(saw: {}, expected: {}).'.format(filename, len(data), len(vars_)))
    return tf.group(
        [v.assign(x) for v, (_, x) in zip(vars_, list(data.items()))])


def variables_summary(variables, name=None):
  """Returns a list of summarizing `str`s."""
  trainable_size = collections.defaultdict(lambda: 0)
  lines = []
  if name is not None:
    lines.append(' '.join(['='*3, name, '='*50]))
  fmt = '{: >6} {:20} {:5} {:40}'
  lines.append(fmt.format(
      'SIZE',
      'SHAPE',
      'TRAIN',
      'NAME',
  ))
  for v in tf.nest.flatten(variables):
    num_elements = tensorshape_util.num_elements(v.shape)
    if v.trainable:
      trainable_size[v.dtype.base_dtype] += num_elements
    lines.append(fmt.format(
        num_elements,
        str(tensorshape_util.as_list(v.shape)),
        str(v.trainable),
        v.name,
    ))
  bytes_ = sum([k.size * v for k, v in trainable_size.items()])
  cnt = sum([v for v in trainable_size.values()])
  lines.append('trainable size: {}  /  {:.3f} MiB  /  {}'.format(
      cnt,
      bytes_ / 2**20,
      '{' + ', '.join(['{}: {}'.format(k.name, v)
                       for k, v in trainable_size.items()]) + '}',
  ))
  return '\n'.join(lines)


# make_kernel_bias* functions must all have the same call signature.


def make_kernel_bias_prior_spike_and_slab(
    kernel_shape,
    bias_shape,
    dtype=tf.float32,
    kernel_initializer=None,
    bias_initializer=None):
  """Create prior for Variational layers with kernel and bias."""
  del kernel_initializer, bias_initializer
  w = MixtureSameFamily(
      mixture_distribution=Categorical(probs=[0.5, 0.5]),
      components_distribution=Normal(
          loc=0.,
          scale=tf.constant([1., 2000.], dtype=dtype)))
  return JointDistributionSequential([
      Sample(w, kernel_shape, name='prior_kernel'),
      Sample(w, bias_shape, name='prior_bias'),
  ])


def make_kernel_bias_posterior_mvn_diag(
    kernel_shape,
    bias_shape,
    dtype=tf.float32,
    kernel_initializer=None,
    bias_initializer=None):
  """Create learnable posterior for Variational layers with kernel and bias."""
  if kernel_initializer is None:
    kernel_initializer = tf.initializers.glorot_normal()
  if bias_initializer is None:
    bias_initializer = tf.initializers.glorot_normal()
  make_loc = lambda shape, init, name: tf.Variable(  # pylint: disable=g-long-lambda
      init(shape, dtype=dtype),
      name=name + '_loc')
  make_scale = lambda shape, name: TransformedVariable(  # pylint: disable=g-long-lambda
      tf.ones(shape, dtype=dtype),
      Chain([Shift(1e-5), Softplus()]),
      name=name + '_scale')
  return JointDistributionSequential([
      Independent(
          Normal(
              loc=make_loc(kernel_shape,
                           kernel_initializer,
                           'posterior_kernel'),
              scale=make_scale(kernel_shape, 'posterior_kernel')),
          reinterpreted_batch_ndims=prefer_static.size(kernel_shape),
          name='posterior_kernel'),
      Independent(
          Normal(
              loc=make_loc(bias_shape,
                           bias_initializer,
                           'posterior_bias'),
              scale=make_scale(bias_shape, 'posterior_bias')),
          reinterpreted_batch_ndims=prefer_static.size(bias_shape),
          name='posterior_bias'),
  ])


def make_kernel_bias(
    kernel_shape,
    bias_shape,
    dtype=tf.float32,
    kernel_initializer=None,
    bias_initializer=None,
    kernel_name='kernel',
    bias_name='bias'):
  """Creates kernel and bias as `tf.Variable`s."""
  # http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf, Equation 16.
  # ==> lim = np.sqrt(6. / max(2., float(fan_in + fan_out)))
  #     tf.random.uniform(-lim, lim)
  # initializer_fn = tf.initializers.glorot_uniform()
  def _make(shape, name, initializer):
    if initializer is None:
      initializer = tf.initializers.glorot_normal()
    return tf.Variable(initializer(shape, dtype), name=name)
  kernel = _make(kernel_shape, kernel_name, kernel_initializer)
  bias = _make(bias_shape, bias_name, bias_initializer)
  return kernel, bias
