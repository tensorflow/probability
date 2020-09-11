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
import functools
import sys

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
from tensorflow_probability.python.experimental.nn import initializers as nn_init_lib
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.util.deferred_tensor import TransformedVariable


__all__ = [
    'batchify_op',
    'convolution_batch',
    'display_imgs',
    'expand_dims',
    'flatten_rightmost',
    'halflife_decay',
    'make_fit_op',
    'make_kernel_bias',
    'make_kernel_bias_posterior_mvn_diag',
    'make_kernel_bias_prior_spike_and_slab',
    'negloglik',
    'prepare_conv_args',
    'prepare_strides',
    'prepare_tuple_argument',
    'tfcompile',
    'trace',
    'tune_dataset',
    'variables_load',
    'variables_save',
    'variables_summary',
]


def display_imgs(x, title=None, fignum=None):
  """Display images as a grid."""
  import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel,g-import-not-at-top
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
    axs.set_title(str(title))
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


def tfcompile(func=None,
              tf_function=True,
              xla_best_effort=True,
              xla_compile_all=False):
  """Centralizes TF compilation related options.

  Args:
    func: Python `callable` to wrapped with the specified TF compilation
      directives.
      Default value: `None`.
    tf_function: `bool` representing whether the resulting function should be
      `tf.function` decoreated.
      Default value: `True`.
    xla_best_effort: `bool` representing whether XLA auto-clustering compilation
      should be performed. (This argument is ignored if the function is executed
      eagerly.)
      Default value: `True`.
    xla_compile_all: `bool` representing whether XLA compilation should be
      performed. (This argument overrides both `tf_function` and
      `xla_best_effort`.
      Default value: `False`.

  Returns:
    wrapped_func: A Python `callable` with the specified compilation directives
      embedded.

  ### Example Usage

  ```python
  tfn = tfp.experimental.nn

  # Use style #1.
  @tfn.util.tfcompile(xla_compile_all=True)
  def foo(...):
       ...

  # Use style #2.
  def foo(...):
    ...
  foo = tfn.util.tfcompile(xla_compile_all=True)(foo)
  ```

  """
  if not (tf_function or xla_best_effort or xla_compile_all):
    # This specialization makes for smaller stack trace and easier debugging.
    return lambda fn: fn if func is None else func

  # Note: xla_compile_all overrides both tf_function and xla_best_effort.
  tf_function = tf_function or xla_compile_all
  xla_best_effort = xla_best_effort and not xla_compile_all
  maybe_tf_function = (tf.function(autograph=False,
                                   experimental_compile=xla_compile_all)
                       if tf_function else _dummy_context())
  def decorator(f):
    @functools.wraps(f)
    @maybe_tf_function
    def wrapped(*args, **kwargs):
      maybe_xla_best_effort = (tf.xla.experimental.jit_scope(compile_ops=True)
                               if not tf.executing_eagerly() and xla_best_effort
                               else _dummy_context())
      with maybe_xla_best_effort:
        return f(*args, **kwargs)
    return wrapped

  if func is None:
    # This branch handles the following use case:
    #   @tfcompile(...)
    #   def foo(...):
    #      ...
    return decorator
  else:
    # This branch handles the following use case:
    #   foo = tfcompile(...)(foo)
    return decorator(func)


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
      `lambda grads: tf.nest.map_structure(
         lambda x: 0. if x is None else tf.norm(x), grads)`.
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
  @tfcompile(tf_function=tf_function, xla_best_effort=xla_compile)
  def fit_op(*args, **kwargs):
    """Performs one gradient descent update to `trainable_variables`."""
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tf.nest.map_structure(tape.watch, trainable_variables)
      loss, other = loss_fn(*args, **kwargs)
    grads = tf.nest.pack_sequence_as(
        trainable_variables,
        tape.gradient(loss, tf.nest.flatten(trainable_variables)))
    try:
      seq_type = collections.abc.Sequence
    except AttributeError:
      seq_type = collections.Sequence
    if isinstance(optimizer, seq_type):
      for opt, g, v in zip(optimizer, grads, trainable_variables):
        _apply_gradients(opt, g, v)
    else:
      _apply_gradients(optimizer, grads, trainable_variables)
    if grad_summary_fn is not None:
      return loss, other, grad_summary_fn(grads)
    return loss, other
  # Note: we can't do `return tf.xla.experimental.compile(fit)` since we can't
  # assume the function arguments are coercible to `tf.Tensor`s.
  return fit_op


def _apply_gradients(opt, g, v):
  gvs = tuple((g_, v_) for g_, v_ in zip(tf.nest.flatten(g),
                                         tf.nest.flatten(v))
              if g_ is not None)
  if gvs:
    opt.apply_gradients(gvs)


def flatten_rightmost(ndims=3):
  """Flatten rightmost dims."""
  def flatten_rightmost_(x):
    """Implementation of `flatten_rightmost`."""
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
  return flatten_rightmost_


def trace(name=None):
  """Returns a function which prints info related to input."""
  name = '' if name is None else 'name:{:10}  '.format(name)
  def trace_(x):
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
  return trace_


def expand_dims(axis, name=None):
  """Like `tf.expand_dims` but accepts a vector of axes to expand."""
  def expand_dims_(x):
    """Implementation of `expand_dims`."""
    with tf.name_scope(name or 'expand_dims'):
      x = tf.convert_to_tensor(x, name='x')
      new_axis = tf.convert_to_tensor(axis, dtype_hint=tf.int32, name='axis')
      nx = prefer_static.rank(x)
      na = prefer_static.size(new_axis)
      is_neg_axis = new_axis < 0
      k = prefer_static.reduce_sum(
          prefer_static.cast(is_neg_axis, new_axis.dtype))
      new_axis = prefer_static.where(is_neg_axis, new_axis + nx, new_axis)
      new_axis = prefer_static.sort(new_axis)
      axis_neg, axis_pos = prefer_static.split(new_axis, [k, -1])
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
  return expand_dims_


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


def make_kernel_bias(
    kernel_shape,
    bias_shape,
    kernel_initializer=None,
    bias_initializer=None,
    kernel_batch_ndims=0,  # pylint: disable=unused-argument
    bias_batch_ndims=0,  # pylint: disable=unused-argument
    dtype=tf.float32,
    kernel_name='kernel',
    bias_name='bias'):
  # pylint: disable=line-too-long
  """Creates kernel and bias as `tf.Variable`s.

  Args:
    kernel_shape: ...
    bias_shape: ...
    kernel_initializer: ...
      Default value: `None` (i.e., `tf.initializers.glorot_uniform()`).
    bias_initializer: ...
      Default value: `None` (i.e., `tf.initializers.zeros()`).
    kernel_batch_ndims: ...
      Default value: `0`.
    bias_batch_ndims: ...
      Default value: `0`.
    dtype: ...
      Default value: `tf.float32`.
    kernel_name: ...
      Default value: `"kernel"`.
    bias_name: ...
      Default value: `"bias"`.

  Returns:
    kernel: ...
    bias: ...

  #### Recomendations:

  ```python
  #   tf.nn.relu    ==> tf.initializers.he_*
  #   tf.nn.elu     ==> tf.initializers.he_*
  #   tf.nn.selu    ==> tf.initializers.lecun_*
  #   tf.nn.tanh    ==> tf.initializers.glorot_*
  #   tf.nn.sigmoid ==> tf.initializers.glorot_*
  #   tf.nn.softmax ==> tf.initializers.glorot_*
  #   None          ==> tf.initializers.glorot_*
  # https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404
  # https://stats.stackexchange.com/a/393012/1835

  def make_uniform(size):
    s = tf.math.rsqrt(size / 3.)
    return tfd.Uniform(low=-s, high=s)

  def make_normal(size):
    # Constant is: `scipy.stats.truncnorm.var(loc=0., scale=1., a=-2., b=2.)`.
    s = tf.math.rsqrt(size) / 0.87962566103423978
    return tfd.TruncatedNormal(loc=0, scale=s, low=-2., high=2.)

  # He.  https://arxiv.org/abs/1502.01852
  he_uniform = make_uniform(fan_in / 2.)
  he_normal  = make_normal (fan_in / 2.)

  # Glorot (aka Xavier). http://proceedings.mlr.press/v9/glorot10a.html
  glorot_uniform = make_uniform((fan_in + fan_out) / 2.)
  glorot_normal  = make_normal ((fan_in + fan_out) / 2.)
  ```

  """
  # pylint: enable=line-too-long
  if kernel_initializer is None:
    kernel_initializer = nn_init_lib.glorot_uniform()
  if bias_initializer is None:
    bias_initializer = tf.initializers.zeros()
  return (
      tf.Variable(_try_call_init_fn(kernel_initializer,
                                    kernel_shape,
                                    dtype,
                                    kernel_batch_ndims),
                  name=kernel_name),
      tf.Variable(_try_call_init_fn(bias_initializer,
                                    bias_shape,
                                    dtype,
                                    bias_batch_ndims),
                  name=bias_name),
  )


def make_kernel_bias_prior_spike_and_slab(
    kernel_shape,
    bias_shape,
    kernel_initializer=None,  # pylint: disable=unused-argument
    bias_initializer=None,  # pylint: disable=unused-argument
    kernel_batch_ndims=0,  # pylint: disable=unused-argument
    bias_batch_ndims=0,  # pylint: disable=unused-argument
    dtype=tf.float32,
    kernel_name='prior_kernel',
    bias_name='prior_bias'):
  """Create prior for Variational layers with kernel and bias.

  Note: Distribution scale is inversely related to regularization strength.
  Consider a "Normal" prior; bigger scale corresponds to less L2 regularization.
  I.e.,
  ```python
  scale    = (2. * l2weight)**-0.5
  l2weight = scale**-2. / 2.
  ```
  have a similar regularizing effect.

  The std. deviation of each of the component distributions returned by this
  function is approximately `1415` (or approximately `l2weight = 25e-6`). In
  other words this prior is extremely "weak".

  Args:
    kernel_shape: ...
    bias_shape: ...
    kernel_initializer: Ignored.
      Default value: `None` (i.e., `tf.initializers.glorot_uniform()`).
    bias_initializer: Ignored.
      Default value: `None` (i.e., `tf.initializers.zeros()`).
    kernel_batch_ndims: ...
      Default value: `0`.
    bias_batch_ndims: ...
      Default value: `0`.
    dtype: ...
      Default value: `tf.float32`.
    kernel_name: ...
      Default value: `"prior_kernel"`.
    bias_name: ...
      Default value: `"prior_bias"`.

  Returns:
    kernel_and_bias_distribution: ...
  """
  w = MixtureSameFamily(
      mixture_distribution=Categorical(probs=[0.5, 0.5]),
      components_distribution=Normal(
          loc=0.,
          scale=tf.constant([1., 2000.], dtype=dtype)))
  return JointDistributionSequential([
      Sample(w, kernel_shape, name=kernel_name),
      Sample(w, bias_shape, name=bias_name),
  ])


def make_kernel_bias_posterior_mvn_diag(
    kernel_shape,
    bias_shape,
    kernel_initializer=None,
    bias_initializer=None,
    kernel_batch_ndims=0,  # pylint: disable=unused-argument
    bias_batch_ndims=0,  # pylint: disable=unused-argument
    dtype=tf.float32,
    kernel_name='posterior_kernel',
    bias_name='posterior_bias'):
  """Create learnable posterior for Variational layers with kernel and bias.

  Args:
    kernel_shape: ...
    bias_shape: ...
    kernel_initializer: ...
      Default value: `None` (i.e., `tf.initializers.glorot_uniform()`).
    bias_initializer: ...
      Default value: `None` (i.e., `tf.initializers.zeros()`).
    kernel_batch_ndims: ...
      Default value: `0`.
    bias_batch_ndims: ...
      Default value: `0`.
    dtype: ...
      Default value: `tf.float32`.
    kernel_name: ...
      Default value: `"posterior_kernel"`.
    bias_name: ...
      Default value: `"posterior_bias"`.

  Returns:
    kernel_and_bias_distribution: ...
  """
  if kernel_initializer is None:
    kernel_initializer = nn_init_lib.glorot_uniform()
  if bias_initializer is None:
    bias_initializer = tf.initializers.zeros()
  make_loc = lambda init_fn, shape, batch_ndims, name: tf.Variable(  # pylint: disable=g-long-lambda
      _try_call_init_fn(init_fn, shape, dtype, batch_ndims),
      name=name + '_loc')
  # Setting the initial scale to a relatively small value causes the `loc` to
  # quickly move toward a lower loss value.
  make_scale = lambda shape, name: TransformedVariable(  # pylint: disable=g-long-lambda
      tf.fill(shape, value=tf.constant(1e-3, dtype=dtype)),
      Chain([Shift(1e-5), Softplus()]),
      name=name + '_scale')
  return JointDistributionSequential([
      Independent(
          Normal(loc=make_loc(kernel_initializer,
                              kernel_shape,
                              kernel_batch_ndims,
                              kernel_name),
                 scale=make_scale(kernel_shape, kernel_name)),
          reinterpreted_batch_ndims=prefer_static.size(kernel_shape),
          name=kernel_name),
      Independent(
          Normal(loc=make_loc(bias_initializer,
                              bias_shape,
                              kernel_batch_ndims,
                              bias_name),
                 scale=make_scale(bias_shape, bias_name)),
          reinterpreted_batch_ndims=prefer_static.size(bias_shape),
          name=bias_name),
  ])


def halflife_decay(time_step, half_life, initial, final=0., dtype=tf.float32,
                   name=None):
  """Interpolates `initial` to `final` using halflife (exponential) decay."""
  with tf.name_scope(name or 'halflife_decay'):
    dtype = dtype_util.common_dtype([initial, final, half_life],
                                    dtype_hint=tf.float32)
    initial = tf.convert_to_tensor(initial, dtype=dtype, name='initial')
    final = tf.convert_to_tensor(final, dtype=dtype, name='final')
    half_life = tf.convert_to_tensor(half_life, dtype=dtype, name='half_life')
    time_step = tf.cast(time_step, dtype=dtype, name='time_step')
    return final + (initial - final) * 0.5**(time_step / half_life)


def prepare_tuple_argument(x, n, arg_name):
  """Helper which puts tuples in standard form."""
  if isinstance(x, int):
    return (x,) * n
  try:
    x = tuple(x)
  except TypeError:
    raise ValueError('Argument {} must be convertible to tuple.'.format(
        arg_name))
  if n != len(x):
    raise ValueError('Argument {} has invalid length; expected:{}, '
                     'saw:{}.'.format(arg_name, n, len(x)))
  for x_ in x:
    try:
      int(x_)
    except (ValueError, TypeError):
      raise ValueError('Argument {} contains non-integer input; '
                       'saw: {}.'.format(arg_name, x_))
  return x


def _prepare_padding_argument(x):
  """Helper which processes the padding argument."""
  if not hasattr(x, 'upper'):
    return tuple(x)
  padding = x.upper()
  if padding in {'CAUSAL', 'FULL'}:
    raise NotImplementedError(
        'Argument `padding` value "{}" currently not supported. If you '
        'require this feature, please create an issue on '
        '`https://github.com/tensorflow/probability` or email '
        '`tfprobability@tensorflow.org`.'.format(padding))
  valid_values = {'VALID', 'SAME'}
  if padding not in valid_values:
    raise ValueError('Argument `padding` must be convertible to a tuple '
                     'or one of {}; saw: "{}".'.format(valid_values, padding))
  return padding


def prepare_conv_args(rank, strides, padding, dilations):
  """Sanitizes use provided input."""
  try:
    rank = int(tf.get_static_value(rank))
  except TypeError:
    raise TypeError('Argument `rank` must be statically known `int`.')
  valid_rank = {1, 2, 3}
  if rank not in valid_rank:
    raise ValueError('Argument `rank` must be in {}.'.format(valid_rank))
  strides = prepare_tuple_argument(strides, rank, arg_name='strides')
  padding = _prepare_padding_argument(padding)
  dilations = prepare_tuple_argument(dilations, rank, arg_name='dilations')
  data_format = {1: 'NWC', 2: 'NHWC', 3: 'NDHWC'}.get(rank)
  return rank, strides, padding, dilations, data_format


def prepare_strides(x, n, arg_name):
  """Mimics prepare_tuple_argument, but puts a 1 for the 0th and 3rd element."""
  if isinstance(x, int):
    return (1,) + (x,) * (n-2) + (1,)
  try:
    x = tuple(x)
  except TypeError:
    raise ValueError('Argument {} must be convertible to tuple.'.format(
        arg_name))
  if n != len(x):
    raise ValueError('Argument {} has invalid length; expected:{}, '
                     'saw:{}.'.format(arg_name, n, len(x)))
  for x_ in x:
    try:
      int(x_)
    except (ValueError, TypeError):
      raise ValueError('Argument {} contains non-integer input; '
                       'saw: {}.'.format(arg_name, x_))
  return x


def _convolution_batch_nhwbc(
    x, kernel, rank, strides, padding, dilations, name):
  """Specialization of batch conv to NHWBC data format."""
  with tf.name_scope(name or 'conv2d_nhwbc'):
    # Prepare arguments.
    [
        rank,
        _,  # strides
        padding,
        dilations,
        _,  # data_format
    ] = prepare_conv_args(rank, strides, padding, dilations)
    strides = prepare_strides(strides, rank + 2, arg_name='strides')

    dtype = dtype_util.common_dtype([x, kernel], dtype_hint=tf.float32)
    x = tf.convert_to_tensor(x, dtype=dtype, name='x')
    kernel = tf.convert_to_tensor(kernel, dtype=dtype, name='kernel')

    # Step 1: Transpose and double flatten kernel.
    # kernel.shape = B + F + [c, c']. Eg: [b, fh, fw, c, c']
    kernel_shape = prefer_static.shape(kernel)
    kernel_batch_shape, kernel_event_shape = prefer_static.split(
        kernel_shape,
        num_or_size_splits=[-1, rank + 2])
    kernel_batch_size = prefer_static.reduce_prod(kernel_batch_shape)
    kernel_ndims = prefer_static.rank(kernel)
    kernel_batch_ndims = kernel_ndims - rank - 2
    perm = prefer_static.concat([
        prefer_static.range(kernel_batch_ndims, kernel_batch_ndims + rank),
        prefer_static.range(0, kernel_batch_ndims),
        prefer_static.range(kernel_batch_ndims + rank, kernel_ndims),
    ], axis=0)  # Eg, [1, 2, 0, 3, 4]
    kernel = tf.transpose(kernel, perm=perm)  # F + B + [c, c']
    kernel = tf.reshape(
        kernel,
        shape=prefer_static.concat([
            kernel_event_shape[:rank],
            [kernel_batch_size * kernel_event_shape[-2],
             kernel_event_shape[-1]],
        ], axis=0))  # F + [bc, c']

    # Step 2: Double flatten x.
    # x.shape = N + D + B + [c]
    x_shape = prefer_static.shape(x)
    [
        x_sample_shape,
        x_rank_shape,
        x_batch_shape,
        x_channel_shape,
    ] = prefer_static.split(
        x_shape,
        num_or_size_splits=[-1, rank, kernel_batch_ndims, 1])
    x = tf.reshape(
        x,  # N + D + B + [c]
        shape=prefer_static.concat([
            [prefer_static.reduce_prod(x_sample_shape)],
            x_rank_shape,
            [prefer_static.reduce_prod(x_batch_shape) *
             prefer_static.reduce_prod(x_channel_shape)],
        ], axis=0))  # [n] + D + [bc]

    # Step 3: Apply convolution.
    y = tf.nn.depthwise_conv2d(
        x, kernel,
        strides=strides,
        padding=padding,
        data_format='NHWC',
        dilations=dilations)
    #  SAME: y.shape = [n, h,      w,      bcc']
    # VALID: y.shape = [n, h-fh+1, w-fw+1, bcc']

    # Step 4: Reshape/reduce for output.
    y_shape = prefer_static.shape(y)
    y = tf.reshape(
        y,
        shape=prefer_static.concat([
            x_sample_shape,
            y_shape[1:-1],
            kernel_batch_shape,
            kernel_event_shape[-2:],
        ], axis=0))  # N + D' + B + [c, c']
    y = tf.reduce_sum(y, axis=-2)  # N + D' + B + [c']

    return y


def _convolution_batch_nbhwc(
    x, kernel, rank, strides, padding, dilations, name):
  """Specialization of batch conv to NBHWC data format."""
  result = _convolution_batch_nhwbc(
      tf.transpose(x, [0, 2, 3, 1, 4]),
      kernel, rank, strides, padding, dilations, name)
  return tf.transpose(result, [0, 3, 1, 2, 4])


def convolution_batch(x, kernel, rank, strides, padding, data_format=None,
                      dilations=None, name=None):
  """Like `tf.nn.conv2d` except applies batch of kernels to batch of `x`."""
  if rank != 2:
    raise NotImplementedError('Argument `rank` currently only supports `2`; '
                              'saw "{}".'.format(rank))
  if data_format is not None and data_format.upper() == 'NHWBC':
    return _convolution_batch_nhwbc(
        x, kernel, rank, strides, padding, dilations, name)
  if data_format is not None and data_format.upper() == 'NBHWC':
    return _convolution_batch_nbhwc(
        x, kernel, rank, strides, padding, dilations, name)
  raise ValueError('Argument `data_format` currently only supports "NHWBC" and '
                   '"NBHWC"; saw "{}".'.format(data_format))


def _try_call_init_fn(fn, *args):
  """Try to call function with first num_args else num_args - 1."""
  try:
    return fn(*args)
  except TypeError:
    return fn(*args[:-1])


def batchify_op(op, op_min_input_ndims, x, *other_op_args):
  """Reshape `op` input `x` to be a vec of `op_min_input_ndims`-rank tensors."""
  if x.shape.rank == op_min_input_ndims + 1:
    # Input is already a vector of `op_min_input_ndims`-rank tensors.
    return op(x, *other_op_args)
  batch_shape, op_shape = prefer_static.split(
      prefer_static.shape(x),
      num_or_size_splits=[-1, op_min_input_ndims])
  flat_shape = prefer_static.pad(
      op_shape,
      paddings=[[1, 0]],
      constant_values=-1)
  y = tf.reshape(x, flat_shape)
  y = op(y, *other_op_args)
  unflat_shape = prefer_static.concat([
      batch_shape,
      prefer_static.shape(y)[1:],
  ], axis=0)
  y = tf.reshape(y, unflat_shape)
  return y
