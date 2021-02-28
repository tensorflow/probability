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
"""Utility functions for building neural networks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import functools
import sys

import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    'batchify_op',
    'display_imgs',
    'expand_dims',
    'flatten_rightmost',
    'halflife_decay',
    'make_fit_op',
    'tfcompile',
    'trace',
    'tune_dataset',
    'variables_load',
    'variables_save',
    'variables_summary'
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

  # TODO(b/144500779): Cant use `jit_compile=True`.
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
                                   jit_compile=xla_compile_all)
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
    leftmost_ndims = ps.rank(x) - ndims
    new_shape = ps.pad(
        ps.shape(x)[:leftmost_ndims],
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
    """Prints input shape and dtype if possible, else value."""
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
      nx = ps.rank(x)
      na = ps.size(new_axis)
      is_neg_axis = new_axis < 0
      k = ps.reduce_sum(
          ps.cast(is_neg_axis, new_axis.dtype))
      new_axis = ps.where(is_neg_axis, new_axis + nx, new_axis)
      new_axis = ps.sort(new_axis)
      axis_neg, axis_pos = ps.split(new_axis, [k, -1])
      idx = ps.argsort(ps.concat([
          axis_pos,
          ps.range(nx),
          axis_neg,
      ], axis=0), stable=True)
      shape = ps.pad(ps.shape(x),
                     paddings=[[na - k, k]],
                     constant_values=1)
      shape = ps.gather(shape, idx)
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
  if not tf.executing_eagerly():
    raise ValueError('Can only load variables while in eager mode.')
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
  cnt = sum(list(trainable_size.values()))
  lines.append('trainable size: {}  /  {:.3f} MiB  /  {}'.format(
      cnt,
      bytes_ / 2**20,
      '{' + ', '.join(['{}: {}'.format(k.name, v)
                       for k, v in trainable_size.items()]) + '}',
  ))
  return '\n'.join(lines)


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


def batchify_op(op, op_min_input_ndims, x, *other_op_args):
  """Reshape `op` input `x` to be a vec of `op_min_input_ndims`-rank tensors."""
  if x.shape.rank == op_min_input_ndims + 1:
    # Input is already a vector of `op_min_input_ndims`-rank tensors.
    return op(x, *other_op_args)
  batch_shape, op_shape = ps.split(
      ps.shape(x),
      num_or_size_splits=[-1, op_min_input_ndims])
  flat_shape = ps.pad(
      op_shape,
      paddings=[[1, 0]],
      constant_values=-1)
  y = tf.reshape(x, flat_shape)
  y = op(y, *other_op_args)
  unflat_shape = ps.concat([
      batch_shape,
      ps.shape(y)[1:],
  ], axis=0)
  y = tf.reshape(y, unflat_shape)
  return y
