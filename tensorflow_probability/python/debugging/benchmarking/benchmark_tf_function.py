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
"""Library for benchmarking a Python function containing TensorFlow code.

This module supports benchmarking user code under various assumptions, including
both hardware (CPU, GPU) and TF execution models (Eager, tfe.function, XLA).

Note: This module requires Eager mode.

Bechmarking GPU: To benchmark GPU, the host machine must have access to a GPU
(either locally or remotely) and TensorFlow must be compiled with GPU support.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import pprint
import time

# Dependency imports

import tensorflow.compat.v2 as tf

RUNTIME_EAGER = 'eager'
RUNTIME_FUNCTION = 'function/graph'
RUNTIME_XLA = 'function/xla'
RUNTIME_XLA_AUTOCLUSTER = 'xla-autoclustering'

HARDWARE_CPU = 'cpu'
HARDWARE_GPU = 'gpu'

BenchmarkTfFunctionConfig = collections.namedtuple(
    'BenchmarkTfFunctionConfig',
    ['strategies', 'hardware'])


def default_benchmark_config():
  return BenchmarkTfFunctionConfig(
      strategies=frozenset([RUNTIME_EAGER, RUNTIME_FUNCTION,
                            RUNTIME_XLA, RUNTIME_XLA_AUTOCLUSTER]),
      hardware=frozenset([HARDWARE_CPU, HARDWARE_GPU])
  )


def _run_function(user_fn, iters):
  """Run a function and report timing.

  Args:
    user_fn: A callable function of zero arguments.
    iters: Number of times to run `user_fn`.

  Returns:
    first_iter_time: Time (in seconds) to run the first iteration.
    total_time: Time (in seconds) to run all `iters` iterations.
  """
  start_time = time.time()

  for i in range(iters):
    _ = user_fn()
    if i == 0:
      first_iter_time = time.time() - start_time

  total_time = time.time() - start_time
  return first_iter_time, total_time


def _merge_dicts(dict_1, dict_2):
  """Merge two dictionaries. (In Python3.5 or greater, {**dict_1, **dict_2}."""
  assert set(dict_1.keys()).intersection(dict_2.keys()) == set([])

  dict_1_copy = dict_1.copy()
  dict_1_copy.update(dict_2)
  return dict_1_copy


def _run_function_under_strategies(user_fn, iters, config, hardware,
                                   extra_columns, use_autograph,
                                   print_intermediates=False):
  """Run user_fn with varying backends. See public API for details."""

  def run_one(function, runtime):
    """Run user_fn. See public API for details."""
    first_iter_time, total_time = _run_function(function, iters)
    new_dict = _merge_dicts(
        {'runtime': runtime,
         'hardware': hardware,
         'iters': iters,
         'first_iter_time': first_iter_time,
         'total_time': total_time,
         'avg_warm_iter_time': (total_time - first_iter_time) / (iters - 1)},
        extra_columns)

    if print_intermediates:
      print('New benchmark result:')
      pprint.pprint(new_dict)

    return new_dict

  data_dicts = []

  if RUNTIME_EAGER in config.strategies:
    data_dicts.append(run_one(user_fn, RUNTIME_EAGER))

  if RUNTIME_FUNCTION in config.strategies:
    graph_fn = tf.function(user_fn, autograph=use_autograph)
    data_dicts.append(run_one(graph_fn, RUNTIME_FUNCTION))

  if RUNTIME_XLA in config.strategies:
    xla_fn = tf.function(
        user_fn, autograph=use_autograph, jit_compile=True)
    data_dicts.append(run_one(xla_fn, RUNTIME_XLA))

  if RUNTIME_XLA_AUTOCLUSTER in config.strategies:
    @tf.function(autograph=use_autograph)
    def autocluster_fn(*args, **kwargs):
      with tf.xla.experimental.jit_scope(compile_ops=True):
        return user_fn(*args, **kwargs)
    data_dicts.append(run_one(autocluster_fn, RUNTIME_XLA_AUTOCLUSTER))

  return data_dicts


# Initial designs for this code inherited from TensorFlow's
# platform.benchmark.Benchmark class. This would be useful for integrating with
# TensorFlow's automatically-run benchmarks. Currently, this code is meant for
# interactive use. If we decide we want to start running automatically and
# logging results and visualizing them in mldash, the change should be easy:
# make a benchmark class that inherits from benchmark.Benchmark, have it call
# benchmark_tf_function, and then report the results via self.report_benchmark.


def benchmark_tf_function(
    user_fn,
    iters=1,
    config=default_benchmark_config(),
    extra_columns=None,
    # As of this writing (February 2019), autograph is the default for
    # tfe.function, but there seem to be many bugs. Hopefully, in future, this
    # default can be changed to True or the argument can be removed.
    use_autograph=False,
    print_intermediates=False,
    cpu_device='cpu:0',
    gpu_device='gpu:0'):
  """Time a TensorFlow function under a variety of strategies and hardware.

  Runs the callable `user_fn` `iters` times under the strategies (any of Eager,
  tfe.function + graph, and XLA) and hardware (CPU, GPU).


  # Example:
  ```python
  data_dicts = []
  for inner_iters in [10, 100]:
    for size in [100, 1000]:
      def f():
        total = tf.constant(0.0)
        for _ in np.arange(inner_iters):
          m = tf.random.uniform((size, size))
          total += tf.reduce_sum(tf.matmul(m, m))
          return total

      data_dicts += benchmark_tf_function.benchmark_tf_function(
          f,
          iters=5,
          extra_columns={'inner_iters': inner_iters,
                         'size': size})
  ```

  Args:
    user_fn: A zero-argument, callable function of TensorFlow code.
    iters: The number of times to run the function for each runtime and
      hardware combination.
    config: A BenchmarkTfFunctionConfig, specifying which strategies and
      hardware to use. Valid strategies are RUNTIME_EAGER, RUNTIME_FUNCTION, and
      RUNTIME_XLA. Valid hardware choices are HARDWARE_CPU, HARDWARE_GPU.
    extra_columns: A dictionary of extra information to add to each dictionary
      in data_dicts.
    use_autograph: Boolean, controlling whether autograph is used for the
      graph and XLA strategies.
    print_intermediates: Boolean. If true, print out each row before adding it
      to the data_dicts.
    cpu_device: String, the TensorFlow device to use for CPU.
    gpu_device: String, the TensorFlow device to use for GPU.

  Returns:

    data_dicts: A list of dictionaries containing the results of benchmarking
      Time for the first run is stored under the `first_iter_time` key, and time
      for all runs is stored under the `total_time` key.
  """
  data_dicts = []

  if extra_columns is None:
    extra_columns = {}

  if HARDWARE_CPU in config.hardware:
    with tf.device(cpu_device):
      data_dicts += _run_function_under_strategies(
          user_fn, iters, config, HARDWARE_CPU,
          extra_columns, use_autograph, print_intermediates)

  if HARDWARE_GPU in config.hardware:
    if tf.config.list_physical_devices('GPU'):
      with tf.device(gpu_device):
        data_dicts += _run_function_under_strategies(
            user_fn, iters, config, HARDWARE_GPU,
            extra_columns, use_autograph, print_intermediates)
    else:
      print('Skipping GPU runs -- no GPU!')

  return data_dicts
