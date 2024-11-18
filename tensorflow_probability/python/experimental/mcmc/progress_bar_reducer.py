# Copyright 2020 The TensorFlow Probability Authors.
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
"""`ProgressBarReducer` for showing progress bars."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.mcmc import reducer as reducer_base


__all__ = [
    'ProgressBarReducer',
    'make_tqdm_progress_bar_fn',
]


def make_tqdm_progress_bar_fn(description='', leave=True):
  """Make a `progress_bar_fn` that uses `tqdm`.

  Args:
    description: `str` to display next to the progress bar, default is "".
    leave: Boolean whether to leave the progress bar up after finished.

  Returns:
    tqdm_progress_bar_fn: A function that takes an integer `num_steps` and
      returns a `tqdm` progress bar iterator.
  """
  def tqdm_progress_bar_fn(num_steps):
    try:
      import tqdm  # pylint: disable=g-import-not-at-top
    except ImportError:
      raise ImportError('Please install tqdm via pip install tqdm')
    return iter(tqdm.tqdm(range(num_steps), desc=description, leave=leave))
  return tqdm_progress_bar_fn


class ProgressBarReducer(reducer_base.Reducer):
  """`Reducer` that displays a progress bar.

  Note this is not XLA-compatible (`tf.function(jit_compile=True)`).
  Numpy and JAX substrates are not supported.

  Example usage:

  ```
  kernel = ...
  current_state = ...
  num_results = ...
  pbar = tfp.experimental.mcmc.ProgressBarReducer(num_results)
  _, final_state, kernel_results = tfp.experimental.mcmc.sample_fold(
      num_steps=num_results,
      current_state=current_state,
      kernel=kernel,
      reducer=pbar,
  )
  ```
  """

  def __init__(
      self,
      num_results,
      progress_bar_fn=make_tqdm_progress_bar_fn()):
    """Instantiates a reducer that displays a progress bar.

    Args:
      num_results: Integer number of results to expect (as passed to sample
        chain).
      progress_bar_fn: A function that takes an integer `num_results` and
        returns an iterator that advances a progress bar. Defaults to `tqdm`
        progress bars (make sure they are pip installed befure using.)
    """
    self._parameters = dict(
        num_results=num_results,
        progress_bar_fn=progress_bar_fn,
    )

  def initialize(self, initial_chain_state, initial_kernel_results=None):  # pylint: disable=unused-argument
    """Initialize progress bars.

    All arguments are ignored.

    Args:
      initial_chain_state: A (possibly nested) structure of `Tensor`s or Python
        `list`s of `Tensor`s representing the current state(s) of the Markov
        chain(s). It is used to infer the structure of future trace results.
      initial_kernel_results: A (possibly nested) structure of `Tensor`s
        representing internal calculations made in a related `TransitionKernel`.
        It is used to infer the structure of future trace results.

    Returns:
      state: empty list.
    """
    num_results = tf.convert_to_tensor(self.num_results)
    def init_bar(num_results):
      self.bar = self.progress_bar_fn(int(num_results))

    tf.py_function(init_bar, (num_results,), ())
    return []

  def one_step(self, new_chain_state, current_reducer_state,
               previous_kernel_results):  # pylint: disable=unused-argument
    """Advance progress bar by one result.

    All arguments are ignored.

    Args:
      new_chain_state: A (possibly nested) structure of incoming chain state(s)
        with shape and dtype compatible with those used to initialize the
        `TracingState`.
      current_reducer_state: `TracingState`s representing all previously traced
        results.
      previous_kernel_results: A (possibly nested) structure of `Tensor`s
        representing internal calculations made in a related
        `TransitionKernel`.

    Returns:
      new_reducer_state: empty list.
    """
    def update_bar():
      try:
        next(self.bar)
      except StopIteration:
        pass

    tf.py_function(update_bar, (), ())
    return []

  @property
  def num_results(self):
    return self._parameters['num_results']

  @property
  def progress_bar_fn(self):
    return self._parameters['progress_bar_fn']

  @property
  def parameters(self):
    return self._parameters
