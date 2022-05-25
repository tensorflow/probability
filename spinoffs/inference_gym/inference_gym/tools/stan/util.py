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
"""Utilities for Stan models."""

import contextlib
import glob
import hashlib
import os
import re
import shutil
import tempfile

from absl import flags
import cmdstanpy
import numpy as np

flags.DEFINE_string(
    'stan_model_filename_template', '/tmp/stan_model_{}',
    'Template for the filename of the compiled model.'
    '"{}" is replaced with the hash of the model source.')

FLAGS = flags.FLAGS

__all__ = [
    'cached_stan_model',
    'get_columns',
    'make_sample_fn',
]


def cached_stan_model(source):
  """Compiles and caches a Stan model.

  Args:
    source: Stan model source code.

  Returns:
    model: Compiled Stan model.
  """
  filename = FLAGS.stan_model_filename_template.format(
      hashlib.md5(source.encode('ascii')).hexdigest()) + '.stan'
  # We rely on CmdStanPy's internal caching, based on mtimes of the source and
  # binary files. CmdStanPy will not recompile the model if the binary is newer
  # than the source file.
  if not os.path.exists(filename):
    with open(filename, 'w') as f:
      f.write(source)
  model = cmdstanpy.CmdStanModel(stan_file=filename)
  model.compile()
  return model


def get_columns(dataframe, name_regex):
  """Extract relevant columns from a dataframe.

  Args:
    dataframe: A dataframe.
    name_regex: Regex to use to match the columns of the dataframe.

  Returns:
    columns: A Numpy array of the extracted colums.

  Raise:
    ValueError: If no columns were matched.
  """
  col_selector = dataframe.columns.map(
      lambda s: bool(re.fullmatch(name_regex, s)))
  if not np.any(col_selector):
    raise ValueError('Regex "{}" selected no params from {}'.format(
        name_regex, dataframe.columns))
  return np.array(dataframe.loc[:, col_selector])


def make_sample_fn(model, **sample_kwargs):
  """Create a sample function from a model with better error reporting.

  The returned generator generates samples from the model, printing an error if
  something went wrong.

  Args:
    model: A `CmdStanModel`.
    **sample_kwargs: Arguments to pass to `CmdStanModel.sample`.

  Returns:
    sample_fn: A generator with signature
      `(output_dir, **kwargs) -> cmdstanpy.CmdStanMCMC`.
      `output_base_dir` specifies the base directory to use for the temporary
      directory created to hold Stan's outputs.
  """

  @contextlib.contextmanager
  def _sample_fn(output_dir=None, **kwargs):
    """The sample function."""
    # Error reporting isn't great in CmdStanPy yet
    # (https://github.com/stan-dev/cmdstanpy/issues/22), so we do a little work
    # to intercept the console output and print it if there's an issue.
    #
    # We use a context manager because CmdStanPy lazily loads quantities from
    # the output directory, and since we're in control of deleting it, we need
    # to have a mechanism to keep it around until the user is done with it.
    if output_dir is None:
      output_dir = tempfile.mkdtemp()
      keep_outputs = False
    else:
      keep_outputs = True
    final_kwargs = sample_kwargs.copy()
    final_kwargs.update(kwargs)
    if 'data' in final_kwargs:
      # Canonicalize the data to be all NumPy arrays,
      final_kwargs['data'] = {
          k: np.array(v) for k, v in final_kwargs['data'].items()
      }
    try:
      yield model.sample(output_dir=output_dir, **final_kwargs)
    except RuntimeError as e:
      for console_filename in glob.glob(os.path.join(output_dir, '*.txt')):
        with open(console_filename, 'r') as f:
          print(console_filename)
          print(f.read())
      raise e
    finally:
      if not keep_outputs:
        shutil.rmtree(output_dir)

  return _sample_fn
