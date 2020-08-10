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
"""Experimental Numpy backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow_probability.python.internal.backend.numpy import _utils as utils


__all__ = [
    'functions_run_eagerly',
    'run_functions_eagerly',
]


@functools.partial(utils.copy_docstring, 'tf.config.functions_run_eagerly')
def functions_run_eagerly():
  return False


@functools.partial(utils.copy_docstring, 'tf.config.run_functions_eagerly')
def run_functions_eagerly(run_eagerly):
  if run_eagerly:
    raise NotImplementedError

