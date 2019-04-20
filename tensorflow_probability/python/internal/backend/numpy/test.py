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
"""Numpy implementations of TensorFlow functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

# Dependency imports
import numpy as np

import tensorflow as tf

__all__ = [
    'TestCase',
]


# --- Begin Public Functions --------------------------------------------------


class TestCase(tf.test.TestCase):
  """Wrapper of `tf.test.TestCase`."""

  def evaluate(self, x):
    return x

  def _GetNdArray(self, a):
    if isinstance(a, (np.generic, np.ndarray)):
      return a
    return np.array(a)

  @contextlib.contextmanager
  def assertRaisesWithPredicateMatch(self,
                                     exception_type,
                                     expected_err_re_or_predicate):
    # pylint: disable=g-doc-return-or-yield
    """Returns a context manager to enclose code expected to raise an exception.

    If the exception is an OpError, the op stack is also included in the message
    predicate search.

    Args:
      exception_type: The expected type of exception that should be raised.
      expected_err_re_or_predicate: If this is callable, it should be a function
        of one argument that inspects the passed-in exception and returns True
        (success) or False (please fail the test). Otherwise, the error message
        is expected to match this regular expression partially.

    Returns:
      A context manager to surround code that is expected to raise an
      exception.
    """
    # pylint: enable=g-doc-return-or-yield
    if not tf.executing_eagerly():
      # Don't test graph mode until we intercept assertions since not
      # all assertions try to operate statically.
      yield
      return
    with super(TestCase, self).assertRaisesWithPredicateMatch(
        exception_type, expected_err_re_or_predicate):
      yield


main = tf.test.main
