# Copyright 2024 The TensorFlow Probability Authors.
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
"""Test utilities for Electric Sheep."""

import functools
import math
import re

from absl import flags
import jax
import numpy as np
from tensorflow_probability.substrates.jax.internal import test_util

from absl.testing import absltest

FLAGS = flags.FLAGS

__all__ = [
    'TestCase',
    'main',
]


class _LeafIndicator:

  def __init__(self, v):
    self.v = v

  def __repr__(self):
    return self.v


class TestCase(test_util.TestCase):
  """Electric Sheep TestCase."""

  def test_seed(self, *args, **kwargs):
    return test_util.test_seed(*args, **kwargs)

  def assertAllEqual(self, a, b, msg=''):
    np.testing.assert_array_equal(a, b, err_msg=msg)

  def assertAllClose(self, a, b, rtol=1e-6, atol=1e-6, msg=''):
    assert_fn = functools.partial(
        np.testing.assert_allclose, rtol=rtol, atol=atol, err_msg=msg
    )

    exceptions = []

    def assert_part(a, b):
      try:
        assert_fn(a, b)
        return _LeafIndicator('.')
      except Exception as e:  # pylint: disable=broad-except
        exceptions.append(e)
        return _LeafIndicator(f'#{len(exceptions)}')

    positions = jax.tree.map(assert_part, a, b)

    if exceptions:
      lines = [
          'Some leaves are not close. Differing leaves:\n',
          f'{positions}\n',
      ]

      for i, e in enumerate(exceptions):
        lines.append(f'Exception #{i + 1}:')
        lines.append(str(e))

      raise AssertionError('\n'.join(lines))

  def assertNear(self, f1, f2, err, msg=None):
    if isinstance(f1, jax.Array):
      f1 = float(f1.item())
    if isinstance(f2, jax.Array):
      f2 = float(f2.item())
    self.assertTrue(
        f1 == f2 or math.fabs(f1 - f2) <= err,
        '%f != %f +/- %f%s'
        % (f1, f2, err, ' (%s)' % msg if msg is not None else ''),
    )


class _TestLoader(absltest.TestLoader):
  """A custom TestLoader that allows for Regex filtering test cases."""

  def getTestCaseNames(self, testCaseClass):  # pylint:disable=invalid-name
    names = super().getTestCaseNames(testCaseClass)
    if FLAGS.test_regex:  # This flag is defined in TFP's test_util.
      pattern = re.compile(FLAGS.test_regex)
      names = [
          name
          for name in names
          if pattern.search(f'{testCaseClass.__name__}.{name}')
      ]
    # Remove the test_seed, as it's not a test despite starting with `test_`.
    names = [name for name in names if name != 'test_seed']
    return names


def main():
  """Test main function that injects a custom loader."""
  absltest.main(testLoader=_TestLoader())
