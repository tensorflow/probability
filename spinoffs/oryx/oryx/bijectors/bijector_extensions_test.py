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
# Lint as: python3
"""Tests for tensorflow_probability.spinoffs.oryx.bijectors.bijectors_extensions."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as onp

from oryx import bijectors as bb
from oryx import core
from oryx.internal import test_util

BIJECTORS = [
    ('exp', lambda: bb.Exp(), 1., []),  # pylint: disable=unnecessary-lambda
    ('shift', lambda: bb.Shift(3.), 1., [3.]),
    ('Scale', lambda: bb.Scale(2.), 1., [2.]),
    ('transform_diagonal', lambda: bb.TransformDiagonal(bb.Exp()),
     onp.eye(2).astype(onp.float32), []),
    ('invert', lambda: bb.Invert(bb.Exp()), 1., []),
]


class BijectorsExtensionsTest(test_util.TestCase):

  @parameterized.named_parameters(BIJECTORS)
  def test_forward(self, bij, inp, flat):
    del flat
    b = bij()
    b.forward(inp)

  @parameterized.named_parameters(BIJECTORS)
  def test_inverse(self, bij, inp, flat):
    del flat
    b = bij()
    b.inverse(inp)

  @parameterized.named_parameters(BIJECTORS)
  def test_flatten(self, bij, inp, flat):
    del inp
    b = bij()
    flat_p, _ = jax.tree_flatten(b)
    for e1, e2 in zip(flat_p, flat):
      onp.testing.assert_allclose(e1, e2)

  @parameterized.named_parameters(BIJECTORS)
  def test_inverse_transformation(self, bij, inp, flat):
    del flat
    b = bij()
    onp.testing.assert_allclose(core.inverse(b, inp)(inp), b.inverse(inp))

  @parameterized.named_parameters(BIJECTORS)
  def test_ildj_transformation(self, bij, inp, flat):
    del flat
    b = bij()
    onp.testing.assert_allclose(
        core.ildj(b, inp)(inp), b.inverse_log_det_jacobian(inp, onp.ndim(inp)))


if __name__ == '__main__':
  absltest.main()
