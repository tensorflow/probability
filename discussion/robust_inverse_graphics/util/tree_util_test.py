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
from typing import Any, NamedTuple

from flax import struct
import jax
import jax.numpy as jnp

from discussion.robust_inverse_graphics.util import test_util
from discussion.robust_inverse_graphics.util import tree_util


@struct.dataclass
class Example:
  a: Any
  b: Any


class Example2(NamedTuple):
  x: Any


class TreeUtilTest(test_util.TestCase):

  def test_dataclass_view(self):
    e = Example(a=1, b=2)
    v_e = tree_util.DataclassView(e, lambda n: n == 'a')
    v_e = jax.tree.map(lambda x: x + 1, v_e)
    self.assertAllEqual(Example(a=2, b=2), v_e.value)

  def test_get_element(self):
    tree = Example(a=[0, [1, 2]], b=Example2(x=(3,)))

    self.assertEqual(0, tree_util.get_element(tree, ['a', 0]))
    self.assertEqual([1, 2], tree_util.get_element(tree, ['a', 1]))
    self.assertEqual(Example2(x=(3,)), tree_util.get_element(tree, ['b']))
    self.assertEqual(3, tree_util.get_element(tree, ['b', 'x', 0]))

  def test_update_element(self):
    tree = Example(a=[0, jnp.array([1, 2])], b=Example2(x=(3,)))

    tree2 = tree_util.update_element(tree, ['a', 1], lambda x: x + 1)
    self.assertAllClose(jnp.array([2, 3]), tree2.a[1])

    tree2 = tree_util.update_element(tree, ['b', 'x', 0], lambda x: x + 1)
    self.assertEqual((4,), tree2.b.x)


if __name__ == '__main__':
  test_util.main()
