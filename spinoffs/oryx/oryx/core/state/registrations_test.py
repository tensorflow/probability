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
"""Tests for tensorflow_probability.spinoffs.oryx.core.state.registrations."""

from absl.testing import absltest
from jax import random

from oryx.core.state import api
from oryx.core.state import function
from oryx.core.state import registrations

# Only needed for registration
del function
del registrations


class RegistrationsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._seed = random.PRNGKey(0)

  def test_empty_tuple_init(self):
    with self.assertRaises(ValueError):
      api.init(())(self._seed, api.Shape((50,)))

  def test_func(self):
    id_ = lambda x: x
    in_spec = api.Shape(())
    out_spec = api.spec(id_)(in_spec)
    func = api.init(id_)(self._seed, in_spec)
    self.assertEqual(out_spec, in_spec)
    self.assertEqual(api.call(func, 1.), 1.)

  def test_tuple(self):
    id_ = lambda x: x
    in_spec = api.Shape(())
    out_spec = api.spec((id_, id_, id_))(in_spec)
    tupl = api.init((id_, id_, id_))(self._seed, in_spec)
    self.assertTupleEqual(out_spec, (in_spec,) * 3)
    self.assertTupleEqual(api.call(tupl, 1.), (1., 1., 1.))

  def test_list(self):
    func = lambda x: x + 1
    in_spec = api.Shape(())
    out_spec = api.spec([func, func, func])(in_spec)
    lst = api.init([func, func, func])(self._seed, in_spec)
    self.assertEqual(out_spec, in_spec)
    self.assertEqual(api.call(lst, 1.), 4.)

  def test_list_multiple_args(self):
    func1 = lambda x: (x, x + 1)
    func2 = lambda x, y: (y, x + 1)
    func3 = lambda x, y: (x + 1, y + 2, x + y)
    in_spec = api.Shape(())
    out_spec = api.spec([func1, func2, func3])(in_spec)
    lst = api.init([func1, func2, func3])(self._seed, in_spec)
    self.assertEqual(out_spec, (api.Shape(()),) * 3)
    self.assertTupleEqual(api.call(lst, 1.), (3., 4., 4.))


if __name__ == '__main__':
  absltest.main()
