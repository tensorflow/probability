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
"""Tests for interceptor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_probability import edward2 as ed

tfe = tf.contrib.eager


class InterceptorTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      {"cls": ed.Normal, "value": 2., "kwargs": {"loc": 0.5, "scale": 1.}},
      {"cls": ed.Bernoulli, "value": 1, "kwargs": {"logits": 0.}},
  )
  @tfe.run_test_in_graph_and_eager_modes()
  def testInterception(self, cls, value, kwargs):
    def interceptor(f, *fargs, **fkwargs):
      name = fkwargs.get("name", None)
      if name == "rv2":
        fkwargs["value"] = value
      return f(*fargs, **fkwargs)
    rv1 = cls(value=value, name="rv1", **kwargs)
    with ed.interception(interceptor):
      rv2 = cls(name="rv2", **kwargs)
    rv1_value, rv2_value = self.evaluate([rv1.value, rv2.value])
    self.assertEqual(rv1_value, value)
    self.assertEqual(rv2_value, value)

  @tfe.run_test_in_graph_and_eager_modes()
  def testInterceptionException(self):
    def f():
      raise NotImplementedError()
    def interceptor(f, *fargs, **fkwargs):
      return f(*fargs, **fkwargs)
    old_interceptor = ed.get_interceptor()
    with self.assertRaises(NotImplementedError):
      with ed.interception(interceptor):
        f()
    new_interceptor = ed.get_interceptor()
    self.assertEqual(old_interceptor, new_interceptor)


if __name__ == "__main__":
  tf.test.main()
