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
"""ConditionalBijector Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from tensorflow_probability.python import bijectors as tfb
tfe = tf.contrib.eager


class _TestBijector(tfb.ConditionalBijector):

  def __init__(self):
    super(_TestBijector, self).__init__(
        forward_min_event_ndims=0,
        graph_parents=[],
        is_constant_jacobian=True,
        validate_args=False,
        dtype=tf.float32,
        name="test_bijector")

  def _forward(self, _, arg1, arg2):
    raise ValueError("forward", arg1, arg2)

  def _inverse(self, _, arg1, arg2):
    raise ValueError("inverse", arg1, arg2)

  def _inverse_log_det_jacobian(self, _, arg1, arg2):
    raise ValueError("inverse_log_det_jacobian", arg1, arg2)

  def _forward_log_det_jacobian(self, _, arg1, arg2):
    raise ValueError("forward_log_det_jacobian", arg1, arg2)


class _TestPassthroughBijector(_TestBijector):
  def __init__(self, *args, **kwargs):
    super(_TestPassthroughBijector, self).__init__(*args, **kwargs)
    self._called = collections.defaultdict(bool)

  def _forward(self, _, arg1, arg2):
    self._called["forward"] = True
    return _

  def _inverse(self, _, arg1, arg2):
    self._called["inverse"] = True
    return _

  def _inverse_log_det_jacobian(self, _, arg1, arg2):
    self._called["inverse_log_det_jacobian"] = True
    return _

  def _forward_log_det_jacobian(self, _, arg1, arg2):
    self._called["forward_log_det_jacobian"] = True
    return _


@tfe.run_all_tests_in_graph_and_eager_modes
class ConditionalBijectorTest(tf.test.TestCase):

  def testConditionalBijector(self):
    b = _TestBijector()
    for name in ["forward", "inverse"]:
      method = getattr(b, name)
      with self.assertRaisesRegexp(ValueError, name + ".*b1.*b2"):
        method(1., arg1="b1", arg2="b2")

    for name in ["inverse_log_det_jacobian", "forward_log_det_jacobian"]:
      method = getattr(b, name)
      with self.assertRaisesRegexp(ValueError, name + ".*b1.*b2"):
        method(1., event_ndims=0, arg1="b1", arg2="b2")

  def testNestedCondition(self):
    b = _TestBijector()
    for name in ["forward", "inverse"]:
      method = getattr(b, name)
      with self.assertRaisesRegexp(
          ValueError, name + ".*{'b1': 'c1'}, {'b2': 'c2'}"):
        method(1., arg1={"b1": "c1"}, arg2={"b2": "c2"})

    for name in ["inverse_log_det_jacobian", "forward_log_det_jacobian"]:
      method = getattr(b, name)
      with self.assertRaisesRegexp(
          ValueError, name + ".*{'b1': 'c1'}, {'b2': 'c2'}"):
        method(1., event_ndims=0, arg1={"b1": "c1"}, arg2={"b2": "c2"})

  def testChainedConditionalBijector(self):
    class ConditionalChain(tfb.ConditionalBijector, tfb.Chain):
      pass

    for name in ["forward", "inverse"]:
      test_bijector = _TestBijector()
      passthrough_bijector = _TestPassthroughBijector()
      chain_components = (test_bijector, passthrough_bijector)
      if name == "inverse":
          chain_components = chain_components[::-1]
      chain = ConditionalChain(chain_components)

      self.assertFalse(passthrough_bijector._called[name])
      method = getattr(chain, name)
      with self.assertRaisesRegexp(ValueError, name + ".*b1.*b2"):
        method(
            1.,
            test_bijector={"arg1": "b1", "arg2": "b2"},
            test_passthrough_bijector={"arg1": "b1", "arg2": "b2"})
      self.assertTrue(passthrough_bijector._called[name])

      ldj_name = name + "_log_det_jacobian"
      self.assertFalse(passthrough_bijector._called[ldj_name])
      method = getattr(chain, ldj_name)
      with self.assertRaisesRegexp(ValueError, ldj_name + ".*b1.*b2"):
        method(
            1.,
            event_ndims=0,
            test_bijector={"arg1": "b1", "arg2": "b2"},
            test_passthrough_bijector={"arg1": "b1", "arg2": "b2"})
      self.assertTrue(passthrough_bijector._called[ldj_name])


if __name__ == "__main__":
  tf.test.main()
