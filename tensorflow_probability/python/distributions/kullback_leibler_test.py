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
"""Tests for distributions KL mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import test_util
# pylint: disable=protected-access

_DIVERGENCES = kullback_leibler._DIVERGENCES
_registered_kl = kullback_leibler._registered_kl

# pylint: enable=protected-access


@test_util.test_all_tf_execution_regimes
class KLTest(test_util.TestCase):

  def testRegistration(self):

    class MyDist(normal.Normal):
      pass

    # Register KL to a lambda that spits out the name parameter
    @kullback_leibler.RegisterKL(MyDist, MyDist)
    def _kl(a, b, name=None):  # pylint: disable=unused-argument,unused-variable
      return name

    a = MyDist(loc=0.0, scale=1.0)
    self.assertEqual("OK", kullback_leibler.kl_divergence(a, a, name="OK"))

  def testDomainErrorExceptions(self):

    class MyDistException(normal.Normal):
      pass

    # Register KL to a lambda that spits out the name parameter
    @kullback_leibler.RegisterKL(MyDistException, MyDistException)
    # pylint: disable=unused-argument,unused-variable
    def _kl(a, b, name=None):
      return tf.identity([float("nan")])

    # pylint: disable=unused-argument,unused-variable

    a = MyDistException(loc=0.0, scale=1.0, allow_nan_stats=False)
    with self.assertRaisesOpError(
        "KL calculation between .* and .* returned NaN values"):
      kl = kullback_leibler.kl_divergence(a, a, allow_nan_stats=False)
      self.evaluate(kl)
    with self.assertRaisesOpError(
        "KL calculation between .* and .* returned NaN values"):
      self.evaluate(a.kl_divergence(a))
    a = MyDistException(loc=0.0, scale=1.0, allow_nan_stats=True)
    kl_ok = kullback_leibler.kl_divergence(a, a)
    self.assertAllEqual([float("nan")], self.evaluate(kl_ok))
    self_kl_ok = a.kl_divergence(a)
    self.assertAllEqual([float("nan")], self.evaluate(self_kl_ok))
    cross_ok = a.cross_entropy(a)
    self.assertAllEqual([float("nan")], self.evaluate(cross_ok))

  def testRegistrationFailures(self):

    class MyDist(normal.Normal):
      pass

    with self.assertRaisesRegexp(TypeError, "must be callable"):
      kullback_leibler.RegisterKL(MyDist, MyDist)("blah")

    # First registration is OK
    kullback_leibler.RegisterKL(MyDist, MyDist)(lambda a, b: None)

    # Second registration fails
    with self.assertRaisesRegexp(ValueError, "has already been registered"):
      kullback_leibler.RegisterKL(MyDist, MyDist)(lambda a, b: None)

  def testExactRegistrationsAllMatch(self):
    for (k, v) in _DIVERGENCES.items():
      self.assertEqual(v, _registered_kl(*k))

  def _testIndirectRegistration(self, fn):

    class Sub1(normal.Normal):

      def entropy(self):
        return ""

    class Sub2(normal.Normal):

      def entropy(self):
        return ""

    class Sub11(Sub1):

      def entropy(self):
        return ""

    # pylint: disable=unused-argument,unused-variable
    @kullback_leibler.RegisterKL(Sub1, Sub1)
    def _kl11(a, b, name=None):
      return "sub1-1"

    @kullback_leibler.RegisterKL(Sub1, Sub2)
    def _kl12(a, b, name=None):
      return "sub1-2"

    @kullback_leibler.RegisterKL(Sub2, Sub1)
    def _kl21(a, b, name=None):
      return "sub2-1"

    # pylint: enable=unused-argument,unused_variable

    sub1 = Sub1(loc=0.0, scale=1.0)
    sub2 = Sub2(loc=0.0, scale=1.0)
    sub11 = Sub11(loc=0.0, scale=1.0)

    self.assertEqual("sub1-1", fn(sub1, sub1))
    self.assertEqual("sub1-2", fn(sub1, sub2))
    self.assertEqual("sub2-1", fn(sub2, sub1))
    self.assertEqual("sub1-1", fn(sub11, sub11))
    self.assertEqual("sub1-1", fn(sub11, sub1))
    self.assertEqual("sub1-2", fn(sub11, sub2))
    self.assertEqual("sub1-1", fn(sub11, sub1))
    self.assertEqual("sub1-2", fn(sub11, sub2))
    self.assertEqual("sub2-1", fn(sub2, sub11))
    self.assertEqual("sub1-1", fn(sub1, sub11))

  def testIndirectRegistrationKLFun(self):
    self._testIndirectRegistration(kullback_leibler.kl_divergence)

  def testIndirectRegistrationKLSelf(self):
    self._testIndirectRegistration(
        lambda p, q: p.kl_divergence(q))

  def testIndirectRegistrationCrossEntropy(self):
    self._testIndirectRegistration(
        lambda p, q: p.cross_entropy(q))

  def testFunctionCrossEntropy(self):
    self._testIndirectRegistration(kullback_leibler.cross_entropy)

  def testMissing(self):

    class MyDist(distribution_lib.Distribution):

      def __init__(self):
        super(MyDist, self).__init__(
            dtype=tf.float32,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            validate_args=True,
            allow_nan_stats=True)

    with self.assertRaisesRegexp(NotImplementedError,
                                 "No KL(distribution_a || distribution_b)"):
      kullback_leibler.kl_divergence(MyDist(), MyDist())


if __name__ == "__main__":
  tf.test.main()
