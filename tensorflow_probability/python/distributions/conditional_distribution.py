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
"""Conditional distribution base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import distribution_util


class ConditionalDistribution(distribution.Distribution):
  """Distribution that supports intrinsic parameters (local latents).

  Subclasses of this distribution may have additional keyword arguments passed
  to their sample-based methods (i.e. `sample`, `log_prob`, etc.).
  """

  @distribution_util.AppendDocstring(kwargs_dict={
      "**condition_kwargs":
      "Named arguments forwarded to subclass implementation."})
  def sample(self, sample_shape=(), seed=None, name="sample",
             **condition_kwargs):
    return self._call_sample_n(sample_shape, seed, name, **condition_kwargs)

  @distribution_util.AppendDocstring(kwargs_dict={
      "**condition_kwargs":
      "Named arguments forwarded to subclass implementation."})
  def log_prob(self, value, name="log_prob", **condition_kwargs):
    return self._call_log_prob(value, name, **condition_kwargs)

  @distribution_util.AppendDocstring(kwargs_dict={
      "**condition_kwargs":
      "Named arguments forwarded to subclass implementation."})
  def prob(self, value, name="prob", **condition_kwargs):
    return self._call_prob(value, name, **condition_kwargs)

  @distribution_util.AppendDocstring(kwargs_dict={
      "**condition_kwargs":
      "Named arguments forwarded to subclass implementation."})
  def log_cdf(self, value, name="log_cdf", **condition_kwargs):
    return self._call_log_cdf(value, name, **condition_kwargs)

  @distribution_util.AppendDocstring(kwargs_dict={
      "**condition_kwargs":
      "Named arguments forwarded to subclass implementation."})
  def cdf(self, value, name="cdf", **condition_kwargs):
    return self._call_cdf(value, name, **condition_kwargs)

  @distribution_util.AppendDocstring(kwargs_dict={
      "**condition_kwargs":
      "Named arguments forwarded to subclass implementation."})
  def log_survival_function(self, value, name="log_survival_function",
                            **condition_kwargs):
    return self._call_log_survival_function(value, name, **condition_kwargs)

  @distribution_util.AppendDocstring(kwargs_dict={
      "**condition_kwargs":
      "Named arguments forwarded to subclass implementation."})
  def survival_function(self, value, name="survival_function",
                        **condition_kwargs):
    return self._call_survival_function(value, name, **condition_kwargs)
