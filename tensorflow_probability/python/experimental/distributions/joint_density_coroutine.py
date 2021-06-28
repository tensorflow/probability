# Copyright 2021 The TensorFlow Probability Authors.
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
"""The `JointDensityCoroutine` class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import joint_distribution as jd_lib
from tensorflow_probability.python.experimental.distribute import joint_distribution as jdc_lib


class JointDensityCoroutine(object):
  """Joint density parameterized by a distribution-making generator.

  This density enables unnormalized joint density computation from a single
  model specification.

  A joint density is like a joint distribution, except we allow for a density
  to be unnormalized (total probability does not integrate to 1).  As such,
  the `log_prob` method is not implemented; to compute the unnormalized log
  density, use method `unnormalized_log_prob`.

  In addition to accepting the usual distributions like
  JointDistributionCoroutine, JointDensityCoroutine also accepts the
  distribution-like IncrementLogProb to adjust the overall unnormalized
  log-density in arbitrary ways.

  As an example, suppose we wish to increase the log probability of a simple
  model.  We might do so as follows:

  ```python
  tfd = tfp.distributions
  tfde = tfp.experimental.distributions

  Root = tfd.JointDistributionCoroutine.Root  # Convenient alias.
  def model():
    w = yield Root(tfd.Normal(0., 1.))
    yield tfd.Normal(w, 1.)
    yield Root(tfde.IncrementLogProb(5.))

  joint = tfd.JointDensityCoroutine(model)

  x = joint.sample()
  # ==> x is a length-3 tuple of Tensors representing a draw/realization from
  #     each distribution, where the IncrementLogProb is represented by an
  #     empty tensor (which is essentially a single nonvarying point).
  joint.unnormalized_log_prob(x)
  # ==> A scalar `Tensor` representing the total log prob under the two
  #     distributions, as well as including the offset effect of the
  #     IncrementLogProb(5.).
  ```
  """

  def __init__(self, *args, name='JointDensityCoroutine', **kwargs):
    """Construct the `JointDensityCoroutine` density.

    See the documentation for JointDistributionCoroutine

    Args:
      *args: Positional arguments forwarded to JointDistributionCoroutine.
      name: The name for ops managed by the density.
        Default value: `JointDensityCoroutine`.
      **kwargs: Named arguments forwarded to JointDistributionCoroutine.
    """
    with tf.name_scope(name) as name:
      self._joint_distribution_coroutine = jdc_lib.JointDistributionCoroutine(
          name=name, *args, **kwargs)

  @property
  def dtype(self):
    return self._joint_distribution_coroutine.dtype

  def unnormalized_log_prob(self, *args, **kwargs):
    """Unnormalized log probability density/mass function.

    Args:
      *args: Positional arguments forwarded to superclass implementation, which
        likely include 'value' or possibly 'name'.
      **kwargs: Named arguments forwarded to superclass implementation.

    Returns:
      unnormalized_log_prob: a `Tensor` of shape `sample_shape(x) +
      self.batch_shape` with values of type `self.dtype`.
    """
    kwargs['name'] = kwargs.get('name', 'unnormalized_log_prob')
    # pylint: disable=protected-access
    value, _ = jd_lib._resolve_value_from_args(
        args,
        kwargs,
        dtype=self._joint_distribution_coroutine.dtype,
        flat_names=self._joint_distribution_coroutine._flat_resolve_names(),
        model_flatten_fn=self._joint_distribution_coroutine._model_flatten,
        model_unflatten_fn=self._joint_distribution_coroutine._model_unflatten)

    return sum(
        self._joint_distribution_coroutine._map_measure_over_dists(
            'unnormalized_log_prob', value))

  def sample(self, sample_shape=(), seed=None, name='sample', **kwargs):
    """Generate samples of the specified shape.

    Note that a call to `sample()` without arguments will generate a single
    sample.

    Args:
      sample_shape: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
      name: name to give to the op.
      **kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      samples: a `Tensor` with prepended dimensions `sample_shape`.
    """
    return self._joint_distribution_coroutine.sample(
        sample_shape=sample_shape, seed=seed, name=name, **kwargs)
