# Lint as: python3
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
r"""Convection Lorenz Bridge data.

This was generated using the following snippet:

```
Root = tfd.JointDistributionCoroutine.Root

def make_lorenz_system(observation_scale, innovation_scale, num_timesteps,
                       step_size=1.):
  def model():
    x = yield Root(tfd.Normal(0.,i 1.))
    y = yield Root(tfd.Normal(0., 1.))
    z = yield Root(tfd.Normal(0., 1.))
    yield tfd.Normal(x, observation_scale)
    for _ in tf.range(num_timesteps - 1):
      dx = 10 * (y - x)
      dy = x * (28 - z) - y
      dz = x * y - 8./3. * z
      x = yield tfd.Normal(x + step_size * dx,
                           tf.sqrt(step_size) * innovation_scale)
      y = yield tfd.Normal(y + step_size * dy,
                           tf.sqrt(step_size) * innovation_scale)
      z = yield tfd.Normal(z + step_size * dz,
                           tf.sqrt(step_size) * innovation_scale)
      yield tfd.Normal(x, observation_scale)
  return tfd.JointDistributionCoroutine(model)

samples = tf.convert_to_tensor(
    make_lorenz_system(1., 0.1, 30, step_size=0.02).sample(
        seed=tf.zeros(2, tf.int32)))
observed_values = tf.transpose(tf.reshape(samples, [-1, 4]), [1, 0])[..., 0]
observed_values = observed_values.numpy()
observed_values[10:20] = np.nan
```

Note that the final `observed_values` is not reproducible across software
versions, hence the output is checked in.

"""

import numpy as np

OBSERVED_VALUES = np.array([
    -0.2761459, 0.18631345, 0.1467675, -1.3148443, -1.2150469, -0.44544014,
    -0.5505127, -0.9422926, -1.9986963, 0.13876402, np.nan, np.nan, np.nan,
    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -16.095385,
    -18.901144, -21.515736, -22.736586, -23.451488, -21.417793, -15.236895,
    -7.6766376, -0.19389218, 6.26647
]).astype(dtype=np.float32)

OBSERVATION_INDEX = 0

OBSERVATION_MASK = ~np.isnan(OBSERVED_VALUES)

INNOVATION_SCALE = .1

OBSERVATION_SCALE = 1.

STEP_SIZE = 0.02
