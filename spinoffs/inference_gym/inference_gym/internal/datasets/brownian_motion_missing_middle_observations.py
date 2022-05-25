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
r"""Brownian Motion data.

This was generated using the following snippet:

```
prior_loc = 0.
innovation_noise = .1
observation_noise = .15
n = 30

Root = tfd.JointDistributionCoroutine.Root

def prior_model():
  new = yield Root(tfd.Normal(loc=prior_loc, scale=innovation_noise))
  for i in range(n-1):
    new = yield tfd.Normal(loc=new, scale=innovation_noise)

prior = tfd.JointDistributionCoroutineAutoBatched(prior_model)

# Take a ground truth sample from the prior
ground_truth = prior.sample()

def likelihood_model():
  for i in range(n):
    if i not in range(10,20):
      yield Root(tfd.Normal(loc=ground_truth[i], scale=observation_noise))

likelihood = tfd.JointDistributionCoroutineAutoBatched(likelihood_model)

# Generate observations by sampling from the likelihood

observed_loc = likelihood.sample(1)

```

Note that the final `observed_loc` is not reproducible across software versions,
hence the output is checked in.

"""

import numpy as np

OBSERVED_LOC = np.array([
    0.21592641, 0.118771404, -0.07945447, 0.037677474, -0.27885845, -0.1484156,
    -0.3250906, -0.22957903, -0.44110894, -0.09830782, np.nan, np.nan, np.nan,
    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -0.8786016,
    -0.83736074, -0.7384849, -0.8939254, -0.7774566, -0.70238715, -0.87771565,
    -0.51853573, -0.6948214, -0.6202789
]).astype(dtype=np.float32)

INNOVATION_NOISE = .1

OBSERVATION_NOISE = .15
