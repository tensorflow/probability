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
r"""Ground truth values for `eight_schools`.

Automatically generated using the command:

```
python -m inference_gym.tools.get_ground_truth \
  --target=eight_schools \
  --stan_samples=20000
```
"""

import numpy as np

IDENTITY_AVG_EFFECT_MEAN = np.array([
    3.2137590385402826,
]).reshape(())

IDENTITY_AVG_EFFECT_MEAN_STANDARD_ERROR = np.array([
    0.011278555132469106,
]).reshape(())

IDENTITY_AVG_EFFECT_STANDARD_DEVIATION = np.array([
    3.972567360644285,
]).reshape(())

IDENTITY_LOG_STDDEV_MEAN = np.array([
    2.4598093936288152,
]).reshape(())

IDENTITY_LOG_STDDEV_MEAN_STANDARD_ERROR = np.array([
    0.0018535105349442114,
]).reshape(())

IDENTITY_LOG_STDDEV_STANDARD_DEVIATION = np.array([
    0.5110436981514841,
]).reshape(())

IDENTITY_SCHOOL_EFFECTS_MEAN = np.array([
    13.355582058626903,
    6.102211570244123,
    1.024737484826771,
    5.369929612269985,
    0.7926315948389041,
    2.244649709577478,
    11.763530998898284,
    6.261180511721017,
]).reshape((8,))

IDENTITY_SCHOOL_EFFECTS_MEAN_STANDARD_ERROR = np.array([
    0.02544285870818445,
    0.014957463084240069,
    0.021316316118379103,
    0.01575696229480018,
    0.01373758544135346,
    0.01550588664268995,
    0.017107240447608994,
    0.02209825465307125,
]).reshape((8,))

IDENTITY_SCHOOL_EFFECTS_STANDARD_DEVIATION = np.array([
    10.807906054798417,
    7.766699608891697,
    10.244640099796499,
    8.219198474822052,
    7.303418026739246,
    8.258823240336538,
    8.19256497826584,
    10.673579208829045,
]).reshape((8,))
