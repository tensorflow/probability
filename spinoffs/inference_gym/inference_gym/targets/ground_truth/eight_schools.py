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
    5.75975286026516,
]).reshape(())

IDENTITY_AVG_EFFECT_MEAN_STANDARD_ERROR = np.array([
    0.01786934334960865,
]).reshape(())

IDENTITY_AVG_EFFECT_STANDARD_DEVIATION = np.array([
    5.46540575237222,
]).reshape(())

IDENTITY_LOG_STDDEV_MEAN = np.array([
    2.45326548536025,
]).reshape(())

IDENTITY_LOG_STDDEV_MEAN_STANDARD_ERROR = np.array([
    0.0019127444310499044,
]).reshape(())

IDENTITY_LOG_STDDEV_STANDARD_DEVIATION = np.array([
    0.5146604914005865,
]).reshape(())

IDENTITY_SCHOOL_EFFECTS_MEAN = np.array([
    14.76492662128668,
    7.1559568069514174,
    2.5889680578429823,
    6.556760136283709,
    1.8189620982794488,
    3.3973176011560953,
    12.791827392809507,
    7.94913889128149,
]).reshape((8,))

IDENTITY_SCHOOL_EFFECTS_MEAN_STANDARD_ERROR = np.array([
    0.024677770813437375,
    0.014741663125129497,
    0.02289079528770855,
    0.015999446293377167,
    0.014481515793592113,
    0.016516334290635457,
    0.01649735808006444,
    0.02296023745612875,
]).reshape((8,))

IDENTITY_SCHOOL_EFFECTS_STANDARD_DEVIATION = np.array([
    10.796464460046044,
    7.813843501513539,
    10.473283439128995,
    8.314419652710829,
    7.45775465347688,
    8.454005086842871,
    8.16339794409753,
    10.913796824041532,
]).reshape((8,))
