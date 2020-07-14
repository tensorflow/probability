# Lint as: python2, python3
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
r"""Ground truth values for `german_credit_numeric_probit_regression`.

Automatically generated using the command:

```
bazel run //tools/inference_gym_ground_truth:get_ground_truth --   --target \
  german_credit_numeric_probit_regression \
```
"""

import numpy as np

IDENTITY_MEAN = np.array([
    -0.42902864825,
    0.23932611706587967,
    -0.23685125007101338,
    0.07467150819163507,
    -0.198722638328584,
    -0.10153488526606244,
    -0.08587360309027331,
    0.002650659744971875,
    0.10173055052450779,
    -0.06646613083152295,
    -0.1341139633839806,
    0.07193107984312329,
    0.022796791018150728,
    -0.07875368738959454,
    -0.16711337565911846,
    0.157838408973684,
    -0.18333329919979208,
    0.17101058247977616,
    0.1487687516257999,
    0.05990832429240769,
    -0.05080371653166095,
    -0.05443157021800493,
    -0.017511078065231004,
    -0.019050138137471598,
    -0.7012749627526668,
]).reshape((25,))

IDENTITY_MEAN_STANDARD_ERROR = np.array([
    3.504233693626756e-05,
    4.403228139726661e-05,
    3.801118035766461e-05,
    4.6379315533109946e-05,
    3.566691988155656e-05,
    3.707866134303629e-05,
    3.184911460681449e-05,
    3.6690751676811455e-05,
    4.5646202686336786e-05,
    3.907082547428988e-05,
    3.070191695446625e-05,
    3.7919929533997395e-05,
    3.3962717010011596e-05,
    3.9655855720208416e-05,
    4.3526255112703006e-05,
    3.2565341468434926e-05,
    4.0708152424153056e-05,
    5.14543377443221e-05,
    4.7781716035135036e-05,
    6.511203175479744e-05,
    6.77889443393989e-05,
    3.7398426804605164e-05,
    5.9663082947314444e-05,
    5.810737980946017e-05,
    3.4903683477940505e-05,
]).reshape((25,))

IDENTITY_STANDARD_DEVIATION = np.array([
    0.05109580268794588,
    0.06025955785801464,
    0.05415025021449016,
    0.06188749265520203,
    0.05253495223066425,
    0.05281513364049032,
    0.04724474523844652,
    0.052893039646505366,
    0.06037783194779746,
    0.05509558187498258,
    0.04519737067753448,
    0.05364364896836504,
    0.04969865269559909,
    0.05499799778393664,
    0.06383048313485705,
    0.04784017010002446,
    0.05908997774845863,
    0.06791848056497145,
    0.06330522659138715,
    0.07966572711599877,
    0.08290617965490085,
    0.0515481445390296,
    0.07433370442157407,
    0.07245614766428457,
    0.05064671110061277,
]).reshape((25,))
