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
r"""Ground truth values for `german_credit_numeric_logistic_regression`.

Automatically generated using the command:

```
bazel run //tools/inference_gym_ground_truth:get_ground_truth -- \
  --target \
  german_credit_numeric_logistic_regression \
```
"""

import numpy as np

IDENTITY_MEAN = np.array([
    -0.7351048553686668,
    0.41854235448568405,
    -0.4140361022849266,
    0.12687262544490638,
    -0.36453584119271787,
    -0.1786990235929577,
    -0.1528721119830067,
    0.0130935930161775,
    0.18071618213137836,
    -0.11077840748059746,
    -0.22434837978228872,
    0.12239538160879522,
    0.028775849958589513,
    -0.13628208974727007,
    -0.29222110498210363,
    0.2783575897857832,
    -0.2996277708109526,
    0.30372734184257766,
    0.27038791575592425,
    0.12251564641333557,
    -0.062930540861664,
    -0.09271734036278598,
    -0.025386265018982113,
    -0.022952091856998594,
    -1.2033366774193333,
]).reshape((25,))

IDENTITY_MEAN_STANDARD_ERROR = np.array([
    5.842293909946494e-05,
    7.242951181494356e-05,
    6.287678982885978e-05,
    7.585193280148798e-05,
    6.115211849593741e-05,
    6.021116416974708e-05,
    5.204191507761724e-05,
    5.860998969304511e-05,
    7.29503297927934e-05,
    6.490239025755679e-05,
    4.990373753354614e-05,
    6.283413887066306e-05,
    5.430645722326503e-05,
    6.406386782855579e-05,
    7.892840871425272e-05,
    5.308342894035861e-05,
    6.703376967839617e-05,
    8.521129854167403e-05,
    7.765561215475798e-05,
    0.00010413139262019992,
    0.00010841073917598099,
    6.237296545620734e-05,
    9.654815236395932e-05,
    9.49005719330975e-05,
    6.225181243823337e-05,
]).reshape((25,))

IDENTITY_STANDARD_DEVIATION = np.array([
    0.0898313720177512,
    0.10433392890125515,
    0.09494358976312321,
    0.10821559696336329,
    0.09451801286114327,
    0.09209986501802636,
    0.08194570882231808,
    0.09096249386093944,
    0.10427142118193244,
    0.09706664883314095,
    0.07886872456716118,
    0.09415178440121623,
    0.08568162266412561,
    0.094635647710843,
    0.11794843143366165,
    0.08278578466826157,
    0.10338649406760281,
    0.12112997506000497,
    0.11129990766341216,
    0.13748697192324197,
    0.14311733514054628,
    0.09036915198426924,
    0.12757406812435373,
    0.12488837996746398,
    0.09189586059142167,
]).reshape((25,))
