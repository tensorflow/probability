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
# Description:
#   TensorFlow Probability implements probabilistic and statistical
#   methods, including modeling and Bayesian inference. APIs here are
#   meant to evolve over time.

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "tensorflow_probability",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
#         "//tensorflow_probability/google/staging",  # DisableOnExport
#         "//tensorflow_probability/google/tfp_google",  # DisableOnExport
        "//tensorflow_probability/python",
        "//tensorflow_probability/substrates",
    ],
)

# The public target for `import tensorflow_probability.substrates.jax as tfp`
py_library(
    name = "jax",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow_probability/substrates:jax"],
)

# The public target for `import tensorflow_probability.substrates.numpy as tfp`
py_library(
    name = "numpy",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow_probability/substrates:numpy"],
)
