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
#   TensorFlow Probability examples.

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_binary(
    name = "bayesian_neural_network",
    srcs = ["bayesian_neural_network.py"],
    srcs_version = "PY2AND3",
    deps = [
        # absl/flags dep,
        # matplotlib dep,
        # numpy dep,
        # seaborn dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "bayesian_neural_network_test",
    size = "small",
    srcs = ["bayesian_neural_network.py"],
    args = [
        "--fake_data",
        "--max_steps=5",
    ],
    main = "bayesian_neural_network.py",
    srcs_version = "PY2AND3",
    deps = [
        ":bayesian_neural_network",
    ],
)

py_binary(
    name = "logistic_regression",
    srcs = ["logistic_regression.py"],
    srcs_version = "PY2AND3",
    deps = [
        # absl/flags dep,
        # matplotlib dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "logistic_regression_test",
    size = "small",
    srcs = [
        "logistic_regression.py",
    ],
    args = [
        "--num_examples=32",
        "--batch_size=8",
        "--max_steps=50",
    ],
    main = "logistic_regression.py",
    srcs_version = "PY2AND3",
    deps = [
        ":logistic_regression",
    ],
)

py_binary(
    name = "vae",
    srcs = ["vae.py"],
    main = "vae.py",
    srcs_version = "PY2AND3",
    deps = [
        # absl/flags dep,
        # matplotlib dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_test(
    name = "vae_test",
    size = "small",
    srcs = ["vae.py"],
    args = [
        "--fake_data",
        "--max_steps=5",
    ],
    main = "vae.py",
    srcs_version = "PY2AND3",
    deps = [
        ":vae",
    ],
)
