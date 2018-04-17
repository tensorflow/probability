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
# Examples of model uncertainty in predictive classifiers.

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "weight_uncertainty",
    srcs = [
        "weight_uncertainty.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_binary(
    name = "logistic_regression",
    srcs = [
        "logistic_regression.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":weight_uncertainty",
        # matplotlib dep,
        # numpy dep,
        # seaborn dep,
        # tensorflow dep,
    ],
)

py_binary(
    name = "mnist_deep_nn",
    srcs = [
        "mnist_deep_nn.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":weight_uncertainty",
        # matplotlib dep,
        # numpy dep,
        # seaborn dep,
        # tensorflow dep,
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

# TODO(b/78137893): Clean this up. For now, we disable tests until
# seaborn/python3.4 dependency issues resolved.
#py_test(
#    name = "mnist_deep_nn_test",
#    size = "small",
#    srcs = [
#        "mnist_deep_nn.py",
#    ],
#    args = [
#        "--fake_data",
#        "--max_steps=5",
#    ],
#    main = "mnist_deep_nn.py",
#    srcs_version = "PY2AND3",
#    deps = [
#        ":mnist_deep_nn",
#    ],
#)
