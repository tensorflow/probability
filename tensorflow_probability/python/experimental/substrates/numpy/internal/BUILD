# Copyright 2019 The TensorFlow Probability Authors.
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
# TFP internals with the numpy backend.

# [internal] load python3.bzl

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

FILENAMES = [
    "assert_util",
    "distribution_util",
    "dtype_util",
    "hypothesis_testlib",
    "prefer_static",
    "special_math",
    "tensor_util",
    "test_case",
    "test_util",
]

TAP_WHITELIST = ["{}_test".format(f) for f in (
    "prefer_static",
    "test_util",
)]

TEST_FILENAMES = ["{}_test".format(filename) for filename in FILENAMES if filename not in ("assert_util",)]

DEPS = {
    "distribution_util": [":prefer_static"],
    "prefer_static_test": [":test_util"],
    "test_util": [
        "//tensorflow_probability/python/experimental/substrates/numpy/distributions:seed_stream",
    ],
    "test_util_test": [
        ":test_util",
    ],
    "special_math": [
        "//tensorflow_probability/python/experimental/substrates/numpy/math",
    ],
    "distribution_util_test": [
        "//tensorflow_probability/python/experimental/substrates/numpy",
    ],
    "hypothesis_testlib_test": [
        ":hypothesis_testlib",
    ],
}

[genrule(
    name = "rewrite_{}".format(filename),
    srcs = ["//tensorflow_probability/python/internal:{}.py".format(filename)],
    outs = ["{}.py".format(filename)],
    cmd = "$(location //tensorflow_probability/python/experimental/substrates/numpy:rewrite) $(SRCS) > $@",
    tools = ["//tensorflow_probability/python/experimental/substrates/numpy:rewrite"],
) for filename in FILENAMES + TEST_FILENAMES]

[py_library(
    name = filename,
    testonly = 1 if "test" in filename else 0,
    srcs = ["{}.py".format(filename)],
    deps = [
        # absl/flags dep,
        "//tensorflow_probability/python/internal/backend/numpy",
        "//tensorflow_probability/python/internal:reparameterization",
        "//tensorflow_probability/python/internal:name_util",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/util",
    ] + DEPS.get(filename, []),
) for filename in FILENAMES]

[py_test(
    name = filename,
    size = "medium",
    srcs = ["{}.py".format(filename)],
    tags = ([] if filename in TAP_WHITELIST else ["notap"]) + ["tfp_numpy"],
    deps = [
        ":internal",
        ":test_case",
        # absl/testing:flagsaver dep,
        # absl/testing:parameterized dep,
        # hypothesis dep,
        # numpy dep,
        # six dep,
    ] + DEPS.get(filename, []),
) for filename in TEST_FILENAMES]

py_library(
    name = "internal",
    srcs = ["__init__.py"],
    deps = [":{}".format(filename) for filename in FILENAMES if "test" not in filename],
)
