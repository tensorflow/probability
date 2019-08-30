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
# Util with the numpy backend.

# [internal] load python3.bzl

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

FILENAMES = [
    "seed_stream",
]

TEST_FILENAMES = ["{}_test".format(filename) for filename in FILENAMES]

[genrule(
    name = "rewrite_{}".format(filename),
    srcs = ["//tensorflow_probability/python/util:{}.py".format(filename)],
    outs = ["{}.py".format(filename)],
    cmd = "$(location //tensorflow_probability/python/experimental/substrates/numpy:rewrite) $(SRCS) > $@",
    tools = ["//tensorflow_probability/python/experimental/substrates/numpy:rewrite"],
) for filename in FILENAMES + TEST_FILENAMES]

[py_library(
    name = filename,
    srcs = ["{}.py".format(filename)],
    deps = [
        "//tensorflow_probability/python/internal/backend/numpy",
    ],
) for filename in FILENAMES]

[py_test(
    name = filename,
    size = "medium",
    srcs = ["{}.py".format(filename)],
    tags = ["tfp_numpy"],
    deps = [
        ":util",
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/substrates/numpy/internal:test_case",
    ],
) for filename in TEST_FILENAMES]

py_library(
    name = "util",
    srcs = ["__init__.py"],
    deps = [":{}".format(filename) for filename in FILENAMES],
)
