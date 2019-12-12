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

# [internal] load python3.bzl

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

FILENAMES = [
    "exp_sin_squared",
    "exponentiated_quadratic",
    "feature_scaled",
    "feature_transformed",
    "kumaraswamy_transformed",
    "matern",
    "polynomial",
    "positive_semidefinite_kernel",
    "rational_quadratic",
    "schur_complement",
]

TEST_FILENAMES = [filename + "_test" for filename in FILENAMES]

TEST_SIZES = {
    "schur_complement_test": "large",
}

[genrule(
    name = "rewrite_{}".format(filename),
    srcs = ["//tensorflow_probability/python/math/psd_kernels:{}.py".format(filename)],
    outs = ["{}.py".format(filename)],
    cmd = "$(location //tensorflow_probability/python/experimental/substrates/meta:rewrite) $(SRCS) --numpy_to_jax > $@",
    tools = ["//tensorflow_probability/python/experimental/substrates/meta:rewrite"],
) for filename in FILENAMES + TEST_FILENAMES + ["__init__"]]

[py_library(
    name = filename,
    srcs = ["{}.py".format(filename)],
    deps = [
        "//tensorflow_probability/python/internal/backend/jax",
    ],
) for filename in FILENAMES + ["__init__"]]

[py_test(
    name = test_filename,
    size = TEST_SIZES.get(test_filename, "medium"),
    srcs = ["{}.py".format(test_filename)],
    tags = ["tfp_jax"],
    deps = [
        ":psd_kernels",
        # jax dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/experimental/substrates/jax/internal:test_util",
    ],
) for test_filename in TEST_FILENAMES]

py_library(
    name = "psd_kernels",
    srcs = ["__init__.py"],
    deps = [":{}".format(filename) for filename in FILENAMES] + [
        "//tensorflow_probability/python/experimental/substrates/jax/math/psd_kernels/internal",
    ],
)
