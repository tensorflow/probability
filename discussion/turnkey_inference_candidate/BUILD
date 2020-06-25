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
# Description:
#   Some Turnkey inference candidates

# [internal] load python3.bzl

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "turnkey_inference_candidate",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":window_tune_nuts_sampling",
    ],
)

py_library(
    name = "window_tune_nuts_sampling",
    srcs = ["window_tune_nuts_sampling.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:assert_util",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:tensorshape_util",
        "//tensorflow_probability/python/mcmc:transformed_kernel",
    ],
)

# py2and3
py_test(
    name = "window_tune_nuts_sampling_test",
    size = "enormous",
    srcs = ["window_tune_nuts_sampling_test.py"],
    shard_count = 1,
    tags = [
        "nozapfhahn",
    ],
    deps = [
        ":turnkey_inference_candidate",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

# py2and3
py_test(
    name = "window_tune_nuts_sampling_test_gpu",
    size = "enormous",
    srcs = ["window_tune_nuts_sampling_test.py"],
    main = "window_tune_nuts_sampling_test.py",
    shard_count = 1,
    tags = [
        "notap",
        "nozapfhahn",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":turnkey_inference_candidate",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)
