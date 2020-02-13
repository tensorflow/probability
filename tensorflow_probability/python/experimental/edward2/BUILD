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
#   Edward2 probabilistic programming language.

# [internal] load python3.bzl

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "edward2",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":generated_random_variables",
        ":interceptor",
        ":program_transformations",
        ":random_variable",
    ],
)

py_library(
    name = "generated_random_variables",
    srcs = ["generated_random_variables.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":interceptor",
        ":random_variable",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:docstring_util",
    ],
)

py_test(
    name = "generated_random_variables_test",
    size = "small",
    srcs = ["generated_random_variables_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "interceptor",
    srcs = ["interceptor.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "interceptor_test",
    size = "small",
    srcs = ["interceptor_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "program_transformations",
    srcs = ["program_transformations.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":interceptor",
        # tensorflow dep,
    ],
)

py_test(
    name = "program_transformations_test",
    size = "small",
    srcs = ["program_transformations_test.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "random_variable",
    srcs = ["random_variable.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "random_variable_test",
    size = "small",
    srcs = ["random_variable_test.py"],
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)
