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
# Example implementation of a simple variational auto-encoder (VAE).

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "vae_model",
    srcs = [
        "vae_model.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_binary(
    name = "vae_main",
    srcs = [
        "main.py",
    ],
    main = "main.py",
    srcs_version = "PY2AND3",
    deps = [
        ":vae_model",
        # matplotlib dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "vae_test",
    size = "small",
    srcs = [
        "main.py",
    ],
    args = [
        "--fake_data",
        "--max_steps=5",
    ],
    main = "main.py",
    srcs_version = "PY2AND3",
    deps = [
        ":vae_main",
    ],
)
