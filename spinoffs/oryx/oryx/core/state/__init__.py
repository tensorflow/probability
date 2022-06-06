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
"""Module for stateful functions."""
from oryx.core.state import registrations
from oryx.core.state.api import ArraySpec
from oryx.core.state.api import call
from oryx.core.state.api import call_and_update
from oryx.core.state.api import init
from oryx.core.state.api import make_array_spec
from oryx.core.state.api import Shape
from oryx.core.state.api import spec
from oryx.core.state.api import update
from oryx.core.state.function import kwargs_rules
from oryx.core.state.module import assign
from oryx.core.state.module import ASSIGN
from oryx.core.state.module import Module
from oryx.core.state.module import variable
from oryx.core.state.module import VARIABLE

# Only need these modules for registration
del registrations
