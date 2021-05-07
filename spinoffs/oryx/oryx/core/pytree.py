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
# Lint as: python3
"""Contains the Pytree class."""
import abc
from jax import tree_util

__all__ = [
    'Pytree',
]


class Pytree(metaclass=abc.ABCMeta):
  """Class that registers objects as Jax pytree_nodes."""

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    tree_util.register_pytree_node(
        cls,
        cls.flatten,
        # Pytype incorrectly thinks that cls.unflatten accepts three arguments.
        cls.unflatten  # type: ignore
    )

  @abc.abstractmethod
  def flatten(self):
    pass

  @abc.abstractclassmethod
  def unflatten(cls, data, xs):
    pass
