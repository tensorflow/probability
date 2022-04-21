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
"""Contains logic for serializing and deserializing PytreeTypes."""
import pickle

__all__ = [
    'to_tuple',
    'from_tuple',
    'serialize',
    'deserialize',
]


def to_tuple(obj):
  return (obj.__class__, obj.flatten())


def from_tuple(obj_tuple):
  cls, (xs, data) = obj_tuple
  return cls.unflatten(data, xs)


def serialize(obj):
  """Serializes an object using pickle."""
  return pickle.dumps(to_tuple(obj))


def deserialize(serialized_obj):
  """Deserializes an object using pickle."""
  obj_tuple = pickle.loads(serialized_obj)
  return from_tuple(obj_tuple)
