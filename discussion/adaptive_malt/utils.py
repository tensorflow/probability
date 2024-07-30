# Copyright 2022 The TensorFlow Probability Authors.
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
"""Utilities."""

import contextlib
import io
from typing import Any, Dict, Iterator

from absl import flags
from etils import epath
import gin
import h5py
import jax
import numpy as np
import tree
import yaml


def save_h5py(path: str, value: Any):
  """Saves a nested value to h5 using a lossy encoding."""
  # Need to go via a buffer for some filesystems...
  buf = io.BytesIO()
  h5 = h5py.File(buf, 'w')
  for p, v in tree.flatten_with_path(value):
    if v is not None:
      h5.create_dataset('/'.join(map(str, p)), data=np.array(v))
  h5.close()

  with epath.Path(path).open('wb') as f:
    f.write(buf.getvalue())


def load_h5py(path: str):
  """Loads an h5."""
  return h5py.File(epath.Path(path).open('rb'))


def h5_to_dict(h5: h5py.Group) -> Dict[str, Any]:
  """Converts an h5 group to a nested dict."""
  out_dict = {}

  def visitor(path, obj):
    if not isinstance(obj, h5py.Dataset):
      return
    elems = path.split('/')
    cur_dict = out_dict
    for e in elems[:-1]:
      new_dict = cur_dict.get(e)
      if new_dict is None:
        new_dict = {}
        cur_dict[e] = new_dict
      cur_dict = new_dict

    cur_dict[elems[-1]] = np.array(obj)
  h5.visititems(visitor)

  return out_dict


class YAMLDictParser(flags.ArgumentParser):
  syntactic_help = """Expects YAML one-line dictionaries without braces, e.g.
  'key1: val1, key2: val2'."""

  def parse(self, argument: str) -> Dict[str, Any]:
    return yaml.safe_load('{' + argument + '}')

  def flag_type(self):
    return 'Dict[str, Any]'


def bind_hparams(hparams: Dict[str, Any]):
  """Binds all Gin parameters from a dictionary.

  Args:
    hparams: HParams to bind.
  """
  for k, v in hparams.items():
    gin.bind_parameter(k, v)


class BufferIdSet:

  def __init__(self):
    self._arrays = set()

  def add(self, array):
    self._arrays.add(array.unsafe_buffer_pointer())

  def __contains__(self, array):
    return array.unsafe_buffer_pointer() in self._arrays


@contextlib.contextmanager
def delete_device_buffers() -> Iterator[BufferIdSet]:
  """Delete device buffers."""
  buffer_set = BufferIdSet()
  for d in jax.devices():
    for b in d.live_buffers():
      buffer_set.add(b)

  try:
    yield buffer_set
  finally:
    for d in jax.devices():
      for b in list(d.live_buffers()):
        if not b.is_deleted() and b not in buffer_set:
          b.delete()
