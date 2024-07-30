# Copyright 2024 The TensorFlow Probability Authors.
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
"""Tree2 implementation.

Example usage:

```python
registry = tree2.Registry(allow_unknown_types=True)

@registry.auto_register_type("MyClass")  # Unique tag.
@dataclasses.dataclass
class MyClass:
  a: int
  b: float

c = MyClass(1, 2.)

registry.save_tree(c, '/tmp/c.tree2')

c2 = registry.load_tree('/tmp/c.tree2')

assert c.a == c2.a
assert c.b == c2.b
```
"""

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import enum
import functools
import io
import json
from typing import Any, BinaryIO, Callable, Generic, Optional, Type, TypeVar, Union
import warnings

from etils import epath
import immutabledict
import numpy as np

_TREE2_TAG = 'TREE2'
_BLOCKS_TAG = 'BLOCKS'
_TREE2_TYPE_TAG = 'type'
_SAVE_VERSION = 1
_JSON_SAFE_TYPES = frozenset({str, int, float, bool, type(None)})

UNKNOWN_SEQUENCE = 'unknown_sequence'
UNKNOWN_MAPPING = 'unknown_mapping'
UNKNOWN_NAMEDTUPLE = 'unknown_namedtuple'
UNKNOWN_DATACLASS = 'unknown_dataclass'
ARRAY = 'array'
SCALAR = 'scalar'

Tree = TypeVar('Tree')
EncodedTree = TypeVar('EncodedTree')
InnerEncodeFn = Callable[[Tree], EncodedTree]
EncodeFn = Callable[[Tree, 'Context', InnerEncodeFn], EncodedTree]
DecodeFn = Callable[[EncodedTree, 'Context'], Tree]
DetectFn = Callable[[Any, 'Context'], Optional[str]]

__all__ = [
    'ARRAY',
    'Context',
    'DecodeFn',
    'DeferredNumpyArray',
    'DetectFn',
    'EncodedTree',
    'EncodeFn',
    'InnerEncodeFn',
    'Registry',
    'SCALAR',
    'Tree',
    'UNKNOWN_MAPPING',
    'UNKNOWN_NAMEDTUPLE',
    'UNKNOWN_SEQUENCE',
]


class DeferredNumpyArray:
  """A numpy-array like class that defers disk IO until accessed."""

  def __init__(self, filename: str, offset: int, shape: tuple[int, ...],
               dtype: np.dtype):
    """Creates a deferred numpy array.

    Args:
      filename: Filename.
      offset: Byte offset into the file.
      shape: Shape of the stored array.
      dtype: Dtype of the stored array.
    """
    self._filename = filename
    self._offset = offset
    self._shape = shape
    self._dtype = dtype
    self._value = None

  @property
  def shape(self) -> tuple[int, ...]:
    return self._shape

  @property
  def dtype(self) -> np.dtype:
    return self._dtype

  def __repr__(self) -> str:
    return (f'{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype},'
            f'numpy={self._value})')

  def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if self._value is None:
      with epath.Path(self._filename).open('rb') as f:
        f.seek(self._offset)
        self._value = np.load(f, allow_pickle=False)
    return self._value.astype(dtype)


class Context:
  """Context used for tree serialization and deserialization."""

  def __init__(self, version: int, filename: Optional[str],
               options: Mapping[str, Any]):
    """Creates the tree context.

    Args:
      version: Version of the protocol.
      filename: Name of the file we're writing to, used for deferred numpy
        loading.
      options: Serialization and desearialization. See `save_tree` for valid
        options.
    """
    combined_options = {
        'tree_format': 'json',
        'block_format': 'cat_npy',
        'defer_numpy': False,
    }
    combined_options.update(options)
    self._arrays = {}
    self._array_offsets = {}
    self._version = version
    self._options = combined_options
    self._filename = filename

  def add_array(self, array: np.ndarray) -> str:
    """Adds an array to be saved. Returns its name."""
    name = f'buf{len(self._arrays)}'
    self._arrays[name] = array
    return name

  def get_array(self, name: str, shape: tuple[int, ...],
                dtype: np.dtype) -> Union[None, DeferredNumpyArray, np.ndarray]:
    """Returns an array given its name, shape and dtype."""
    if self.options['defer_numpy']:
      offset = self._array_offsets.get(name)
      if offset is None:
        return None
      else:
        if self._filename is None:
          raise ValueError(
              'Cannot defer numpy loading if loading a tree from a file '
              'object.')
        return DeferredNumpyArray(self._filename, offset, shape, dtype)
    else:
      return self._arrays.get(name)

  def save_blocks(self, f: BinaryIO):
    """Save blocks to a file."""
    if self._arrays:
      block_format = self.options['block_format']
      if block_format == 'cat_npy':
        f.write(f'{_BLOCKS_TAG},{block_format}\n'.encode('utf-8'))

        array_buf = io.BytesIO()
        offsets = {}
        for k, v in self._arrays.items():
          offsets[k] = array_buf.tell()
          np.save(array_buf, v, allow_pickle=False)
        f.write(json.dumps(offsets, sort_keys=True).encode('utf-8'))
        f.write(b'\n')
        f.write(array_buf.getvalue())
      else:
        raise ValueError(f'Unknown block format: {block_format}')

  def load_blocks(self, f: BinaryIO, block_format: str):
    """Load blocks from a file."""
    if block_format == 'cat_npy':
      offsets = json.loads(f.readline().decode('utf-8'))
      global_offset = f.tell()
      defer_numpy = self.options['defer_numpy']
      for k, offset in offsets.items():
        if defer_numpy:
          self._array_offsets[k] = global_offset + offset
        else:
          f.seek(global_offset + offset)
          self._arrays[k] = np.load(f, allow_pickle=False)
    else:
      raise ValueError(f'Unknown block format: {block_format}')

  @property
  def options(self) -> Mapping[str, Any]:
    return self._options

  @property
  def version(self) -> int:
    return self._version


@dataclasses.dataclass
class _TreeSerialization(Generic[Tree, EncodedTree]):
  encode_fn: EncodeFn
  decode_fn: DecodeFn


def _get_first_tag(tag: Union[str, Sequence[str]]) -> str:
  if isinstance(tag, str):
    return tag
  else:
    return tag[0]


@dataclasses.dataclass
class Registry:
  """Type registry for Tree2.

  Attributes:
    interactive_mode: Whether to construct the registry in interactive mode,
      where duplicate registration errors are turned to warnings.
    allow_unknown_types: Whether to allow saving/loading unknown types.
    save_version: Which format version to save.
  """

  interactive_mode: bool = False
  allow_unknown_types: bool = False
  save_version: int = _SAVE_VERSION

  _tag_to_serializer: dict[str, _TreeSerialization] = dataclasses.field(
      default_factory=dict)
  _tree_type_to_tag: dict[Type[Any], Union[str, Sequence[str]]] = (
      dataclasses.field(default_factory=dict))
  _detectors: collections.OrderedDict[Union[Type[Any], str], DetectFn] = (
      dataclasses.field(default_factory=collections.OrderedDict))

  def __post_init__(self):
    self.register_sequence_type('list', fallback=None)(list)
    self.register_sequence_type('tuple', fallback=None)(tuple)
    self.register_sequence_type('set', fallback=None)(set)
    self.register_sequence_type('fronzenset', fallback=None)(frozenset)
    if self.allow_unknown_types:
      self.register_tags(UNKNOWN_SEQUENCE, _encode_unknown_sequence,
                         _decode_unknown_sequence)
      self.register_detector(UNKNOWN_SEQUENCE, _detect_unknown_sequence)

    self.register_mapping_type('dict', fallback=None)(dict)
    self.register_mapping_type(
        'immutabledict', fallback=None)(
            immutabledict.immutabledict)
    if self.allow_unknown_types:
      self.register_tags(UNKNOWN_MAPPING, _encode_unknown_mapping,
                         _decode_unknown_mapping)
      self.register_detector(UNKNOWN_MAPPING, _detect_unknown_mapping)

      self.register_tags(UNKNOWN_NAMEDTUPLE, _encode_unknown_namedtuple,
                         _decode_unknown_namedtuple)
      self.register_detector(UNKNOWN_NAMEDTUPLE, _detect_unknown_namedtuple)
      self.register_tags(UNKNOWN_DATACLASS, _encode_unknown_dataclass,
                         _decode_unknown_dataclass)
      self.register_detector(UNKNOWN_DATACLASS, _detect_unknown_dataclass)

    self._register_numpy()
    self._maybe_register_flax()
    self._maybe_register_jax()
    self._maybe_register_tensorflow()
    self._maybe_register_tfp()

  def register_type(self, tag: Union[str, Sequence[str]], tree_type: Type[Tree],
                    encode_fn: EncodeFn, decode_fn: DecodeFn) -> Type[Tree]:
    """Registers a type.

    Args:
      tag: One or more tags for this type. The first tag, if more than one, is
        used for serialization.
      tree_type: The type of the tree.
      encode_fn: Encoding function.
      decode_fn: Decoding function.

    Returns:
      Same value as `tree_type`.
    """
    if isinstance(tag, str):
      tags = [tag]
    else:
      tags = tag

    if tree_type is not None:
      existing_tags = self._tree_type_to_tag.get(tree_type)
      if existing_tags is not None:
        msg = f'Type \'{tree_type}\' is already registered.'
        if self.interactive_mode:
          warnings.warn(msg)
          for tag in existing_tags:
            del self._tag_to_serializer[tag]
        else:
          raise TypeError(msg)

      self._tree_type_to_tag[tree_type] = tags

    self.register_tags(tags, encode_fn, decode_fn)
    return tree_type

  def register_tags(self, tag: Union[str, Sequence[str]], encode_fn: EncodeFn,
                    decode_fn: DecodeFn):
    """Registers encode and decode functions for tags.

    This is typically paired with `register_detector`.

    Args:
      tag: One or more tags for this type. The first tag, if more than one, is
        used for serialization.
      encode_fn: Encoding function.
      decode_fn: Decoding function.
    """
    if isinstance(tag, str):
      tags = [tag]
    else:
      tags = tag

    for tag in tags:
      if tag in self._tag_to_serializer:
        msg = f'Tag \'{tag}\' is already registered.'
        if self.interactive_mode:
          warnings.warn(msg)
        else:
          raise ValueError(msg)
      self._tag_to_serializer[tag] = _TreeSerialization(encode_fn, decode_fn)

  def register_detector(self, type_hint_or_id: Union[Type[Any], str],
                        detector_fn: DetectFn):
    """Registers a detector function.

    Given a tree, this detects which tag to assign it for serialization
    purposes. The `type_hint_or_id` can either be used as a type hint, meaning
    that if a tree has that type, the detector is called on it. If
    `type_hint_or_id` is a string, this acts merely as re-registration id.

    Args:
      type_hint_or_id: Idenfitier of this detector or a type hint.
      detector_fn: The detector function.
    """
    if type_hint_or_id in self._detectors:
      msg = f'Detector \'{type_hint_or_id}\' is already registered.'
      if self.interactive_mode:
        warnings.warn(msg)
      else:
        raise TypeError(msg)

    self._detectors[type_hint_or_id] = detector_fn

  def _json_obj_hook(self, obj: dict[str, Any], ctx: Context) -> Any:
    """Object loading hook for JSON."""
    tag = obj.get(_TREE2_TYPE_TAG)
    if tag is None:
      return obj

    serializer = self._tag_to_serializer.get(tag)
    if serializer is None:
      fallback = obj.get('fallback_type')
      if fallback is None:
        raise ValueError(f'Unknown tree type with no fallback: {obj}')
      serializer = self._tag_to_serializer.get(fallback)
      if serializer is None:
        raise ValueError(f'Tree type with unknown fallback type: {obj}')

    return serializer.decode_fn(obj, ctx)

  def _encode_tree(self, tree: Any, ctx: Context) -> Any:
    """Encode a tree into JSON-safe format."""
    tree_type = type(tree)
    if tree_type in _JSON_SAFE_TYPES:
      return tree
    tags = self._tree_type_to_tag.get(tree_type)
    if tags is None:
      detector_fn = self._detectors.get(tree_type)
      if detector_fn is None:
        for detector_fn in reversed(list(self._detectors.values())):
          tag = detector_fn(tree, ctx)
          if tag is not None:
            break
        else:
          raise TypeError(f'Unknown tree type: {tree}')
      else:
        tag = detector_fn(tree, ctx)
    else:
      tag = tags[0]

    encode_fn = self._tag_to_serializer[tag].encode_fn
    return encode_fn(tree, ctx, functools.partial(self._encode_tree, ctx=ctx))

  def save_tree(self,
                tree: Any,
                path: Union[str, BinaryIO],
                options: Mapping[str, Any] = immutabledict.immutabledict({})):
    """Saves a tree to a path or a file object.

    Args:
      tree: A tree.
      path: Either path to a file or a file object.
      options: Options for serialization. See below for options.
    Options:
      tree_format: Format of the tree structure encoding. Must be 'json'.
      block_format: Format of the blocks encoding. Must be 'cat_npy'.
    """
    ctx = Context(self.save_version, None, options)

    if isinstance(path, str):
      f = epath.Path(path).open('wb')
      need_close = True
    else:
      f = path
      need_close = False

    try:
      tree_format = ctx.options['tree_format']

      f.write(f'{_TREE2_TAG},{ctx.version},{tree_format}\n'.encode('utf-8'))

      if tree_format == 'json':
        tree = self._encode_tree(tree, ctx)
        f.write(
            json.dumps(
                {
                    'tree': tree
                },
                indent=None,
                ensure_ascii=False,
                sort_keys=True,
            ).encode('utf-8'))
      else:
        raise ValueError(f'Unknown tree format: {tree_format}')

      f.write(b'\n')

      ctx.save_blocks(f)
    finally:
      if need_close:
        f.close()

  def load_tree(
      self,
      path: Union[str, BinaryIO],
      options: Mapping[str, Any] = immutabledict.immutabledict({})
  ) -> Any:
    """Loads a tree from a path or a file object.

    Args:
      path: Either path to a file or a file object.
      options: Options for serialization. See below for options.

    Returns:
      The loaded tree.

    Options:
      defer_numpy: Whether to defer loading numpy arrays. Numpy arrays will be
        replaced with instances of `DeferredNumpyArray`. Default: False
    """
    if isinstance(path, str):
      f = epath.Path(path).open('rb')
      need_close = True
      filename = path
    else:
      f = path
      need_close = False
      filename = None

    try:
      header = f.readline().decode('utf-8')
      header_parts = header.strip().split(',')

      if len(header_parts) != 3:
        raise ValueError('Bad header')
      if header_parts[0] != _TREE2_TAG:
        raise ValueError('Bad magic constant')
      version = int(header_parts[1])
      if version != _SAVE_VERSION:
        raise ValueError(f'Unknown version: {header_parts[1]}')
      tree_format = header_parts[2]

      ctx = Context(version, filename, options)

      block_header = None
      tree_lines = []
      while True:
        line = f.readline().decode('utf-8')
        if not line:
          break
        if line.startswith(_BLOCKS_TAG):
          block_header = line
          break
        else:
          tree_lines.append(line)

      if not tree_lines:
        raise ValueError('Empty tree?')

      if block_header is None:
        block_format = None
      else:
        block_parts = line.strip().split(',')
        if len(block_parts) != 2:
          raise ValueError('Bad block header')
        block_format = block_parts[1]

      if block_format is not None:
        ctx.load_blocks(f, block_format)

    finally:
      if need_close:
        f.close()

    if tree_format == 'json':
      tree = json.loads(
          '\n'.join(tree_lines),
          object_hook=functools.partial(self._json_obj_hook, ctx=ctx))
    else:
      raise ValueError(f'Unknown tree format: {tree_format}')

    return tree['tree']

  def auto_register_type(
      self, tag: Union[str,
                       Sequence[str]]) -> Callable[[Type[Tree]], Type[Tree]]:
    """Registers a type, with an automatic encoder/decoder.

    Only namedtuples, dataclasses, sequences, mappings and enums are supported.

    Args:
      tag: Tags to register the type under.

    Returns:
      Registration decorator.
    """

    def reg_fn(tree_type: Type[Tree]) -> Type[Tree]:
      if issubclass(tree_type, tuple) and hasattr(tree_type, '_fields'):
        return self.register_namedtuple_type(tag)(tree_type)
      elif issubclass(tree_type, Sequence):
        return self.register_sequence_type(tag)(tree_type)
      elif issubclass(tree_type, Mapping):
        return self.register_mapping_type(tag)(tree_type)
      elif dataclasses.is_dataclass(tree_type):
        return self.register_dataclass_type(tag)(tree_type)
      elif issubclass(tree_type, enum.Enum):
        return self.register_enum_type(tag)(tree_type)
      else:
        raise TypeError(
            f'Cannot register \'{tree_type}\' automatically. Use '
            '`register_type` with manual encode/decode functions.')

    return reg_fn

  def register_sequence_type(
      self,
      tag: Union[None, str, Sequence[str]] = None,
      fallback: Optional[str] = UNKNOWN_SEQUENCE
  ) -> Callable[[Type[Tree]], Type[Tree]]:
    """Registers a sequence type.

    Args:
      tag: Tags to register the type under.
      fallback: Fallback type to use for loading if this type is not registered
        at loading time, typically `UNKNOWN_SEQUENCE`. Can be `None` if you want
        that situation to raise an error.

    Returns:
      Registration decorator.
    """

    def reg_fn(tree_type: Type[Tree]) -> Type[Tree]:
      return self.register_type(
          tag, tree_type,
          functools.partial(
              _encode_sequence, tag=_get_first_tag(tag), fallback=fallback),
          functools.partial(_decode_sequence, tree_type=tree_type))

    return reg_fn

  def register_mapping_type(
      self,
      tag: Union[None, str, Sequence[str]] = None,
      fallback: Optional[str] = UNKNOWN_MAPPING
  ) -> Callable[[Type[Tree]], Type[Tree]]:
    """Registers a mapping type.

    Args:
      tag: Tags to register the type under.
      fallback: Fallback type to use for loading if this type is not registered
        at loading time, typically `UNKNOWN_MAPPING`. Can be `None` if you want
        that situation to raise an error.

    Returns:
      Registration decorator.
    """

    def reg_fn(tree_type: Type[Tree]) -> Type[Tree]:
      return self.register_type(
          tag, tree_type,
          functools.partial(
              _encode_mapping, tag=_get_first_tag(tag), fallback=fallback),
          functools.partial(_decode_mapping, tree_type=tree_type))

    return reg_fn

  def register_namedtuple_type(
      self,
      tag: Union[None, str, Sequence[str]] = None,
      fallback: Optional[str] = UNKNOWN_NAMEDTUPLE
  ) -> Callable[[Type[Tree]], Type[Tree]]:
    """Registers a namedtuple type.

    Args:
      tag: Tags to register the type under.
      fallback: Fallback type to use for loading if this type is not registered
        at loading time, typically `UNKNOWN_NAMEDTUPLE`. Can be `None` if you
        want that situation to raise an error.

    Returns:
      Registration decorator.
    """

    def reg_fn(tree_type: Type[Tree]) -> Type[Tree]:
      return self.register_type(
          tag, tree_type,
          functools.partial(
              _encode_namedtuple, tag=_get_first_tag(tag), fallback=fallback),
          functools.partial(_decode_namedtuple, tree_type=tree_type))

    return reg_fn

  def register_dataclass_type(
      self,
      tag: Union[None, str, Sequence[str]] = None,
      fallback: Optional[str] = UNKNOWN_DATACLASS
  ) -> Callable[[Type[Tree]], Type[Tree]]:
    """Registers a dataclass type.

    Args:
      tag: Tags to register the type under.
      fallback: Fallback type to use for loading if this type is not registered
        at loading time, typically `UNKNOWN_DATACLASS`. Can be `None` if you
        want that situation to raise an error.

    Returns:
      Registration decorator.
    """

    def reg_fn(tree_type: Type[Tree]) -> Type[Tree]:
      return self.register_type(
          tag, tree_type,
          functools.partial(
              _encode_dataclass, tag=_get_first_tag(tag), fallback=fallback),
          functools.partial(_decode_dataclass, tree_type=tree_type))

    return reg_fn

  def register_enum_type(
      self,
      tag: Union[None, str, Sequence[str]] = None
  ) -> Callable[[Type[Tree]], Type[Tree]]:
    """Registers an enum type.

    Args:
      tag: Tags to register the type under.

    Returns:
      Registration decorator.
    """

    def reg_fn(tree_type: Type[Tree]) -> Type[Tree]:
      return self.register_type(
          tag, tree_type,
          functools.partial(_encode_enum, tag=_get_first_tag(tag)),
          functools.partial(_decode_enum, tree_type=tree_type))

    return reg_fn

  def _maybe_register_flax(self):
    """Registers Flax types if Flax is importable."""
    try:
      # pytype: disable=import-error
      import flax  # pylint: disable=g-import-not-at-top
      # pytype: enable=import-error

      self.register_mapping_type('flax_frozen_dict')(
          flax.core.frozen_dict.FrozenDict)
    except ImportError:
      pass

  def _maybe_register_jax(self):
    """Registers JAX types if JAX is importable."""

    try:
      # pytype: disable=import-error
      import jax  # pylint: disable=g-import-not-at-top

      # pytype: enable=import-error

      def detect_jax_array(tree: Any, ctx: Context) -> Optional[str]:
        del ctx
        if isinstance(tree, jax.Array):
          return ARRAY
        else:
          return None

      self.register_detector('jax_array', detect_jax_array)
    except ImportError:
      pass

  def _maybe_register_tensorflow(self):
    """Registers TensorFlow types if TensorFlow is importable."""

    try:
      # pytype: disable=import-error
      import tensorflow as tf  # pylint: disable=g-import-not-at-top

      # pytype: enable=import-error

      def detect_tf_tensor(tree: Any, ctx: Context) -> Optional[str]:
        del ctx
        if isinstance(tree, tf.Tensor):
          return ARRAY
        else:
          return None

      self.register_detector('tensorflow_tensor', detect_tf_tensor)
    except ImportError:
      pass

  def _maybe_register_tfp(self):
    """Registers TFP types if TFP is importable."""

    structural_tuple = None
    try:
      # pytype: disable=import-error
      from tensorflow_probability.python.internal import structural_tuple  # pylint: disable=g-import-not-at-top
      # pytype: enable=import-error
    except ImportError:
      pass

    if structural_tuple is None:
      try:
        # pytype: disable=import-error
        import tensorflow_probability.substrates.jax as tfp  # pylint: disable=g-import-not-at-top
        structural_tuple = tfp.internal.structural_tuple
        # pytype: enable=import-error
      except ImportError:
        pass

    if structural_tuple is None:
      try:
        # pytype: disable=import-error
        import tensorflow_probability.substrates.numpy as tfp  # pylint: disable=g-import-not-at-top
        structural_tuple = tfp.internal.structural_tuple
        # pytype: enable=import-error
      except ImportError:
        pass

    if structural_tuple is not None:
      tfp_struct_tuple = 'tfp_struct_tuple'

      def detect_tfp_struct_tuple(tree: Any, ctx: Context) -> Optional[str]:
        del ctx
        if (hasattr(tree, '_tfp_nest_expansion_force_args') and
            type(tree).__name__ == 'StructTuple'):
          return tfp_struct_tuple
        else:
          return None

      def encode_tfp_struct_tuple(tree: Type[Tree], ctx: Context,
                                  encode_fn: InnerEncodeFn) -> EncodedTree:
        """Encodes a StructTuple type."""
        del ctx
        encoded = {}
        encoded[_TREE2_TYPE_TAG] = tfp_struct_tuple
        encoded['val'] = {k: encode_fn(v) for k, v in tree._asdict().items()}
        encoded['fallback_type'] = UNKNOWN_NAMEDTUPLE
        return encoded

      def decode_tfp_struct_tuple(encoded: Any, ctx: Context) -> Any:
        del ctx
        return structural_tuple.structtuple(
            encoded['val'].keys())(**encoded['val'])

      self.register_detector(tfp_struct_tuple, detect_tfp_struct_tuple)
      self.register_tags(tfp_struct_tuple, encode_tfp_struct_tuple,
                         decode_tfp_struct_tuple)

  def _register_numpy(self):
    """Registers np.ndarray and np.generic handling."""

    def encode_array_fn(tree: Tree, ctx: Context,
                        encode_fn: InnerEncodeFn) -> EncodedTree:
      del encode_fn
      tree = np.asarray(tree)

      encoded = {}
      encoded[_TREE2_TYPE_TAG] = ARRAY
      encoded['dtype'] = np.dtype(tree.dtype).name
      encoded['shape'] = list(tree.shape)
      if np.size(tree) < 64:
        encoded['val'] = tree.tolist()
      else:
        encoded['head'] = tree.flatten()[:10].tolist()
        encoded['tail'] = tree.flatten()[-10:].tolist()
        encoded['block'] = ctx.add_array(tree)

      return encoded

    def decode_array_fn(encoded: Any, ctx: Context) -> Any:
      val = encoded.get('val')
      if val is None:
        array = ctx.get_array(encoded['block'], encoded['shape'],
                              np.dtype(encoded['dtype']))
      else:
        array = np.array(val).astype(encoded['dtype'])
      return array

    self.register_type(ARRAY, np.ndarray, encode_array_fn, decode_array_fn)

    def encode_scalar_fn(tree: Tree, ctx: Context,
                         encode_fn: InnerEncodeFn) -> EncodedTree:
      del ctx, encode_fn
      tree = np.asarray(tree)

      encoded = {}
      encoded[_TREE2_TYPE_TAG] = SCALAR
      encoded['dtype'] = np.dtype(tree.dtype).name
      encoded['val'] = tree.tolist()

      return encoded

    def decode_scalar_fn(encoded: Any, ctx: Context) -> Any:
      del ctx
      return np.dtype(encoded['dtype']).type(encoded['val'])

    def detect_scalar(tree: Any, ctx: Context) -> Optional[str]:
      del ctx
      if isinstance(tree, np.generic):
        return SCALAR
      else:
        return None

    self.register_tags(SCALAR, encode_scalar_fn, decode_scalar_fn)
    self.register_detector(SCALAR, detect_scalar)


#
# Sequences
#


def _encode_sequence(tree: Type[Tree],
                     ctx: Context,
                     encode_fn: InnerEncodeFn,
                     tag: str,
                     fallback: Optional[str] = UNKNOWN_SEQUENCE) -> EncodedTree:
  """Encodes a sequence type."""
  del ctx
  encoded = {}
  encoded[_TREE2_TYPE_TAG] = tag
  if type(tree) is list:  # pylint: disable=unidiomatic-typecheck
    return [encode_fn(v) for v in tree]
  else:
    encoded['val'] = [encode_fn(v) for v in tree]
    if fallback is not None:
      encoded['fallback_type'] = fallback
  return encoded


def _decode_sequence(encoded: Any, ctx: Context, tree_type: Type[Tree]) -> Tree:
  del ctx
  return tree_type(encoded['val'])


def _detect_unknown_sequence(tree: Any, ctx: Context) -> Optional[str]:
  del ctx
  if isinstance(tree, Sequence):
    return UNKNOWN_SEQUENCE
  else:
    return None


def _encode_unknown_sequence(tree: Type[Tree], ctx: Context,
                             encode_fn: InnerEncodeFn) -> EncodedTree:
  warnings.warn(f'Encoding unknown sequence type: {type(tree).__name__}')
  return _encode_sequence(tree, ctx, encode_fn, type(tree).__name__)


def _decode_unknown_sequence(encoded: Any, ctx: Context) -> list[Any]:
  warnings.warn(f'Decoding unknown sequence type: {encoded[_TREE2_TYPE_TAG]}')
  return _decode_sequence(encoded, ctx, list)


#
# Mappings
#


def _encode_mapping(tree: Type[Tree],
                    ctx: Context,
                    encode_fn: InnerEncodeFn,
                    tag: str,
                    fallback: Optional[str] = UNKNOWN_MAPPING) -> EncodedTree:
  """Encodes a mapping type."""
  del ctx
  encoded = {}
  encoded[_TREE2_TYPE_TAG] = tag
  # Fast path: all-string keys and no special tags inside the mapping lets us
  # use a more efficient encoding.
  if all(isinstance(x, str) for x in tree) and _TREE2_TYPE_TAG not in tree:
    if type(tree) is dict:  # pylint: disable=unidiomatic-typecheck
      return {k: encode_fn(v) for k, v in tree.items()}
    else:
      tree = {k: encode_fn(v) for k, v in tree.items()}
  else:
    tree = [[encode_fn(k), encode_fn(v)] for k, v in tree.items()]
  encoded['val'] = tree
  if fallback is not None:
    encoded['fallback_type'] = fallback
  return encoded


def _decode_mapping(encoded: Any, ctx: Context, tree_type: Type[Tree]) -> Tree:
  del ctx
  return tree_type(encoded['val'])


def _detect_unknown_mapping(tree: Any, ctx: Context) -> Optional[str]:
  del ctx
  if isinstance(tree, Mapping):
    return UNKNOWN_MAPPING
  else:
    return None


def _encode_unknown_mapping(tree: Type[Tree], ctx: Context,
                            encode_fn: InnerEncodeFn) -> EncodedTree:
  warnings.warn(f'Encoding unknown mapping type: {type(tree).__name__}')
  return _encode_mapping(tree, ctx, encode_fn, type(tree).__name__)


def _decode_unknown_mapping(encoded: Any, ctx: Context) -> dict[Any, Any]:
  warnings.warn(f'Decoding unknown mapping type: {encoded[_TREE2_TYPE_TAG]}')
  return _decode_mapping(encoded, ctx, dict)


#
# NamedTuples
#


def _encode_namedtuple(
    tree: Type[Tree],
    ctx: Context,
    encode_fn: InnerEncodeFn,
    tag: str,
    fallback: Optional[str] = UNKNOWN_NAMEDTUPLE) -> EncodedTree:
  """Encodes a namedtuple."""
  del ctx
  encoded = {}
  encoded[_TREE2_TYPE_TAG] = tag
  encoded['val'] = {k: encode_fn(v) for k, v in tree._asdict().items()}
  if fallback is not None:
    encoded['fallback_type'] = fallback
  return encoded


def _decode_namedtuple(encoded: Any, ctx: Context,
                       tree_type: Type[Tree]) -> Tree:
  """Decodes a namedtuple."""
  del ctx
  fields = set(tree_type._fields)
  sanitized_val = {}
  for k, v in encoded['val'].items():
    if k in fields:
      sanitized_val[k] = v
    else:
      warnings.warn(f'Saw unknown field \'{k}\' while decoding '
                    f'\'{encoded[_TREE2_TYPE_TAG]}\'')
  return tree_type(**sanitized_val)


def _detect_unknown_namedtuple(tree: Any, ctx: Context) -> Optional[str]:
  del ctx
  if isinstance(tree, tuple) and hasattr(tree, '_fields'):
    return UNKNOWN_NAMEDTUPLE
  else:
    return None


def _encode_unknown_namedtuple(tree: Type[Tree], ctx: Context,
                               encode_fn: InnerEncodeFn) -> EncodedTree:
  warnings.warn(f'Encoding unknown namedtuple type: {type(tree).__name__}')
  return _encode_namedtuple(tree, ctx, encode_fn, type(tree).__name__)


def _decode_unknown_namedtuple(encoded: Any, ctx: Context) -> Any:
  del ctx
  warnings.warn(f'Decoding unknown namedtuple type: {encoded[_TREE2_TYPE_TAG]}')

  tree_type = collections.namedtuple(encoded[_TREE2_TYPE_TAG],
                                     list(encoded['val'].keys()))
  return tree_type(**encoded['val'])


#
# Dataclasses
#


def _encode_dataclass(
    tree: Type[Tree],
    ctx: Context,
    encode_fn: InnerEncodeFn,
    tag: str,
    fallback: Optional[str] = UNKNOWN_DATACLASS) -> EncodedTree:
  """Encodes a dataclass."""
  del ctx
  encoded = {}
  encoded[_TREE2_TYPE_TAG] = tag
  encoded['val'] = {
      f.name: encode_fn(getattr(tree, f.name)) for f in dataclasses.fields(tree)
  }
  if fallback is not None:
    encoded['fallback_type'] = fallback
  return encoded


def _decode_dataclass(encoded: Any, ctx: Context,
                      tree_type: Type[Tree]) -> Tree:
  """Decodes a dataclass."""
  del ctx
  fields = set(f.name for f in dataclasses.fields(tree_type))
  sanitized_val = {}
  for k, v in encoded['val'].items():
    if k in fields:
      sanitized_val[k] = v
    else:
      warnings.warn(f'Saw unknown field \'{k}\' while decoding '
                    f'\'{encoded[_TREE2_TYPE_TAG]}\'')
  return tree_type(**sanitized_val)


def _detect_unknown_dataclass(tree: Any, ctx: Context) -> Optional[str]:
  del ctx
  if dataclasses.is_dataclass(tree):
    return UNKNOWN_DATACLASS
  else:
    return None


def _encode_unknown_dataclass(tree: Type[Tree], ctx: Context,
                              encode_fn: InnerEncodeFn) -> EncodedTree:
  warnings.warn(f'Encoding unknown dataclass type: {type(tree).__name__}')
  return _encode_dataclass(tree, ctx, encode_fn, type(tree).__name__)


def _decode_unknown_dataclass(encoded: Any, ctx: Context) -> Any:
  del ctx
  warnings.warn(f'Decoding unknown dataclass type: {encoded[_TREE2_TYPE_TAG]}')

  tree_type = dataclasses.make_dataclass(encoded[_TREE2_TYPE_TAG],
                                         list(encoded['val'].keys()))
  return tree_type(**encoded['val'])


#
# Enums
#


def _encode_enum(tree: Type[Tree], ctx: Context, encode_fn: InnerEncodeFn,
                 tag: str) -> EncodedTree:
  """Encodes an enum type."""
  del ctx, encode_fn
  encoded = {}
  encoded[_TREE2_TYPE_TAG] = tag
  encoded['val'] = tree.name
  return encoded


def _decode_enum(encoded: Any, ctx: Context, tree_type: Type[Tree]) -> Tree:
  del ctx
  return tree_type[encoded['val']]
