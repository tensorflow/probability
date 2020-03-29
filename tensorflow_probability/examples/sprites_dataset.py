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
"""Liberated Pixel Cup [(LPC)][1] Sprites Dataset.

This file provides logic to download and build a version of the sprites
video sequence dataset as used in the Disentangled Sequential
Autoencoder paper [(Li and Mandt, 2018)][2].

#### References:

[1]: Liberated Pixel Cup. http://lpc.opengameart.org. Accessed:
     2018-07-20.
[2]: Yingzhen Li and Stephan Mandt. Disentangled Sequential Autoencoder.
     In _International Conference on Machine Learning_, 2018.
     https://arxiv.org/abs/1803.02991
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import os
import random
import zipfile

from absl import flags
from six.moves import urllib
import tensorflow.compat.v1 as tf
from tensorflow.python.ops import lookup_ops  # pylint: disable=g-direct-tensorflow-import


__all__ = ["SpritesDataset"]


flags.DEFINE_string(
    "data_dir",
    default=os.path.join(
        os.getenv("TEST_TMPDIR", "/tmp"),
        os.path.join("disentangled_vae", "data")),
    help="Directory where the dataset is stored.")

DATA_SPRITES_URL = "https://github.com/jrconway3/Universal-LPC-spritesheet/archive/master.zip"
DATA_SPRITES_DIR = "Universal-LPC-spritesheet-master"
WIDTH = 832
HEIGHT = 1344
FRAME_SIZE = 64
CHANNELS = 4

SKIN_COLORS = [
    os.path.join("body", "male", "light.png"),
    os.path.join("body", "male", "tanned2.png"),
    os.path.join("body", "male", "darkelf.png"),
    os.path.join("body", "male", "darkelf2.png"),
    os.path.join("body", "male", "dark.png"),
    os.path.join("body", "male", "dark2.png")
]
HAIRSTYLES = [
    os.path.join("hair", "male", "messy2", "green2.png"),
    os.path.join("hair", "male", "ponytail", "blue2.png"),
    os.path.join("hair", "male", "messy1", "light-blonde.png"),
    os.path.join("hair", "male", "parted", "white.png"),
    os.path.join("hair", "male", "plain", "ruby-red.png"),
    os.path.join("hair", "male", "jewfro", "purple.png")
]
TOPS = [
    os.path.join(
        "torso", "shirts", "longsleeve", "male", "maroon_longsleeve.png"),
    os.path.join(
        "torso", "shirts", "longsleeve", "male", "teal_longsleeve.png"),
    os.path.join(
        "torso", "shirts", "longsleeve", "male", "white_longsleeve.png"),
    os.path.join("torso", "plate", "chest_male.png"),
    os.path.join("torso", "leather", "chest_male.png"),
    os.path.join("formal_male_no_th-sh", "shirt.png")
]
PANTS = [
    os.path.join("legs", "pants", "male", "white_pants_male.png"),
    os.path.join("legs", "armor", "male", "golden_greaves_male.png"),
    os.path.join("legs", "pants", "male", "red_pants_male.png"),
    os.path.join("legs", "armor", "male", "metal_pants_male.png"),
    os.path.join("legs", "pants", "male", "teal_pants_male.png"),
    os.path.join("formal_male_no_th-sh", "pants.png")
]

Action = namedtuple("Action", ["name", "start_row", "frames"])
ACTIONS = [
    Action("walk", 8, 9),
    Action("spellcast", 0, 7),
    Action("slash", 12, 6)
]

Direction = namedtuple("Direction", ["name", "row_offset"])
DIRECTIONS = [
    Direction("west", 1),
    Direction("south", 2),
    Direction("east", 3),
]

FLAGS = flags.FLAGS


def read_image(filepath):
  """Returns an image tensor."""
  im_bytes = tf.io.read_file(filepath)
  im = tf.image.decode_image(im_bytes, channels=CHANNELS)
  im = tf.image.convert_image_dtype(im, tf.float32)
  return im


def join_seq(seq):
  """Joins a sequence side-by-side into a single image."""
  return tf.concat(tf.unstack(seq), 1)


def download_sprites():
  """Downloads the sprites data and returns the saved filepath."""
  filepath = os.path.join(FLAGS.data_dir, DATA_SPRITES_DIR)
  if not tf.io.gfile.exists(filepath):
    if not tf.io.gfile.exists(FLAGS.data_dir):
      tf.io.gfile.makedirs(FLAGS.data_dir)
    zip_name = "{}.zip".format(filepath)
    urllib.request.urlretrieve(DATA_SPRITES_URL, zip_name)
    with zipfile.ZipFile(zip_name, "r") as zip_file:
      zip_file.extractall(FLAGS.data_dir)
    tf.io.gfile.remove(zip_name)
  return filepath


def create_character(skin, hair, top, pants):
  """Creates a character sprite from a set of attribute sprites."""
  dtype = skin.dtype
  hair_mask = tf.cast(hair[..., -1:] <= 0, dtype)
  top_mask = tf.cast(top[..., -1:] <= 0, dtype)
  pants_mask = tf.cast(pants[..., -1:] <= 0, dtype)
  char = (skin * hair_mask) + hair
  char = (char * top_mask) + top
  char = (char * pants_mask) + pants
  return char


def create_seq(character, action_metadata, direction, length=8, start=0):
  """Creates a sequence.

  Args:
    character: A character sprite tensor.
    action_metadata: An action metadata tuple.
    direction: An integer representing the direction, i.e., the row
      offset within each action group corresponding to a particular
      direction.
    length: Desired length of the sequence. If this is longer than
      the number of available frames, it will roll over to the
      beginning.
    start: Index of possible frames at which to start the sequence.

  Returns:
    A sequence tensor.
  """
  sprite_start = (action_metadata[0]+direction) * FRAME_SIZE
  sprite_end = (action_metadata[0]+direction+1) * FRAME_SIZE
  sprite_line = character[sprite_start:sprite_end, ...]

  # Extract 64x64 patches that are side-by-side in the sprite, and limit
  # to the actual number of frames for the given action.
  frames = tf.stack(tf.split(sprite_line, 13, axis=1))  # 13 is a hack
  frames = frames[0:action_metadata[1]]

  # Extract a slice of the desired length.
  # NOTE: Length could be longer than the number of frames, so tile as needed.
  frames = tf.roll(frames, shift=-start, axis=0)
  frames = tf.tile(frames, [2, 1, 1, 1])  # 2 is a hack
  frames = frames[:length]
  frames = tf.cast(frames, dtype=tf.float32)
  frames.set_shape([length, FRAME_SIZE, FRAME_SIZE, CHANNELS])
  return frames


def create_random_seq(character, action_metadata, direction, length=8):
  """Creates a random sequence."""
  start = tf.random.uniform([], maxval=action_metadata[1], dtype=tf.int32)
  return create_seq(character, action_metadata, direction, length, start)


def create_sprites_dataset(characters, actions, directions, channels=3,
                           length=8, shuffle=False, fake_data=False):
  """Creates a tf.data pipeline for the sprites dataset.

  Args:
    characters: A list of (skin, hair, top, pants) tuples containing
      relative paths to the sprite png image for each attribute.
    actions: A list of Actions.
    directions: A list of Directions.
    channels: Number of image channels to yield.
    length: Desired length of the sequences.
    shuffle: Whether or not to shuffle the characters and sequences
      start frame.
    fake_data: Boolean for whether or not to yield synthetic data.

  Returns:
    A tf.data.Dataset yielding (seq, skin label index, hair label index,
    top label index, pants label index, action label index, skin label
    name, hair label_name, top label name, pants label name, action
    label name) tuples.
  """
  if fake_data:
    dummy_image = tf.random.normal([HEIGHT, WIDTH, CHANNELS])
  else:
    basedir = download_sprites()

  action_names = [action.name for action in actions]
  action_metadata = [(action.start_row, action.frames) for action in actions]

  direction_rows = [direction.row_offset for direction in directions]

  chars = tf.data.Dataset.from_tensor_slices(characters)
  act_names = tf.data.Dataset.from_tensor_slices(action_names).repeat()
  acts_metadata = tf.data.Dataset.from_tensor_slices(action_metadata).repeat()
  dir_rows = tf.data.Dataset.from_tensor_slices(direction_rows).repeat()

  if shuffle:
    chars = chars.shuffle(len(characters))

  dataset = tf.data.Dataset.zip((chars, act_names, acts_metadata, dir_rows))

  skin_table = lookup_ops.index_table_from_tensor(sorted(SKIN_COLORS))
  hair_table = lookup_ops.index_table_from_tensor(sorted(HAIRSTYLES))
  top_table = lookup_ops.index_table_from_tensor(sorted(TOPS))
  pants_table = lookup_ops.index_table_from_tensor(sorted(PANTS))
  action_table = lookup_ops.index_table_from_tensor(sorted(action_names))

  def process_example(attrs, act_name, act_metadata, dir_row_offset):
    """Processes a dataset row."""
    skin_name = attrs[0]
    hair_name = attrs[1]
    top_name = attrs[2]
    pants_name = attrs[3]

    if fake_data:
      char = dummy_image
    else:
      skin = read_image(basedir + os.sep + skin_name)
      hair = read_image(basedir + os.sep + hair_name)
      top = read_image(basedir + os.sep + top_name)
      pants = read_image(basedir + os.sep + pants_name)
      char = create_character(skin, hair, top, pants)

    if shuffle:
      seq = create_random_seq(char, act_metadata, dir_row_offset, length)
    else:
      seq = create_seq(char, act_metadata, dir_row_offset, length)
    seq = seq[..., :channels]  # limit output channels

    skin_idx = skin_table.lookup(skin_name)
    hair_idx = hair_table.lookup(hair_name)
    top_idx = top_table.lookup(top_name)
    pants_idx = pants_table.lookup(pants_name)
    act_idx = action_table.lookup(act_name)

    return (seq, skin_idx, hair_idx, top_idx, pants_idx, act_idx,
            skin_name, hair_name, top_name, pants_name, act_name)

  dataset = dataset.map(process_example)
  return dataset


class SpritesDataset(object):
  """Liberated Pixel Cup [(LPC)][1] Sprites Dataset.

  This file provides logic to download and build a version of the
  sprites video sequence dataset as used in the Disentangled Sequential
  Autoencoder paper [(Li and Mandt, 2018)][2]. The dataset contains
  sprites (graphics files used to generate animated sequences) of human
  characters wearing a variety of clothing, and performing a variety of
  actions. The paper limits the dataset used for training to four
  attribute categories (skin color, hairstyles, tops, and pants), each
  of which are limited to include six variants. Thus, there are
  6^4 = 1296 possible animated characters in this dataset. The
  characters are shuffled and deterministically split such that 1000
  characters are used for the training set, and 296 are used for the
  testing set. The numbers are consistent with the paper, but the exact
  split is impossible to match given the currently available paper
  details. The actions are limited to three categories (walking,
  casting spells, and slashing), each with three viewing angles.
  Sequences of length T=8 frames are generated depicting a given
  character performing a given action, starting at a random frame in the
  sequence.

  Attributes:
    train: Training dataset with 1000 characters each performing an
      action.
    test: Testing dataset with 296 characters each performing an action.

  #### References:

  [1]: Liberated Pixel Cup. http://lpc.opengameart.org. Accessed:
       2018-07-20.
  [2]: Yingzhen Li and Stephan Mandt. Disentangled Sequential
       Autoencoder. In _International Conference on Machine Learning_,
       2018. https://arxiv.org/abs/1803.02991
  """

  def __init__(self, channels=3, shuffle_train=True, fake_data=False):
    """Creates the SpritesDataset and stores train and test datasets.

    The datasets yield (seq, skin label index, hair label index, top
    label index, pants label index, action label index, skin label name,
    hair label_name, top label name, pants label name, action label
    name) tuples.

    Args:
      channels: Number of image channels to yield.
      shuffle_train: Boolean for whether or not to shuffle the training
        set.
      fake_data: Boolean for whether or not to yield synthetic data.

    Raises:
      ValueError: If the number of training or testing examples is
        incorrect, or if there is overlap betweem the two datasets.
    """
    super(SpritesDataset, self).__init__()
    self.frame_size = FRAME_SIZE
    self.channels = channels
    self.length = 8
    num_train = 1000
    num_test = 296
    characters = [(skin, hair, top, pants)
                  for skin in sorted(SKIN_COLORS)
                  for hair in sorted(HAIRSTYLES)
                  for top in sorted(TOPS)
                  for pants in sorted(PANTS)]
    random.seed(42)
    random.shuffle(characters)
    train_chars = characters[:num_train]
    test_chars = characters[num_train:]
    num_train_actual = len(set(train_chars))
    num_test_actual = len(set(test_chars))
    num_train_test_overlap = len(set(train_chars) & set(test_chars))
    if num_train_actual != num_train:
      raise ValueError(
          "Unexpected number of training examples: {}.".format(
              num_train_actual))
    if num_test_actual != num_test:
      raise ValueError(
          "Unexpected number of testing examples: {}.".format(
              num_test_actual))
    if num_train_test_overlap > 0:  # pylint: disable=g-explicit-length-test
      raise ValueError(
          "Overlap between train and test datasets detected: {}.".format(
              num_train_test_overlap))

    self.train = create_sprites_dataset(
        train_chars, ACTIONS, DIRECTIONS, self.channels, self.length,
        shuffle=shuffle_train, fake_data=fake_data)
    self.test = create_sprites_dataset(
        test_chars, ACTIONS, DIRECTIONS, self.channels, self.length,
        shuffle=False, fake_data=fake_data)
