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
"""Glow bijector."""

import functools
import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.bijectors import blockwise
from tensorflow_probability.python.bijectors import chain
from tensorflow_probability.python.bijectors import composition
from tensorflow_probability.python.bijectors import exp
from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.bijectors import real_nvp
from tensorflow_probability.python.bijectors import reshape
from tensorflow_probability.python.bijectors import scale
from tensorflow_probability.python.bijectors import scale_matvec_lu
from tensorflow_probability.python.bijectors import shift
from tensorflow_probability.python.bijectors import transpose
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.util.deferred_tensor import TransformedVariable
from tensorflow_probability.python.util.seed_stream import SeedStream

tfk = tf.keras
tfkl = tfk.layers

__all__ = [
    'Glow',
    'GlowDefaultNetwork',
    'GlowDefaultExitNetwork',
]


class Glow(composition.Composition):
  r"""Implements the Glow Bijector from Kingma & Dhariwal (2018)[1].

  Overview: `Glow` is a chain of bijectors which transforms a rank-1 tensor
  (vector) into a rank-3 tensor (e.g. an RGB image). `Glow` does this by
  chaining together an alternating series of "Blocks," "Squeezes," and "Exits"
  which are each themselves special chains of other bijectors. The intended use
  of `Glow` is as part of a `tfp.distributions.TransformedDistribution`, in
  which the base distribution over the vector space is used to generate samples
  in the image space. In the paper, an Independent Normal distribution is used
  as the base distribution.

  A "Block" (implemented as the `GlowBlock` Bijector) performs much of the
  transformations which allow glow to produce sophisticated and complex mappings
  between the image space and the latent space and therefore achieve rich image
  generation performance. A Block is composed of `num_steps_per_block` steps,
  which are each implemented as a `Chain` containing an
  `ActivationNormalization` (ActNorm) bijector, followed by an (invertible)
  `OneByOneConv` bijector, and finally a coupling bijector. The coupling
  bijector is an instance of a `RealNVP` bijector, and uses the
  `coupling_bijector_fn` function to instantiate the coupling bijector function
  which is given to the `RealNVP`. This function returns a bijector which
  defines the coupling (e.g. `Shift(Scale)` for affine coupling or `Shift` for
  additive coupling).

  A "Squeeze" converts spatial features into channel features. It is
  implemented using the `Expand` bijector. The difference in names is
  due to the fact that the `forward` function from glow is meant to ultimately
  correspond to sampling from a `tfp.util.TransformedDistribution` object,
  which would use `Expand` (Squeeze is just Invert(Expand)). The `Expand`
  bijector takes a tensor with shape `[H, W, C]` and returns a tensor with shape
  `[2H, 2W, C / 4]`, such that each 2x2x1 spatial tile in the output is composed
  from a single 1x1x4 tile in the input tensor, as depicted in the figure below.

                           Forward pass (Expand)
                          ______        __________
                          \     \       \    \    \
                          \\     \ ----> \  1 \  2 \
                          \\\__1__\       \____\____\
                          \\\__2__\        \    \    \
                           \\__3__\  <----  \  3 \  4 \
                            \__4__\          \____\____\
                               Inverse pass (Squeeze)

  This is implemented using a chain of `Reshape` -> `Transpose` -> `Reshape`
  bijectors. Note that on an inverse pass through the bijector, each Squeeze
  will cause the width/height of the image to decrease by a factor of 2.
  Therefore, the input image must be evenly divisible by 2 at least
  `num_glow_blocks` times, since it will pass through a Squeeze step that many
  times.

  An "Exit" is simply a junction at which some of the tensor "exits" from the
  glow bijector and therefore avoids any further alteration. Each exit is
  implemented as a `Blockwise` bijector, where some channels are given to the
  rest of the glow model, and the rest are given to a bypass implemented using
  the `Identity` bijector. The fraction of channels to be removed at each exit
  is determined by the `grab_after_block` arg, indicates the fraction of
  _remaining_ channels which join the identity bypass. The fraction is
  converted to an integer number of channels by multiplying by the remaining
  number of channels and rounding.

  Additionally, at each exit, glow couples the tensor exiting the highway to
  the tensor continuing onward. This makes small scale features in the image
  dependent on larger scale features, since the larger scale features dictate
  the mean and scale of the distribution over the smaller scale features.
  This coupling is done similarly to the Coupling bijector in each step of the
  flow (i.e. using a RealNVP bijector). However for the exit bijector, the
  coupling is instantiated using `exit_bijector_fn` rather than coupling
  bijector fn, allowing for different behaviors between standard coupling and
  exit coupling. Also note that because the exit utilizes a coupling bijector,
  there are two special cases (all channels exiting and no channels exiting).

  The full Glow bijector consists of `num_glow_blocks` Blocks each of which
  contains `num_steps_per_block` steps. Each step implements a coupling using
  `bijector_coupling_fn`. Between blocks, glow converts between spatial pixels
  and channels using the Expand Bijector, and splits channels out of the
  bijector using the Exit Bijector. The channels which have exited continue
  onward through Identity bijectors and those which have not exited are given
  to the next block. After passing through all Blocks, the tensor is reshaped
  to a rank-1 tensor with the same number of elements. This is where the
  distribution will be defined.

  A schematic diagram of Glow is shown below. The `forward` function of the
  bijector starts from the bottom and goes upward, while the `inverse` function
  starts from the top and proceeds downward.

  ```None
  ==============================================================================
                           Glow Schematic Diagram

  Input Image     ########################   shape = [H, W, C]

                  \                      /<- Expand Bijector turns spatial
                   \                    /    dimensions into channels.
                  _
                 |  XXXXXXXXXXXXXXXXXXXX
                 |  XXXXXXXXXXXXXXXXXXXX
                 |  XXXXXXXXXXXXXXXXXXXX     A single step of the flow consists
   Glow Block  - |  XXXXXXXXXXXXXXXXXXXX  <- of ActNorm -> 1x1Conv -> Coupling.
                 |  XXXXXXXXXXXXXXXXXXXX     there are num_steps_per_block
                 |  XXXXXXXXXXXXXXXXXXXX     steps of the flow in each block.
                 |_ XXXXXXXXXXXXXXXXXXXX

                    \                  / <-- Expand bijectors follow each glow
                     \                /      block

                      XXXXXXXX\\\\\\\\   <-- Exit Bijector removes channels
                  _                    _     from additional alteration.
                 |    XXXXXXXX !  |  !
                 |    XXXXXXXX !  |  !
                 |    XXXXXXXX !  |  !       After exiting, channels are passed
   Glow Block  - |    XXXXXXXX !  |  !  <--- downward using the Blockwise and
                 |    XXXXXXXX !  |  !       Identify bijectors.
                 |    XXXXXXXX !  |  !
                 |_   XXXXXXXX !  |  !

                      \              / <---- Expand Bijector
                       \            /

                        XXX\\\    | !  <---- Exit Bijector
                  _
                 |      XXX ! |   | !
                 |      XXX ! |   | !
                 |      XXX ! |   | !
   Glow Block  - |      XXX ! |   | !
                 |      XXX ! |   | !
                 |      XXX ! |   | !
                 |_     XXX ! |   | !

                        XX\ ! |   | ! <----- (Optional) Exit Bijector

                         |    |   |
                         v    v   v
  Output Distribution    ##########          shape = [H * W * C]
                                                     _________________________
                                                    |         Legend          |
                                                    | XX  = Step of flow      |
                                                    | X\  = Exit bijector     |
                                                    | \/  = Expand bijector   |
                                                    | !|! = Identity bijector |
                                                    |                         |
                                                    | up  = Forward pass      |
                                                    | dn  = Inverse pass      |
                                                    |_________________________|

  ==============================================================================
  ```
  The default configuration for glow is meant to replicate the architecture in
  [1] for generating images from CIFAR-10.

  Example usage:
  ```python

  from functools import reduce
  from operator import mul
  import tensorflow as tf
  import tensorflow_datasets as tfds
  import tensorflow_probability as tfp
  tfb = tfp.bijectors
  tfd = tfp.distributions

  data, info = tfds.load('cifar10', with_info=True)
  train_data, test_data = data['train'], data['test']

  preprocess = lambda x: tf.cast(x['image'], tf.float32)
  train_data = train_data.batch(4).map(preprocess)
  test_data = test_data.batch(4).map(preprocess)

  x = next(iter(train_data))

  glow = tfb.Glow(output_shape=info.features['image'].shape,
                  coupling_bijector_fn=tfb.GlowDefaultNetwork,
                  exit_bijector_fn=tfb.GlowDefaultExitNetwork)

  z_shape = glow.inverse_event_shape(info.features['image'].shape)

  pz = tfd.Sample(tfd.Normal(0., 1.), z_shape)

  # Calling glow on distribution p(z) creates our glow distribution over images.
  px = glow(pz)

  # Take samples from the distribution to get images from your dataset
  images = px.sample(4)

  # Map images to positions in the distribution
  z = glow.inverse(x)

  # Get the z's corresponding to each spatial scale. To do this, we have to
  # find out how many zs are passed through blockwise at each stage that were
  # not passed at the previous stage. This is encoded in the second element of
  # each list of blockwise splits. However because the bijector iteratively
  # converts spatial pixels to channels, we also need to multiply the size of
  # that second element by the number of spatial-to-channel conversions that the
  # tensor receives after exiting (including after any alteration).
  ztake = [bs[1] * 4**(i+2) for i, bs in enumerate(glow.blockwise_splits)]
  total_z_taken = sum(ztake)
  split_sizes = [z_shape.as_list()[0]-total_z_taken] + ztake
  zsplits = tf.split(z, num_or_size_splits=split_sizes, axis=-1)
  ```

  #### References:

  [1]: Diederik P Kingma, Prafulla Dhariwal, Glow: Generative Flow
       with Invertible 1x1 Convolutions. In _Neural Information
       Processing Systems_, 2018. https://arxiv.org/abs/1807.03039

  [2]: Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density Estimation
       using Real NVP. In _International Conference on Learning
       Representations_, 2017. https://arxiv.org/abs/1605.08803
  """

  def __init__(self,
               output_shape=(32, 32, 3),
               num_glow_blocks=3,
               num_steps_per_block=32,
               coupling_bijector_fn=None,
               exit_bijector_fn=None,
               grab_after_block=None,
               use_actnorm=True,
               seed=None,
               validate_args=False,
               name='glow'):
    """Creates the Glow bijector.

    Args:
      output_shape: A list of integers, specifying the event shape of the
        output, of the bijectors forward pass (the image).  Specified as
        [H, W, C].
        Default Value: (32, 32, 3)
      num_glow_blocks: An integer, specifying how many downsampling levels to
        include in the model. This must divide equally into both H and W,
        otherwise the bijector would not be invertible.
        Default Value: 3
      num_steps_per_block: An integer specifying how many Affine Coupling and
        1x1 convolution layers to include at each level of the spatial
        hierarchy.
        Default Value: 32 (i.e. the value used in the original glow paper).
      coupling_bijector_fn: A function which takes the argument `input_shape`
        and returns a callable neural network (e.g. a keras.Sequential). The
        network should either return a tensor with the same event shape as
        `input_shape` (this will employ additive coupling), a tensor with the
        same height and width as `input_shape` but twice the number of channels
        (this will employ affine coupling), or a bijector which takes in a
        tensor with event shape `input_shape`, and returns a tensor with shape
        `input_shape`.
      exit_bijector_fn: Similar to coupling_bijector_fn, exit_bijector_fn is
        a function which takes the argument `input_shape` and `output_chan`
        and returns a callable neural network. The neural network it returns
        should take a tensor of shape `input_shape` as the input, and return
        one of three options: A tensor with `output_chan` channels, a tensor
        with `2 * output_chan` channels, or a bijector. Additional details can
        be found in the documentation for ExitBijector.
      grab_after_block: A tuple of floats, specifying what fraction of the
        remaining channels to remove following each glow block. Glow will take
        the integer floor of this number multiplied by the remaining number of
        channels. The default is half at each spatial hierarchy.
        Default value: None (this will take out half of the channels after each
          block.
      use_actnorm: A bool deciding whether or not to use actnorm. Data-dependent
        initialization is used to initialize this layer.
        Default value: `False`
      seed: A seed to control randomness in the 1x1 convolution initialization.
        Default value: `None` (i.e., non-reproducible sampling).
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
        Default value: `False`
      name: Python `str`, name given to ops managed by this object.
        Default value: `'glow'`.
    """
    parameters = dict(locals())
    # Make sure that the input shape is fully defined.
    if not tensorshape_util.is_fully_defined(output_shape):
      raise ValueError('Shape must be fully defined.')
    if tensorshape_util.rank(output_shape) != 3:
      raise ValueError('Shape ndims must be 3 for images.  Your shape is'
                       '{}'.format(tensorshape_util.rank(output_shape)))

    num_glow_blocks_ = tf.get_static_value(num_glow_blocks)
    if (num_glow_blocks_ is None or
        int(num_glow_blocks_) != num_glow_blocks_ or
        num_glow_blocks_ < 1):
      raise ValueError('Argument `num_glow_blocks` must be a statically known'
                       'positive `int` (saw: {}).'.format(num_glow_blocks))
    num_glow_blocks = int(num_glow_blocks_)

    output_shape = tensorshape_util.as_list(output_shape)
    h, w, c = output_shape
    n = num_glow_blocks
    nsteps = num_steps_per_block

    # Default Glow: Half of the channels are split off after each block,
    # and after the final block, no channels are split off.
    if grab_after_block is None:
      grab_after_block = tuple([0.5] * (n - 1) + [0.])

    # Thing we know must be true: h and w are evenly divisible by 2, n times.
    # Otherwise, the squeeze bijector will not work.
    if w % 2**n != 0:
      raise ValueError('Width must be divisible by 2 at least n times.'
                       'Saw: {} % {} != 0'.format(w, 2**n))
    if h % 2**n != 0:
      raise ValueError('Height should be divisible by 2 at least n times.')
    if h // 2**n < 1:
      raise ValueError('num_glow_blocks ({0}) is too large. The image height '
                       '({1}) must be divisible by 2 no more than {2} '
                       'times.'.format(num_glow_blocks, h,
                                       int(np.log(h) / np.log(2.))))
    if w // 2**n < 1:
      raise ValueError('num_glow_blocks ({0}) is too large. The image width '
                       '({1}) must be divisible by 2 no more than {2} '
                       'times.'.format(num_glow_blocks, w,
                                       int(np.log(h) / np.log(2.))))

    # Other things we want to be true:
    # - The number of times we take must be equal to the number of glow blocks.
    if len(grab_after_block) != num_glow_blocks:
      raise ValueError('Length of grab_after_block ({0}) must match the number'
                       'of blocks ({1}).'.format(len(grab_after_block),
                                                 num_glow_blocks))

    self._blockwise_splits = self._get_blockwise_splits(output_shape,
                                                        grab_after_block[::-1])

    # Now check on the values of blockwise splits
    if any([bs[0] < 1 for bs in self._blockwise_splits]):
      first_offender = [bs[0] for bs in self._blockwise_splits].index(True)
      raise ValueError('At at least one exit, you are taking out all of your '
                       'channels, and therefore have no inputs to later blocks.'
                       ' Try setting grab_after_block to a lower value at index'
                       '{}.'.format(first_offender))

    if any(np.isclose(gab, 0) for gab in grab_after_block):
      # Special case: if specifically exiting no channels, then the exit is
      # just an identity bijector.
      pass
    elif any([bs[1] < 1 for bs in self._blockwise_splits]):
      first_offender = [bs[1] for bs in self._blockwise_splits].index(True)
      raise ValueError('At least one of your layers has < 1 output channels. '
                       'This means you set grab_at_block too small. '
                       'Try setting grab_after_block to a larger value at index'
                       '{}.'.format(first_offender))

    # Lets start to build our bijector. We assume that the distribution is 1
    # dimensional. First, lets reshape it to an image.
    glow_chain = [
        reshape.Reshape(
            event_shape_out=[h // 2**n, w // 2**n, c * 4**n],
            event_shape_in=[h * w * c])
    ]

    seedstream = SeedStream(seed=seed, salt='random_beta')

    for i in range(n):

      # This is the shape of the current tensor
      current_shape = (h // 2**n * 2**i, w // 2**n * 2**i, c * 4**(i + 1))

      # This is the shape of the input to both the glow block and exit bijector.
      this_nchan = sum(self._blockwise_splits[i][0:2])
      this_input_shape = (h // 2**n * 2**i, w // 2**n * 2**i, this_nchan)

      glow_chain.append(invert.Invert(ExitBijector(current_shape,
                                                   self._blockwise_splits[i],
                                                   exit_bijector_fn)))

      glow_block = GlowBlock(input_shape=this_input_shape,
                             num_steps=nsteps,
                             coupling_bijector_fn=coupling_bijector_fn,
                             use_actnorm=use_actnorm,
                             seedstream=seedstream)

      if self._blockwise_splits[i][2] == 0:
        # All channels are passed to the RealNVP
        glow_chain.append(glow_block)
      else:
        # Some channels are passed around the block.
        # This is done with the Blockwise bijector.
        glow_chain.append(
            blockwise.Blockwise(
                [glow_block, identity.Identity()],
                [sum(self._blockwise_splits[i][0:2]),
                 self._blockwise_splits[i][2]]))

      # Finally, lets expand the channels into spatial features.
      glow_chain.append(
          Expand(input_shape=[
              h // 2**n * 2**i,
              w // 2**n * 2**i,
              c * 4**n // 4**i,
          ]))

    glow_chain = glow_chain[::-1]
    # To finish off, we build a bijector that chains the components together
    # sequentially.
    super(Glow, self).__init__(
        bijectors=chain.Chain(glow_chain, validate_args=validate_args),
        validate_args=validate_args,
        parameters=parameters,
        name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict()

  def _get_blockwise_splits(self, input_shape, grab_after_block):
    """build list of splits to give to the blockwise_bijectors.

    The list will have 3 different splits. The first element is `nleave`
    which shows how many channels will remain in the network after each exit.
    The second element is `ngrab`, which shows how many channels will be removed
    at the exit. The third is `npass`, which shows how many channels have
    already exited at a previous junction, and are therefore passed to an
    identity bijector instead of the glow block.

    Args:
      input_shape: shape of the input data
      grab_after_block: list of floats specifying what fraction of the channels
        should exit the network after each glow block.
    Returns:
      blockwise_splits: the number of channels left, taken, and passed over for
        each glow block.
    """
    blockwise_splits = []

    ngrab, nleave, npass = 0, 0, 0

    # Build backwards
    for i, frac in enumerate(reversed(grab_after_block)):
      nchan = 4**(i + 1) * input_shape[-1]
      ngrab = int((nchan - npass) * frac)
      nleave = nchan - ngrab - npass

      blockwise_splits.append([nleave, ngrab, npass])

      # update npass for the next level
      npass += ngrab
      npass *= 4

    return blockwise_splits[::-1]

  @property
  def blockwise_splits(self):
    return self._blockwise_splits


class ExitBijector(composition.Composition):
  """The spatial coupling bijector used in Glow.

  This bijector consists of a blockwise bijector of a realNVP bijector. It is
  where Glow adds a fork between points that are split off and passed to the
  base distribution, and points that are passed onward through more Glow blocks.

  For this bijector, we include spatial coupling between the part being forked
  off, and the part being passed onward. This induces a hierarchical spatial
  dependence on samples, and results in images which look better.
  """

  def __init__(self,
               input_shape,
               blockwise_splits,
               coupling_bijector_fn=None):
    """Creates the exit bijector.

    Args:
      input_shape: A list specifying the input shape to the exit bijector.
        Used in constructing the network.
      blockwise_splits: A list of integers specifying the number of channels
        exiting the model, as well as those being left in the model, and those
        bypassing the exit bijector altogether.
      coupling_bijector_fn: A function which takes the argument `input_shape`
        and returns a callable neural network (e.g. a keras Sequential). The
        network should either return a tensor with the same event shape as
        `input_shape` (this will employ additive coupling), a tensor with the
        same height and width as `input_shape` but twice the number of channels
        (this will employ affine coupling), or a bijector which takes in a
        tensor with event shape `input_shape`, and returns a tensor with shape
        `input_shape`.
    """
    parameters = dict(locals())
    nleave, ngrab, npass = blockwise_splits

    new_input_shape = input_shape[:-1]+(nleave,)
    target_output_shape = input_shape[:-1]+(ngrab,)

    # if nleave or ngrab == 0, then just use an identity for everything.
    if nleave == 0 or ngrab == 0:
      exit_layer = None
      exit_bijector_fn = None

      self.exit_layer = exit_layer
      shift_distribution = identity.Identity()

    else:
      exit_layer = coupling_bijector_fn(new_input_shape,
                                        output_chan=ngrab)
      exit_bijector_fn = self.make_bijector_fn(
          exit_layer,
          target_shape=target_output_shape,
          scale_fn=tf.exp)
      self.exit_layer = exit_layer  # For variable tracking.
      shift_distribution = real_nvp.RealNVP(
          num_masked=nleave,
          bijector_fn=exit_bijector_fn)

    super(ExitBijector, self).__init__(
        blockwise.Blockwise(
            [shift_distribution, identity.Identity()], [nleave + ngrab, npass]),
        parameters=parameters,
        name='exit_bijector')

  @staticmethod
  def make_bijector_fn(layer, target_shape, scale_fn=tf.nn.sigmoid):

    def bijector_fn(inputs, ignored_input):
      """Decorated function to get the RealNVP bijector."""
      # Build this so we can handle a user passing a NN that returns a tensor
      # OR an NN that returns a bijector
      possible_output = layer(inputs)

      # We need to produce a bijector, but we do not know if the layer has done
      # so. We are setting this up to handle 2 possibilities:
      # 1) The layer outputs a bijector --> all is good
      # 2) The layer outputs a tensor --> we need to turn it into a bijector.
      if isinstance(possible_output, bijector.Bijector):
        output = possible_output
      elif isinstance(possible_output, tf.Tensor):
        input_shape = inputs.get_shape().as_list()
        output_shape = possible_output.get_shape().as_list()
        assert input_shape[:-1] == output_shape[:-1]
        c = input_shape[-1]

        # For layers which output a tensor, we have two possibilities:
        # 1) There are twice as many output channels as the target --> the
        #    coupling is affine, meaning there is a scale followed by a shift.
        # 2) The number of output channels equals the target --> the
        #    coupling is additive, meaning there is just a shift
        if target_shape[-1] == output_shape[-1] // 2:
          this_scale = scale.Scale(scale_fn(possible_output[..., :c] + 2.))
          this_shift = shift.Shift(possible_output[..., c:])
          output = this_shift(this_scale)
        elif target_shape[-1] == output_shape[-1]:

          output = shift.Shift(possible_output[..., :c])
        else:
          raise ValueError('Shape inconsistent with input. Expected shape'
                           '{0} or {1} but tensor was shape {2}'.format(
                               input_shape, tf.concat(
                                   [input_shape[:-1],
                                    [2 * input_shape[-1]]], 0),
                               output_shape))
      else:
        raise ValueError('Expected a bijector or a tensor, but instead got'
                         '{}'.format(possible_output.__class__))
      return output

    return bijector_fn


class GlowBlock(composition.Composition):
  """Single block for a glow model.

  This bijector contains `num_steps` steps of the flow, each consisting of an
  actnorm-OneByOneConv-RealNVP chain of bijectors. Use of actnorm is optional
  and the RealNVP behavior is controlled by the coupling_bijector_fn, which
  implements a function (e.g. deep neural network) to dictate the behavior of
  the flow. A default (GlowDefaultNetwork) function is provided.
  """

  def __init__(self, input_shape, num_steps, coupling_bijector_fn,
               use_actnorm, seedstream):
    parameters = dict(locals())
    rnvp_block = [identity.Identity()]
    this_nchan = input_shape[-1]

    for j in range(num_steps):  # pylint: disable=unused-variable

      this_layer_input_shape = input_shape[:-1] + (input_shape[-1] // 2,)
      this_layer = coupling_bijector_fn(this_layer_input_shape)
      bijector_fn = self.make_bijector_fn(this_layer)

      # For each step in the block, we do (optional) actnorm, followed
      # by an invertible 1x1 convolution, then affine coupling.
      this_rnvp = invert.Invert(
          real_nvp.RealNVP(this_nchan // 2, bijector_fn=bijector_fn))

      # Append the layer to the realNVP bijector for variable tracking.
      this_rnvp.coupling_bijector_layer = this_layer
      rnvp_block.append(this_rnvp)

      rnvp_block.append(
          invert.Invert(OneByOneConv(
              this_nchan, seed=seedstream(),
              dtype=dtype_util.common_dtype(this_rnvp.variables,
                                            dtype_hint=tf.float32))))

      if use_actnorm:
        rnvp_block.append(ActivationNormalization(
            this_nchan,
            dtype=dtype_util.common_dtype(this_rnvp.variables,
                                          dtype_hint=tf.float32)))

    # Note that we reverse the list since Chain applies bijectors in reverse
    # order.
    super(GlowBlock, self).__init__(
        chain.Chain(rnvp_block[::-1]), parameters=parameters, name='glow_block')

  @staticmethod
  def make_bijector_fn(layer, scale_fn=tf.nn.sigmoid):

    def bijector_fn(inputs, ignored_input):
      """Decorated function to get the RealNVP bijector."""
      # Build this so we can handle a user passing a NN that returns a tensor
      # OR an NN that returns a bijector
      possible_output = layer(inputs)

      # We need to produce a bijector, but we do not know if the layer has done
      # so. We are setting this up to handle 2 possibilities:
      # 1) The layer outputs a bijector --> all is good
      # 2) The layer outputs a tensor --> we need to turn it into a bijector.
      if isinstance(possible_output, bijector.Bijector):
        output = possible_output
      elif isinstance(possible_output, tf.Tensor):
        input_shape = inputs.get_shape().as_list()
        output_shape = possible_output.get_shape().as_list()
        assert input_shape[:-1] == output_shape[:-1]
        c = input_shape[-1]

        # For layers which output a tensor, we have two possibilities:
        # 1) There are twice as many output channels as inputs --> the coupling
        #    is affine, meaning there is a scale followed by a shift.
        # 2) There are an equal number of input and output channels --> the
        #    coupling is additive, meaning there is just a shift
        if input_shape[-1] == output_shape[-1] // 2:
          this_scale = scale.Scale(scale_fn(possible_output[..., :c] + 2.))
          this_shift = shift.Shift(possible_output[..., c:])
          output = this_shift(this_scale)
        elif input_shape[-1] == output_shape[-1]:

          output = shift.Shift(possible_output[..., :c])
        else:
          raise ValueError('Shape inconsistent with input. Expected shape'
                           '{0} or {1} but tensor was shape {2}'.format(
                               input_shape, tf.concat(
                                   [input_shape[:-1],
                                    [2 * input_shape[-1]]], 0),
                               output_shape))
      else:
        raise ValueError('Expected a bijector or a tensor, but instead got'
                         '{}'.format(possible_output.__class__))
      return output

    return bijector_fn


class OneByOneConv(bijector.Bijector):
  """The 1x1 Conv bijector used in Glow.

  This class has a convenience function which initializes the parameters
  of the bijector.
  """

  def __init__(self, event_size, seed=None, dtype=tf.float32,
               name='OneByOneConv', **kwargs):
    parameters = dict(locals())
    with tf.name_scope(name) as bijector_name:
      lower_upper, permutation = self.trainable_lu_factorization(
          event_size, seed=seed, dtype=dtype)
      self._bijector = scale_matvec_lu.ScaleMatvecLU(
          lower_upper, permutation, **kwargs)
      super(OneByOneConv, self).__init__(
          dtype=self._bijector.lower_upper.dtype,
          is_constant_jacobian=True,
          forward_min_event_ndims=1,
          parameters=parameters,
          name=bijector_name)

  def forward(self, x):
    return self._bijector.forward(x)

  def inverse(self, y):
    return self._bijector.inverse(y)

  def inverse_log_det_jacobian(self, y, event_ndims=None):
    return self._bijector.inverse_log_det_jacobian(y, event_ndims)

  def forward_log_det_jacobian(self, x, event_ndims=None):
    return self._bijector.forward_log_det_jacobian(x, event_ndims)

  @staticmethod
  def trainable_lu_factorization(event_size,
                                 seed=None,
                                 dtype=tf.float32,
                                 name=None):
    with tf.name_scope(name or 'trainable_lu_factorization'):
      event_size = tf.convert_to_tensor(
          event_size, dtype_hint=tf.int32, name='event_size')
      random_matrix = tf.random.uniform(
          shape=[event_size, event_size],
          dtype=dtype,
          seed=seed)
      random_orthonormal = tf.linalg.qr(random_matrix)[0]
      lower_upper, permutation = tf.linalg.lu(random_orthonormal)
      lower_upper = tf.Variable(
          initial_value=lower_upper, trainable=True, name='lower_upper')
      # Initialize a non-trainable variable for the permutation indices so
      # that its value isn't re-sampled from run-to-run.
      permutation = tf.Variable(
          initial_value=permutation, trainable=False, name='permutation')
      return lower_upper, permutation


class ActivationNormalization(bijector.Bijector):
  """Bijector to implement Activation Normalization (ActNorm)."""

  def __init__(self, nchan, dtype=tf.float32, validate_args=False, name=None):
    parameters = dict(locals())

    self._initialized = tf.Variable(False, trainable=False)
    self._m = tf.Variable(tf.zeros(nchan, dtype))
    self._s = TransformedVariable(tf.ones(nchan, dtype), exp.Exp())
    self._bijector = invert.Invert(
        chain.Chain([
            scale.Scale(self._s),
            shift.Shift(self._m),
        ]))
    super(ActivationNormalization, self).__init__(
        validate_args=validate_args,
        forward_min_event_ndims=1,
        parameters=parameters,
        name=name or 'ActivationNormalization')

  def _inverse(self, y, **kwargs):
    with tf.control_dependencies([self._maybe_init(y, inverse=True)]):
      return self._bijector.inverse(y, **kwargs)

  def _forward(self, x, **kwargs):
    with tf.control_dependencies([self._maybe_init(x, inverse=False)]):
      return self._bijector.forward(x, **kwargs)

  def _inverse_log_det_jacobian(self, y, **kwargs):
    with tf.control_dependencies([self._maybe_init(y, inverse=True)]):
      return self._bijector.inverse_log_det_jacobian(y, 1, **kwargs)

  def _forward_log_det_jacobian(self, x, **kwargs):
    with tf.control_dependencies([self._maybe_init(x, inverse=False)]):
      return self._bijector.forward_log_det_jacobian(x, 1, **kwargs)

  def _maybe_init(self, inputs, inverse):
    """Initialize if not already initialized."""

    def _init():
      """Build the data-dependent initialization."""
      axis = prefer_static.range(prefer_static.rank(inputs) - 1)
      m = tf.math.reduce_mean(inputs, axis=axis)
      s = (
          tf.math.reduce_std(inputs, axis=axis) +
          10. * np.finfo(dtype_util.as_numpy_dtype(inputs.dtype)).eps)
      if inverse:
        s = 1 / s
        m = -m
      else:
        m = m / s
      with tf.control_dependencies([self._m.assign(m), self._s.assign(s)]):
        return self._initialized.assign(True)

    return tf.cond(self._initialized, tf.no_op, _init)


class Expand(composition.Composition):
  """A bijector to transform channels into spatial pixels."""

  def __init__(self, input_shape, block_size=2, validate_args=False, name=None):
    parameters = dict(locals())
    self._block_size = block_size
    _, h, w, c = prefer_static.split(input_shape, [-1, 1, 1, 1])
    h, w, c = h[0], w[0], c[0]
    n = self._block_size
    b = [
        reshape.Reshape(
            event_shape_out=[h * n, w * n, c // n**2],
            event_shape_in=[h, n, w, n, c // n**2]),
        transpose.Transpose(perm=[0, 3, 1, 4, 2]),
        reshape.Reshape(
            event_shape_in=[h, w, c],
            event_shape_out=[h, w, c // n**2, n, n]),
    ]
    super(Expand, self).__init__(
        bijectors=chain.Chain(b, validate_args=validate_args),
        name=name or 'Expand',
        parameters=parameters)


class GlowDefaultNetwork(tfk.Sequential):
  """Default network for the glow bijector.

  This builds a 3 layer convolutional network, with relu activation functions
  and he_normal initializer. The first and third layers have default kernel
  shape of 3, and the second layer is a 1x1 convolution. This is the setup
  in the public version of Glow.

  The output of the convolutional network defines the components of an Affine
  transformation (i.e. y = m * x + b), where m, x, and b are all tensors of
  the same shape, and * indicates elementwise multiplication.
  """

  def __init__(self, input_shape, num_hidden=400, kernel_shape=3):
    """Default network for glow bijector."""
    # Default is scale and shift, so 2c outputs.
    this_nchan = input_shape[-1] * 2
    conv_last = functools.partial(
        tfkl.Conv2D,
        padding='same',
        kernel_initializer=tf.initializers.zeros(),
        bias_initializer=tf.initializers.zeros())
    super(GlowDefaultNetwork, self).__init__([
        tfkl.Input(shape=input_shape),
        tfkl.Conv2D(num_hidden, kernel_shape, padding='same',
                    kernel_initializer=tf.initializers.he_normal(),
                    activation='relu'),
        tfkl.Conv2D(num_hidden, 1, padding='same',
                    kernel_initializer=tf.initializers.he_normal(),
                    activation='relu'),
        conv_last(this_nchan, kernel_shape)
    ])


class GlowDefaultExitNetwork(tfk.Sequential):
  """Default network for the glow exit bijector.

  This is just a single convolutional layer.
  """

  def __init__(self, input_shape, output_chan, kernel_shape=3):
    """Default network for glow bijector."""
    # Default is scale and shift, so 2c outputs.
    this_nchan = output_chan * 2
    conv = functools.partial(
        tfkl.Conv2D,
        padding='same',
        kernel_initializer=tf.initializers.zeros(),
        bias_initializer=tf.initializers.zeros())

    super(GlowDefaultExitNetwork, self).__init__([
        tfkl.Input(input_shape),
        conv(this_nchan, kernel_shape)])
