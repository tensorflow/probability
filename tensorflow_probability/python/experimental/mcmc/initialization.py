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
"""MCMC initializations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd

__all__ = [
    'init_near_unconstrained_zero',
]


def init_near_unconstrained_zero(
    model=None, constraining_bijector=None, event_shapes=None, dtypes=None):
  """Returns an initialization Distribution for starting a Markov chain.

  This initialization scheme follows Stan: we sample every latent
  independently, uniformly from -2 to 2 in its unconstrained space,
  and then transform into constrained space to construct an initial
  state that can be passed to `sample_chain` or other MCMC drivers.

  The argument signature is arranged to let the user pass either a
  `JointDistribution` describing their model, if it's in that form, or
  the essential information necessary for the sampling, namely a
  bijector (from unconstrained to constrained space) and the desired
  shape and dtype of each sample (specified in constrained space).

  Note: As currently implemented, this function has the limitation
  that the batch shape of the supplied model is ignored, but that
  could probably be generalized if needed.

  Args:
    model: A `Distribution` (typically a `JointDistribution`) giving the
      model to be initialized.  If supplied, it is queried for
      its default event space bijector, its event shape, and its dtype.
      If not supplied, those three elements must be supplied instead.
    constraining_bijector: A (typically multipart) `Bijector` giving
      the mapping from unconstrained to constrained space.  If
      supplied together with a `model`, acts as an override.  A nested
      structure of `Bijector`s is accepted, and interpreted as
      applying in parallel to a corresponding structure of state parts
      (see `JointMap` for details).
    event_shapes: A structure of shapes giving the (unconstrained)
      event space shape of the desired samples.  Must be an acceptable
      input to `constraining_bijector.inverse_event_shape`.  If
      supplied together with `model`, acts as an override.
    dtypes: A structure of dtypes giving the (unconstrained) dtypes of
      the desired samples.  Must be an acceptable input to
      `constraining_bijector.inverse_dtype`.  If supplied together
      with `model`, acts as an override.

  Returns:
    init_dist: A `Distribution` representing the initialization
      distribution, in constrained space.  Samples from this
      `Distribution` are valid initial states for a Markov chain
      targeting the model.

  #### Example

  Initialize 100 chains from the unconstrained -2, 2 distribution
  for a model expressed as a `JointDistributionCoroutine`:

  ```python
  @tfp.distributions.JointDistributionCoroutine
  def model():
    ...

  init_dist = tfp.experimental.mcmc.init_near_unconstrained_zero(model)
  states = tfp.mcmc.sample_chain(
    current_state=init_dist.sample(100, seed=[4, 8]),
    ...)
  ```

  """
  # Canonicalize arguments into the parts we need, namely
  # the constraining_bijector, the event_shapes, and the dtypes.
  if model is not None:
    # Got a Distribution model; treat other arguments as overrides if
    # present.
    if constraining_bijector is None:
      # pylint: disable=protected-access
      constraining_bijector = model.experimental_default_event_space_bijector()
    if event_shapes is None:
      event_shapes = model.event_shape
    if dtypes is None:
      dtypes = model.dtype
  elif constraining_bijector is None or event_shapes is None or dtypes is None:
    msg = ('Must pass either a Distribution (typically a JointDistribution), '
           'or a bijector, a structure of event shapes, and a structure '
           'of dtypes')
    raise ValueError(msg)

  # Interpret a structure of Bijectors as the joint multipart bijector.
  if not isinstance(constraining_bijector, tfb.Bijector):
    constraining_bijector = tfb.JointMap(constraining_bijector)

  # Actually initialize
  def one_term(shape, dtype):
    return tfd.Sample(
        tfd.Uniform(low=tf.constant(-2., dtype=dtype),
                    high=tf.constant(2., dtype=dtype)),
        sample_shape=shape)

  inv_shapes = constraining_bijector.inverse_event_shape(event_shapes)
  inv_dtypes = constraining_bijector.inverse_dtype(dtypes)
  terms = tf.nest.map_structure(one_term, inv_shapes, inv_dtypes)
  unconstrained = tfb.pack_sequence_as(inv_shapes)(
      tfd.JointDistributionSequential(tf.nest.flatten(terms)))
  return tfd.TransformedDistribution(
      unconstrained, bijector=constraining_bijector)
