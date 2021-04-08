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
from tensorflow_probability.python.internal import batched_rejection_sampler as brs
from tensorflow_probability.python.internal import tensorshape_util

__all__ = [
    'init_near_unconstrained_zero',
]


def init_near_unconstrained_zero(
    model=None, constraining_bijector=None, event_shapes=None,
    event_shape_tensors=None, dtypes=None):
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
    event_shape_tensors: A structure of tensors giving the (unconstrained)
      event space shape of the desired samples.  Must be an acceptable
      input to `constraining_bijector.inverse_event_shape_tensor`.  If
      supplied together with `model`, acts as an override. Required if any of
      `event_shapes` are not fully-defined.
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
    if event_shape_tensors is None:
      event_shape_tensors = model.event_shape_tensor()
    if dtypes is None:
      dtypes = model.dtype

  else:
    if constraining_bijector is None or event_shapes is None or dtypes is None:
      msg = ('Must pass either a Distribution (typically a JointDistribution), '
             'or a bijector, a structure of event shapes, and a '
             'structure of dtypes')
      raise ValueError(msg)
    event_shapes_fully_defined = all(tensorshape_util.is_fully_defined(s)
                                     for s in tf.nest.flatten(event_shapes))
    if not event_shapes_fully_defined and event_shape_tensors is None:
      raise ValueError('Must specify `event_shape_tensors` when `event_shapes` '
                       f'are not fully-defined: {event_shapes}')

  # Interpret a structure of Bijectors as the joint multipart bijector.
  if not isinstance(constraining_bijector, tfb.Bijector):
    constraining_bijector = tfb.JointMap(constraining_bijector)

  # Actually initialize
  def one_term(shape, shape_tensor, dtype):
    if not tensorshape_util.is_fully_defined(shape):
      shape = shape_tensor
    return tfd.Sample(
        tfd.Uniform(low=tf.constant(-2., dtype=dtype),
                    high=tf.constant(2., dtype=dtype)),
        sample_shape=shape)

  inv_shapes = constraining_bijector.inverse_event_shape(event_shapes)
  if event_shape_tensors is not None:
    inv_shape_tensors = constraining_bijector.inverse_event_shape_tensor(
        event_shape_tensors)
  else:
    inv_shape_tensors = tf.nest.map_structure(lambda _: None, inv_shapes)
  inv_dtypes = constraining_bijector.inverse_dtype(dtypes)
  terms = tf.nest.map_structure(
      one_term, inv_shapes, inv_shape_tensors, inv_dtypes)
  unconstrained = tfb.pack_sequence_as(inv_shapes)(
      tfd.JointDistributionSequential(tf.nest.flatten(terms)))
  return tfd.TransformedDistribution(
      unconstrained, bijector=constraining_bijector)


def retry_init(proposal_fn, target_fn, *args, max_trials=50,
               seed=None, name=None, **kwargs):
  """Tries an MCMC initialization proposal until it gets a valid state.

  In this case, "valid" is defined as the value of `target_fn` is
  finite.  This corresponds to an MCMC workflow where `target_fn`
  compute the log-probability one wants to sample from, in which case
  "finite `target_fn`" means "finite and positive probability state".
  If `target_fn` returns a Tensor of size greater than 1, the results
  are assumed to be independent of each other, so that different batch
  members can be accepted individually.

  The method is bounded rejection sampling.  The bound serves to avoid
  wasting computation on hopeless initialization procedures.  In
  interactive MCMC, one would presumably rather come up with a better
  initialization proposal than wait for an unbounded number of
  attempts with a bad one.  If unbounded re-trials are desired,
  set `max_trials` to `None`.

  Note: XLA and @jax.jit do not support assertions, so this function
  can return invalid states on those platforms without raising an
  error (unless `max_trials` is set to `None`).

  Args:
    proposal_fn: A function accepting a `seed` keyword argument and no other
      required arguments which generates proposed initial states.
    target_fn: A function accepting the return value of `proposal_fn`
      and returning a floating-point Tensor.
    *args: Additional arguments passed to `proposal_fn`.
    max_trials: Size-1 integer `Tensor` or None. Maximum number of
      calls to `proposal_fn` to attempt.  If acceptable states are not
      found in this many trials, `retry_init` signals an error.  If
      `None`, there is no limit, and `retry_init` skips the control
      flow cost of checking for success.
    seed: Optional, a PRNG seed for reproducible sampling.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'mcmc_sample_chain').
    **kwargs: Additional keyword arguments passed to `proposal_fn`.

  Returns:
    states: An acceptable result from `proposal_fn`.

  #### Example

  One popular MCMC initialization scheme is to start the chains near 0
  in unconstrained space.  There are models where the unconstraining
  transformation cannot exactly capture the space of valid states,
  such that this initialization has some material but not overwhelming
  chance of failure.  In this case, we can use `retry_init` to compensate.

  ```python
  @tfp.distributions.JointDistributionCoroutine
  def model():
    ...

  raw_init_dist = tfp.experimental.mcmc.init_near_unconstrained_zero(model)
  init_states = tfp.experimental.mcmc.retry_init(
    proposal_fn=raw_init_dist.sample,
    target_fn=model.log_prob,
    sample_shape=[100],
    seed=[4, 8])
  states = tfp.mcmc.sample_chain(
    current_state=init_states,
    ...)
  ```

  """
  def trial(seed):
    values = proposal_fn(*args, seed=seed, **kwargs)
    log_probs = target_fn(values)
    success = tf.math.is_finite(log_probs)
    return values, success
  with tf.name_scope(name or 'mcmc_retry_init'):
    values, successes, _ = brs.batched_las_vegas_algorithm(
        trial, max_trials=max_trials, seed=seed)
    if max_trials is None:
      # We were authorized to compute until success, so no need to
      # check for failure
      return values
    else:
      num_states = tf.size(successes)
      num_successes = tf.reduce_sum(tf.cast(successes, tf.int32))
      msg = ('Failed to find acceptable initial states after {} trials;\n'
             '{} of {} states have non-finite log probability').format(
                 max_trials, num_states - num_successes, num_states)
      with tf.control_dependencies([tf.debugging.assert_equal(
          successes, tf.ones_like(successes), message=msg)]):
        return tf.nest.map_structure(tf.identity, values)
