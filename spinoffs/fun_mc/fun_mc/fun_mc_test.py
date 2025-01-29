# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for fun_mc."""

import collections
import functools
import os

from absl.testing import parameterized
import jax as real_jax
from jax import config as jax_config
import numpy as np
import scipy.stats
import tensorflow.compat.v2 as real_tf
from tensorflow_probability.python.internal import test_util as tfp_test_util
from fun_mc import backend
from fun_mc import fun_mc_lib as fun_mc
from fun_mc import prefab
from fun_mc import test_util

jnp = backend.jnp
jax = backend.jax
tfp = backend.tfp
util = backend.util

real_tf.enable_v2_behavior()
real_tf.experimental.numpy.experimental_enable_numpy_behavior()
jax_config.update('jax_enable_x64', True)

TestNamedTuple = collections.namedtuple('TestNamedTuple', 'x, y')

BACKEND = None  # Rewritten by backends/rewrite.py.

if BACKEND == 'backend_jax':
  os.environ['XLA_FLAGS'] = (
      f'{os.environ.get("XLA_FLAGS", "")} '
      '--xla_force_host_platform_device_count=4'
  )


def _test_seed():
  return tfp_test_util.test_seed() % (2**32 - 1)


def _no_compile(fn):
  return fn


def _fwd_mclachlan_optimal_4th_order_step(*args, **kwargs):
  return fun_mc.mclachlan_optimal_4th_order_step(*args, forward=True, **kwargs)


def _rev_mclachlan_optimal_4th_order_step(*args, **kwargs):
  return fun_mc.mclachlan_optimal_4th_order_step(*args, forward=False, **kwargs)


def _skip_on_jax(fn):
  @functools.wraps(fn)
  def _wrapper(self, *args, **kwargs):
    if not self._is_on_jax:
      return fn(self, *args, **kwargs)

  return _wrapper


def _gen_cov(data, axis):
  """This computes a generalized covariance, supporting batch and reduction.

  This computes a batch of covariances from data, with the relevant dimensions
  are determined as follows:

  - Dimensions specified by `axis` are reduced over.
  - The final non-reduced over dimension is taken as the event dimension.
  - The remaining dimensions are batch dimensions.

  Args:
    data: An NDArray.
    axis: An integer or a tuple of integers.

  Returns:
    cov: A batch of covariances.
  """
  axis = tuple(util.flatten_tree(axis))
  shape = tuple(data.shape)
  rank = len(shape)
  centered_data = data - np.mean(data, axis, keepdims=True)
  symbols = 'abcdefg'
  # Destination is missing the axes we reduce over.
  dest = []
  last_unaggregated_src_dim = None
  last_unaggregated_dest_dim = None
  for i in range(rank):
    if i not in axis:
      dest.append(symbols[i])
      last_unaggregated_src_dim = i
      last_unaggregated_dest_dim = len(dest) - 1
  source_1 = list(symbols[:rank])
  source_1[last_unaggregated_src_dim] = 'x'
  source_2 = list(symbols[:rank])
  source_2[last_unaggregated_src_dim] = 'y'
  dest = (
      dest[:last_unaggregated_dest_dim]
      + ['x', 'y']
      + dest[last_unaggregated_dest_dim + 1 :]
  )
  formula = '{source_1},{source_2}->{dest}'.format(
      source_1=''.join(source_1), source_2=''.join(source_2), dest=''.join(dest)
  )
  cov = np.einsum(formula, centered_data, centered_data) / np.prod(
      np.array(shape)[np.array(axis)]
  )
  return cov


class GenCovTest(real_tf.test.TestCase):

  def testGenCov(self):
    x = np.arange(10).reshape(5, 2)
    true_cov = np.cov(x, rowvar=False, bias=True)
    self.assertAllClose(true_cov, _gen_cov(x, 0))

    true_cov = np.cov(x, rowvar=True, bias=True)
    self.assertAllClose(true_cov, _gen_cov(x, 1))


class FunMCTest(tfp_test_util.TestCase, parameterized.TestCase):
  _is_on_jax = BACKEND == 'backend_jax'

  def _make_seed(self, seed):
    if self._is_on_jax:
      return real_jax.random.PRNGKey(seed)
    else:
      return util.make_tensor_seed([seed, 0])

  @property
  def _dtype(self):
    raise NotImplementedError()

  def _constant(self, value):
    return jnp.array(value, self._dtype)

  @parameterized.named_parameters(
      ('Unrolled', True),
      ('NotUnrolled', False),
  )
  def testTraceSingle(self, unroll):
    def fun(x):
      return x + 1.0, 2 * x

    x, e_trace = fun_mc.trace(
        state=0.0,
        fn=fun,
        num_steps=5,
        trace_fn=lambda _, xp1: xp1,
        unroll=unroll,
    )

    self.assertAllEqual(5.0, x)
    self.assertAllEqual([0.0, 2.0, 4.0, 6.0, 8.0], e_trace)

  @parameterized.named_parameters(
      ('Unrolled', True),
      ('NotUnrolled', False),
  )
  def testTraceNested(self, unroll):
    def fun(x, y):
      return (x + 1.0, y + 2.0), ()

    (x, y), (x_trace, y_trace) = fun_mc.trace(
        state=(0.0, 0.0),
        fn=fun,
        num_steps=5,
        trace_fn=lambda xy, _: xy,
        unroll=unroll,
    )

    self.assertAllEqual(5.0, x)
    self.assertAllEqual(10.0, y)
    self.assertAllEqual([1.0, 2.0, 3.0, 4.0, 5.0], x_trace)
    self.assertAllEqual([2.0, 4.0, 6.0, 8.0, 10.0], y_trace)

  @parameterized.named_parameters(
      ('Unrolled', True),
      ('NotUnrolled', False),
  )
  def testTraceTrace(self, unroll):
    def fun(x):
      return fun_mc.trace(
          x, lambda x: (x + 1.0, x + 1.0), 2, trace_mask=False, unroll=unroll
      )

    x, trace = fun_mc.trace(0.0, fun, 2)
    self.assertAllEqual(4.0, x)
    self.assertAllEqual([2.0, 4.0], trace)

  def testTraceDynamic(self):

    @jax.jit
    def trace_n(num_steps):
      return fun_mc.trace(
          0,
          lambda x: (x + 1, (10 * x, 100 * x)),
          num_steps,
          max_steps=6,
          trace_mask=(True, False),
      )

    x, (traced, untraced) = trace_n(5)
    self.assertAllEqual(5, x)
    self.assertAllEqual(40, traced[4])
    self.assertEqual(6, traced.shape[0])
    self.assertAllEqual(400, untraced)

  @parameterized.named_parameters(
      ('Unrolled', True),
      ('NotUnrolled', False),
  )
  def testTraceMaxSteps(self, unroll):
    x, (traced, untraced) = fun_mc.trace(
        0,
        lambda x: (x + 1, (10 * x, 100 * x)),
        5,
        max_steps=6,
        unroll=unroll,
        trace_mask=(True, False),
    )
    self.assertAllEqual(5, x)
    self.assertAllEqual(40, traced[4])
    self.assertEqual(6, traced.shape[0])
    self.assertAllEqual(400, untraced)

  @parameterized.named_parameters(
      ('Unrolled', True),
      ('NotUnrolled', False),
  )
  def testTraceStopFnSingle(self, unroll):
    x, (traced, untraced) = fun_mc.trace(
        0,
        lambda x: (x + 1, (10 * x, 100 * x)),
        5,
        unroll=unroll,
        trace_mask=(True, False),
        stop_fn=lambda x, _: x == 1,
    )
    self.assertAllEqual(1, x)
    self.assertAllEqual(0, traced[0])
    self.assertEqual(5, traced.shape[0])
    self.assertAllEqual(0, untraced)

  @parameterized.named_parameters(
      ('Unrolled', True),
      ('NotUnrolled', False),
  )
  def testTraceStopFnMulti(self, unroll):
    x, (traced, untraced) = fun_mc.trace(
        0,
        lambda x: (x + 1, (10 * x, 100 * x)),
        5,
        unroll=unroll,
        trace_mask=(True, False),
        stop_fn=lambda x, _: x == 3,
    )
    self.assertAllEqual(3, x)
    self.assertAllEqual(20, traced[2])
    self.assertEqual(5, traced.shape[0])
    self.assertAllEqual(200, untraced)

  @parameterized.named_parameters(
      ('Unrolled', True),
      ('NotUnrolled', False),
  )
  def testTraceMask(self, unroll):
    def fun(x):
      return x + 1, (2 * x, 3 * x)

    x, (trace_1, trace_2) = fun_mc.trace(
        state=0, fn=fun, num_steps=3, trace_mask=(True, False), unroll=unroll
    )

    self.assertAllEqual(3, x)
    self.assertAllEqual([0, 2, 4], trace_1)
    self.assertAllEqual(6, trace_2)

    x, (trace_1, trace_2) = fun_mc.trace(
        state=0, fn=fun, num_steps=3, trace_mask=False, unroll=unroll
    )

    self.assertAllEqual(3, x)
    self.assertAllEqual(4, trace_1)
    self.assertAllEqual(6, trace_2)

  def testInterruptibleTrace(self):
    def fun(x, y):
      x = x + 1.0
      y = y + 2.0
      return (x, y), (x, y)

    state, _ = fun_mc.trace(
        state=fun_mc.interruptible_trace_init((0.0, 0.0), fn=fun, num_steps=5),
        fn=functools.partial(fun_mc.interruptible_trace_step, fn=fun),
        num_steps=4,
    )

    x_trace, y_trace = state.trace()

    self.assertAllEqual([1.0, 2.0, 3.0, 4.0], x_trace)
    self.assertAllEqual([2.0, 4.0, 6.0, 8.0], y_trace)

  def testInterruptibleTraceMask(self):
    def fun(x, y):
      x = x + 1.0
      y = y + 2.0
      return (x, y), (x, y)

    state, _ = fun_mc.trace(
        state=fun_mc.interruptible_trace_init(
            (0.0, 0.0), fn=fun, num_steps=5, trace_mask=(True, False)
        ),
        fn=functools.partial(fun_mc.interruptible_trace_step, fn=fun),
        num_steps=4,
    )

    x_trace, y = state.trace()

    self.assertAllEqual([1.0, 2.0, 3.0, 4.0], x_trace)
    self.assertAllEqual(8.0, y)

  def testCallFn(self):
    sum_fn = lambda *args: sum(args)

    self.assertEqual(1, fun_mc.call_fn(sum_fn, 1))
    self.assertEqual(3, fun_mc.call_fn(sum_fn, (1, 2)))

  def testCallFnDict(self):
    sum_fn = lambda a, b: a + b

    self.assertEqual(3, fun_mc.call_fn(sum_fn, [1, 2]))
    self.assertEqual(3, fun_mc.call_fn(sum_fn, {'a': 1, 'b': 2}))

  @parameterized.named_parameters(
      ('ArgsToTuple1', (1,), {}, (1,)),
      ('ArgsToList1', (1,), {}, [1]),
      ('ArgsToTuple3', (1, 2, 3), {}, [1, 2, 3]),
      ('ArgsToList3', (1, 2, 3), {}, [1, 2, 3]),
      (
          'ArgsToOrdDict3',
          (1, 2, 3),
          {},
          collections.OrderedDict([('c', 1), ('b', 2), ('a', 3)]),
      ),
      (
          'ArgsKwargsToOrdDict3',
          (1, 2),
          {'a': 3},
          collections.OrderedDict([('c', 1), ('b', 2), ('a', 3)]),
      ),
      (
          'KwargsToOrdDict3',
          (),
          {'a': 3, 'b': 2, 'c': 1},
          collections.OrderedDict([('c', 1), ('b', 2), ('a', 3)]),
      ),
      ('KwargsToDict3', (), {'a': 3, 'b': 2, 'c': 1}, {'c': 1, 'b': 2, 'a': 3}),
      ('ArgsToNamedTuple', (TestNamedTuple(1, 2),), {}, TestNamedTuple(1, 2)),
      (
          'KwargsToNamedTuple',
          (),
          {'a': TestNamedTuple(1, 2)},
          TestNamedTuple(1, 2),
      ),
      ('ArgsToScalar', (1,), {}, 1),
      ('KwargsToScalar', (), {'a': 1}, 1),
      ('Tuple0', (), {}, ()),
      ('List0', (), {}, []),
      ('Dict0', (), {}, {}),
  )
  def testRecoverStateFromArgs(self, args, kwargs, state_structure):
    state = fun_mc.recover_state_from_args(args, kwargs, state_structure)
    self.assertEqual(type(state_structure), type(state))
    self.assertAllEqual(state_structure, state)

  @parameterized.named_parameters(
      ('BadKwargs', (), {'a': 1, 'b': 2}, 'c'),
      ('ArgsOverlap', (1, 2), {'c': 1, 'b': 2}, 'a'),
  )
  def testRecoverStateFromArgsMissing(self, args, kwargs, missing):
    state_structure = collections.OrderedDict([('c', 1), ('b', 2), ('a', 3)])
    with self.assertRaisesRegex(
        ValueError, "Missing '{}' from kwargs.".format(missing)
    ):
      fun_mc.recover_state_from_args(args, kwargs, state_structure)

  @parameterized.named_parameters(
      ('Tuple1', {'a': 1}, (1,)),
      ('List1', {'a': 1}, [1]),
  )
  def testRecoverStateFromArgsNoKwargs(self, kwargs, state_structure):
    with self.assertRaisesRegex(ValueError, 'This wrapper does not'):
      fun_mc.recover_state_from_args((), kwargs, state_structure)

  def testBroadcastStructure(self):
    struct = fun_mc.maybe_broadcast_structure(1, [1, 2])
    self.assertEqual([1, 1], struct)

    struct = fun_mc.maybe_broadcast_structure([3, 4], [1, 2])
    self.assertEqual([3, 4], struct)

    struct = fun_mc.maybe_broadcast_structure([1, 2], [[0, 0], [0, 0, 0]])
    self.assertEqual([[1, 1], [2, 2, 2]], struct)

  def testCallPotentialFn(self):
    def potential(x):
      return x, ()

    x, extra = fun_mc.call_potential_fn(potential, 0.0)

    self.assertEqual(0.0, x)
    self.assertEqual((), extra)

  def testCallPotentialFnMissingExtra(self):
    def potential(x):
      return x

    with self.assertRaisesRegex(TypeError, 'A common solution is to adjust'):
      fun_mc.call_potential_fn(potential, 0.0)

  def testCallTransitionOperator(self):
    def kernel(x, y):
      del y
      return [x, [1]], ()

    [x, [y]], extra = fun_mc.call_transition_operator(kernel, [0.0, None])
    self.assertEqual(0.0, x)
    self.assertEqual(1, y)
    self.assertEqual((), extra)

  def testCallTransitionOperatorMissingExtra(self):
    def potential(x):
      return x

    with self.assertRaisesRegex(TypeError, 'A common solution is to adjust'):
      fun_mc.call_transition_operator(potential, 0.0)

  def testCallTransitionOperatorBadArgs(self):
    def potential(x, y, z):
      del z
      return (x, y), ()

    with self.assertRaisesRegex(TypeError, 'The structure of `new_args=`'):
      fun_mc.call_transition_operator(potential, (1, 2, 3))

  def testTransformLogProbFn(self):
    def log_prob_fn(x, y):
      return (
          tfp.distributions.Normal(self._constant(0.0), 1.0).log_prob(x)
          + tfp.distributions.Normal(self._constant(1.0), 1.0).log_prob(y)
      ), ()

    bijectors = [
        tfp.bijectors.Scale(scale=self._constant(2.0)),
        tfp.bijectors.Scale(scale=self._constant(3.0)),
    ]

    (transformed_log_prob_fn, transformed_init_state) = (
        fun_mc.transform_log_prob_fn(
            log_prob_fn, bijectors, [self._constant(2.0), self._constant(3.0)]
        )
    )

    self.assertIsInstance(transformed_init_state, list)
    self.assertAllClose([1.0, 1.0], transformed_init_state)
    tlp, (orig_space, _) = transformed_log_prob_fn(
        self._constant(1.0), self._constant(1.0)
    )
    lp = log_prob_fn(self._constant(2.0), self._constant(3.0))[0] + sum(
        b.forward_log_det_jacobian(self._constant(1.0), event_ndims=0)
        for b in bijectors
    )

    self.assertAllClose([2.0, 3.0], orig_space)
    self.assertAllClose(lp, tlp)

  def testTransformLogProbFnKwargs(self):
    def log_prob_fn(x, y):
      return (
          tfp.distributions.Normal(self._constant(0.0), 1.0).log_prob(x)
          + tfp.distributions.Normal(self._constant(1.0), 1.0).log_prob(y)
      ), ()

    bijectors = {
        'x': tfp.bijectors.Scale(scale=self._constant(2.0)),
        'y': tfp.bijectors.Scale(scale=self._constant(3.0)),
    }

    (transformed_log_prob_fn, transformed_init_state) = (
        fun_mc.transform_log_prob_fn(
            log_prob_fn,
            bijectors,
            {
                'x': self._constant(2.0),
                'y': self._constant(3.0),
            },
        )
    )

    self.assertIsInstance(transformed_init_state, dict)
    self.assertAllCloseNested(
        {
            'x': self._constant(1.0),
            'y': self._constant(1.0),
        },
        transformed_init_state,
    )

    tlp, (orig_space, _) = transformed_log_prob_fn(
        x=self._constant(1.0), y=self._constant(1.0)
    )
    lp = log_prob_fn(x=self._constant(2.0), y=self._constant(3.0))[0] + sum(
        b.forward_log_det_jacobian(self._constant(1.0), event_ndims=0)
        for b in bijectors.values()
    )

    self.assertAllCloseNested(
        {'x': self._constant(2.0), 'y': self._constant(3.0)}, orig_space
    )
    self.assertAllClose(lp, tlp)

  # The +1's here are because we initialize the `state_grads` at 1, which
  # require an extra call to `target_log_prob_fn`.
  @parameterized.named_parameters(
      ('Leapfrog', lambda: fun_mc.leapfrog_step, 1 + 1),
      ('Ruth4', lambda: fun_mc.ruth4_step, 3 + 1),
      ('Blanes3', lambda: fun_mc.blanes_3_stage_step, 3 + 1),
      (
          'McLachlan4Fwd',
          lambda: _fwd_mclachlan_optimal_4th_order_step,
          4 + 1,
          9,
      ),
      (
          'McLachlan4Rev',
          lambda: _rev_mclachlan_optimal_4th_order_step,
          4 + 1,
          9,
      ),
  )
  def testIntegratorStep(
      self, method_fn, num_tlp_calls, num_tlp_calls_jax=None
  ):
    method = method_fn()

    tlp_call_counter = [0]

    def target_log_prob_fn(q):
      tlp_call_counter[0] += 1
      return -(q**2), 1.0

    def kinetic_energy_fn(p):
      return jnp.abs(p) ** 3.0, 2.0

    state = self._constant(1.0)
    _, _, state_grads = fun_mc.call_potential_fn_with_grads(
        target_log_prob_fn,
        state,
    )

    state, extras = method(
        integrator_step_state=fun_mc.IntegratorStepState(
            state=state, state_grads=state_grads, momentum=self._constant(2.0)
        ),
        step_size=self._constant(0.1),
        target_log_prob_fn=target_log_prob_fn,
        kinetic_energy_fn=kinetic_energy_fn,
    )

    if num_tlp_calls_jax is not None and self._is_on_jax:
      num_tlp_calls = num_tlp_calls_jax
    self.assertEqual(num_tlp_calls, tlp_call_counter[0])
    self.assertEqual(1.0, extras.state_extra)
    self.assertEqual(2.0, extras.kinetic_energy_extra)

    initial_hamiltonian = (
        -target_log_prob_fn(self._constant(1.0))[0]
        + kinetic_energy_fn(self._constant(2.0))[0]
    )
    fin_hamiltonian = (
        -target_log_prob_fn(state.state)[0]
        + kinetic_energy_fn(state.momentum)[0]
    )

    self.assertAllClose(fin_hamiltonian, initial_hamiltonian, atol=0.2)

  @parameterized.named_parameters(
      ('Leapfrog', fun_mc.leapfrog_step),
      ('Ruth4', fun_mc.ruth4_step),
      ('Blanes3', fun_mc.blanes_3_stage_step),
  )
  def testIntegratorStepReversible(self, method):
    def target_log_prob_fn(q):
      return -(q**2), []

    def kinetic_energy_fn(p):
      return p**2.0, []

    seed = self._make_seed(_test_seed())

    state = self._constant(1.0)
    _, _, state_grads = fun_mc.call_potential_fn_with_grads(
        target_log_prob_fn,
        state,
    )

    state_fwd, _ = method(
        integrator_step_state=fun_mc.IntegratorStepState(
            state=state,
            state_grads=state_grads,
            momentum=util.random_normal([], self._dtype, seed),
        ),
        step_size=self._constant(0.1),
        target_log_prob_fn=target_log_prob_fn,
        kinetic_energy_fn=kinetic_energy_fn,
    )

    state_rev, _ = method(
        integrator_step_state=state_fwd._replace(momentum=-state_fwd.momentum),
        step_size=self._constant(0.1),
        target_log_prob_fn=target_log_prob_fn,
        kinetic_energy_fn=kinetic_energy_fn,
    )

    self.assertAllClose(state, state_rev.state, atol=1e-6)

  def testMclachlanIntegratorStepReversible(self):
    def target_log_prob_fn(q):
      return -(q**2), []

    def kinetic_energy_fn(p):
      return p**2.0, []

    seed = self._make_seed(_test_seed())

    state = self._constant(1.0)
    _, _, state_grads = fun_mc.call_potential_fn_with_grads(
        target_log_prob_fn,
        state,
    )

    state_fwd, _ = _fwd_mclachlan_optimal_4th_order_step(
        integrator_step_state=fun_mc.IntegratorStepState(
            state=state,
            state_grads=state_grads,
            momentum=util.random_normal([], self._dtype, seed),
        ),
        step_size=self._constant(0.1),
        target_log_prob_fn=target_log_prob_fn,
        kinetic_energy_fn=kinetic_energy_fn,
    )

    state_rev, _ = _rev_mclachlan_optimal_4th_order_step(
        integrator_step_state=state_fwd._replace(momentum=-state_fwd.momentum),
        step_size=self._constant(0.1),
        target_log_prob_fn=target_log_prob_fn,
        kinetic_energy_fn=kinetic_energy_fn,
    )

    self.assertAllClose(state, state_rev.state, atol=1e-6)

  def testMetropolisHastingsStep(self):
    seed = self._make_seed(_test_seed())

    zero = self._constant(0.0)
    one = self._constant(1.0)

    accepted, mh_extra = fun_mc.metropolis_hastings_step(
        current_state=zero, proposed_state=one, energy_change=-np.inf, seed=seed
    )
    self.assertAllEqual(one, accepted)
    self.assertAllEqual(True, mh_extra.is_accepted)

    accepted, mh_extra = fun_mc.metropolis_hastings_step(
        current_state=zero, proposed_state=one, energy_change=np.inf, seed=seed
    )
    self.assertAllEqual(zero, accepted)
    self.assertAllEqual(False, mh_extra.is_accepted)

    accepted, mh_extra = fun_mc.metropolis_hastings_step(
        current_state=zero, proposed_state=one, energy_change=np.nan, seed=seed
    )
    self.assertAllEqual(zero, accepted)
    self.assertAllEqual(False, mh_extra.is_accepted)

    accepted, mh_extra = fun_mc.metropolis_hastings_step(
        current_state=zero, proposed_state=one, energy_change=np.nan, seed=seed
    )
    self.assertAllEqual(zero, accepted)
    self.assertAllEqual(False, mh_extra.is_accepted)

    accepted, mh_extra = fun_mc.metropolis_hastings_step(
        current_state=zero,
        proposed_state=one,
        log_uniform=-one,
        energy_change=self._constant(-np.log(0.5)),
        seed=seed,
    )
    self.assertAllEqual(one, accepted)
    self.assertAllEqual(True, mh_extra.is_accepted)

    accepted, mh_extra = fun_mc.metropolis_hastings_step(
        current_state=zero,
        proposed_state=one,
        log_uniform=zero,
        energy_change=self._constant(-np.log(0.5)),
        seed=seed,
    )
    self.assertAllEqual(zero, accepted)
    self.assertAllEqual(False, mh_extra.is_accepted)

    accepted, _ = fun_mc.metropolis_hastings_step(
        current_state=jnp.zeros(1000, dtype=self._dtype),
        proposed_state=jnp.ones(1000, dtype=self._dtype),
        energy_change=-jnp.log(0.5 * jnp.ones(1000, dtype=self._dtype)),
        seed=seed,
    )
    self.assertAllClose(0.5, jnp.mean(accepted), rtol=0.1)

  def testMetropolisHastingsStepStructure(self):
    struct_type = collections.namedtuple('Struct', 'a, b')

    current = struct_type([1, 2], (3, [4, [0, 0]]))
    proposed = struct_type([5, 6], (7, [8, [0, 0]]))

    accepted, mh_extra = fun_mc.metropolis_hastings_step(
        current_state=current,
        proposed_state=proposed,
        energy_change=-np.inf,
        seed=self._make_seed(_test_seed()),
    )
    self.assertAllEqual(True, mh_extra.is_accepted)
    self.assertAllEqual(
        util.flatten_tree(proposed), util.flatten_tree(accepted)
    )

  @parameterized.named_parameters(
      ('Unrolled', True),
      ('NotUnrolled', False),
  )
  def testBasicHMC(self, unroll):
    step_size = self._constant(0.2)
    num_steps = 2000
    num_leapfrog_steps = 10
    state = jnp.ones([16, 2], dtype=self._dtype)

    base_mean = self._constant([2.0, 3.0])
    base_scale = self._constant([2.0, 0.5])

    def target_log_prob_fn(x):
      return -jnp.sum(0.5 * jnp.square((x - base_mean) / base_scale), -1), ()

    def kernel(hmc_state, seed):
      hmc_seed, seed = util.split_seed(seed, 2)
      hmc_state, _ = fun_mc.hamiltonian_monte_carlo_step(
          hmc_state,
          step_size=step_size,
          num_integrator_steps=num_leapfrog_steps,
          target_log_prob_fn=target_log_prob_fn,
          unroll_integrator=unroll,
          seed=hmc_seed,
      )
      return (hmc_state, seed), hmc_state.state

    seed = self._make_seed(_test_seed())

    _, chain = jax.jit(
        lambda state, seed: fun_mc.trace(  # pylint: disable=g-long-lambda
            state=(
                fun_mc.hamiltonian_monte_carlo_init(state, target_log_prob_fn),
                seed,
            ),
            fn=kernel,
            num_steps=num_steps,
        )
    )(state, seed)
    # Discard the warmup samples.
    chain = chain[1000:]

    sample_mean = jnp.mean(chain, axis=[0, 1])
    sample_var = jnp.var(chain, axis=[0, 1])

    true_samples = (
        util.random_normal(shape=[4096, 2], dtype=self._dtype, seed=seed)
        * base_scale
        + base_mean
    )

    true_mean = jnp.mean(true_samples, axis=0)
    true_var = jnp.var(true_samples, axis=0)

    self.assertAllClose(true_mean, sample_mean, rtol=0.1, atol=0.1)
    self.assertAllClose(true_var, sample_var, rtol=0.1, atol=0.1)

  def testPreconditionedHMC(self):
    step_size = self._constant(0.2)
    num_steps = 2000
    num_leapfrog_steps = 10
    state = jnp.ones([16, 2], dtype=self._dtype)

    base_mean = self._constant([1.0, 0])
    base_cov = self._constant([[1, 0.5], [0.5, 1]])

    bijector = tfp.bijectors.Softplus()
    base_dist = tfp.distributions.MultivariateNormalFullCovariance(
        loc=base_mean, covariance_matrix=base_cov
    )
    target_dist = bijector(base_dist)

    def orig_target_log_prob_fn(x):
      return target_dist.log_prob(x), ()

    target_log_prob_fn, state = fun_mc.transform_log_prob_fn(
        orig_target_log_prob_fn, bijector, state
    )

    # pylint: disable=g-long-lambda
    def kernel(hmc_state, seed):
      hmc_seed, seed = util.split_seed(seed, 2)
      hmc_state, _ = fun_mc.hamiltonian_monte_carlo_step(
          hmc_state,
          step_size=step_size,
          num_integrator_steps=num_leapfrog_steps,
          target_log_prob_fn=target_log_prob_fn,
          seed=hmc_seed,
      )
      return (hmc_state, seed), hmc_state.state_extra[0]

    seed = self._make_seed(_test_seed())

    _, chain = jax.jit(
        lambda state, seed: fun_mc.trace(  # pylint: disable=g-long-lambda
            state=(
                fun_mc.hamiltonian_monte_carlo_init(state, target_log_prob_fn),
                seed,
            ),
            fn=kernel,
            num_steps=num_steps,
        )
    )(state, seed)
    # Discard the warmup samples.
    chain = chain[1000:]

    sample_mean = jnp.mean(chain, axis=[0, 1])
    sample_cov = tfp.stats.covariance(chain, sample_axis=[0, 1])

    true_samples = target_dist.sample(4096, seed=self._make_seed(_test_seed()))

    true_mean = jnp.mean(true_samples, axis=0)
    true_cov = tfp.stats.covariance(chain, sample_axis=[0, 1])

    self.assertAllClose(true_mean, sample_mean, rtol=0.1, atol=0.1)
    self.assertAllClose(true_cov, sample_cov, rtol=0.1, atol=0.1)

  def testAdaptiveStepSize(self):
    step_size = self._constant(0.2)
    num_steps = 200
    num_adapt_steps = 100
    num_leapfrog_steps = 10
    state = jnp.ones([16, 2], dtype=self._dtype)

    base_mean = self._constant([1.0, 0])
    base_cov = self._constant([[1, 0.5], [0.5, 1]])

    @jax.jit
    def computation(state, seed):
      bijector = tfp.bijectors.Softplus()
      base_dist = tfp.distributions.MultivariateNormalFullCovariance(
          loc=base_mean, covariance_matrix=base_cov
      )
      target_dist = bijector(base_dist)

      def orig_target_log_prob_fn(x):
        return target_dist.log_prob(x), ()

      target_log_prob_fn, state = fun_mc.transform_log_prob_fn(
          orig_target_log_prob_fn, bijector, state
      )

      def kernel(hmc_state, step_size_state, step, seed):
        hmc_seed, seed = util.split_seed(seed, 2)
        hmc_state, hmc_extra = fun_mc.hamiltonian_monte_carlo_step(
            hmc_state,
            step_size=jnp.exp(step_size_state.state),
            num_integrator_steps=num_leapfrog_steps,
            target_log_prob_fn=target_log_prob_fn,
            seed=hmc_seed,
        )

        rate = prefab._polynomial_decay(  # pylint: disable=protected-access
            step=step,
            step_size=self._constant(0.01),
            power=0.5,
            decay_steps=num_adapt_steps,
            final_step_size=0.0,
        )
        mean_p_accept = jnp.mean(
            jnp.exp(
                jnp.minimum(self._constant(0.0), hmc_extra.log_accept_ratio)
            )
        )

        loss_fn = fun_mc.make_surrogate_loss_fn(
            lambda _: (0.9 - mean_p_accept, ())
        )
        step_size_state, _ = fun_mc.adam_step(
            step_size_state, loss_fn, learning_rate=rate
        )

        return (
            (hmc_state, step_size_state, step + 1, seed),
            (hmc_state.state_extra[0], hmc_extra.log_accept_ratio),
        )

      _, (chain, log_accept_ratio_trace) = fun_mc.trace(
          state=(
              fun_mc.hamiltonian_monte_carlo_init(state, target_log_prob_fn),
              fun_mc.adam_init(jnp.log(step_size)),
              0,
              seed,
          ),
          fn=kernel,
          num_steps=num_adapt_steps + num_steps,
      )
      true_samples = target_dist.sample(
          4096, seed=self._make_seed(_test_seed())
      )
      return chain, log_accept_ratio_trace, true_samples

    seed = self._make_seed(_test_seed())
    chain, log_accept_ratio_trace, true_samples = computation(state, seed)

    log_accept_ratio_trace = log_accept_ratio_trace[num_adapt_steps:]
    chain = chain[num_adapt_steps:]

    sample_mean = jnp.mean(chain, axis=[0, 1])
    sample_cov = tfp.stats.covariance(chain, sample_axis=[0, 1])

    true_mean = jnp.mean(true_samples, axis=0)
    true_cov = tfp.stats.covariance(chain, sample_axis=[0, 1])

    self.assertAllClose(true_mean, sample_mean, rtol=0.05, atol=0.05)
    self.assertAllClose(true_cov, sample_cov, rtol=0.05, atol=0.05)
    self.assertAllClose(
        jnp.mean(jnp.exp(jnp.minimum(0.0, log_accept_ratio_trace))),
        0.9,
        rtol=0.1,
    )

  def testSignAdaptation(self):
    new_control = fun_mc.sign_adaptation(
        control=self._constant(1.0),
        output=self._constant(0.5),
        set_point=self._constant(1.0),
        adaptation_rate=self._constant(0.1),
    )
    self.assertAllClose(new_control, 1.0 / 1.1)

    new_control = fun_mc.sign_adaptation(
        control=self._constant(1.0),
        output=self._constant(0.5),
        set_point=self._constant(0.0),
        adaptation_rate=self._constant(0.1),
    )
    self.assertAllClose(new_control, 1.0 * 1.1)

  def testOBABOLangevinIntegrator(self):
    def target_log_prob_fn(q):
      return -(q**2), q

    def kinetic_energy_fn(p):
      return p**2.0, p

    def momentum_refresh_fn(p, seed):
      del seed
      return p

    def energy_change_fn(old_is, new_is):
      old_energy = -old_is.target_log_prob + old_is.momentum**2
      new_energy = -new_is.target_log_prob + new_is.momentum**2
      return new_energy - old_energy, ()

    integrator_step_fn = lambda state: fun_mc.leapfrog_step(  # pylint: disable=g-long-lambda
        state,
        step_size=0.1,
        target_log_prob_fn=target_log_prob_fn,
        kinetic_energy_fn=kinetic_energy_fn,
    )

    lt_integrator_fn = lambda state: fun_mc.obabo_langevin_integrator(  # pylint: disable=g-long-lambda
        state,
        num_steps=10,
        integrator_step_fn=integrator_step_fn,
        energy_change_fn=energy_change_fn,
        momentum_refresh_fn=momentum_refresh_fn,
        integrator_trace_fn=lambda state, _, __: state.state,
        seed=self._make_seed(_test_seed()),
    )

    ham_integrator_fn = lambda state: fun_mc.hamiltonian_integrator(  # pylint: disable=g-long-lambda
        state,
        num_steps=10,
        kinetic_energy_fn=kinetic_energy_fn,
        integrator_step_fn=integrator_step_fn,
        integrator_trace_fn=lambda state, _: state.state,
    )

    state = jnp.zeros([2], dtype=self._dtype)
    momentum = jnp.ones([2], dtype=self._dtype)
    target_log_prob, _, state_grads = fun_mc.call_potential_fn_with_grads(
        target_log_prob_fn, state
    )

    start_state = fun_mc.IntegratorState(
        target_log_prob=target_log_prob,
        momentum=momentum,
        state=state,
        state_grads=state_grads,
        state_extra=state,
    )

    lt_state, lt_extra = lt_integrator_fn(start_state)
    ham_state, ham_extra = ham_integrator_fn(start_state)

    # Langevin and Hamiltonian integrator should be the same when no noise is
    # present.
    self.assertAllClose(lt_state, ham_state)
    self.assertAllClose(lt_extra.energy_change, ham_extra.energy_change)
    self.assertAllClose(
        lt_extra.energy_change,
        ham_extra.final_energy - ham_extra.initial_energy,
    )
    self.assertAllClose(lt_extra.integrator_trace, ham_extra.integrator_trace)

  def testRaggedIntegrator(self):
    def target_log_prob_fn(q):
      return -(q**2), q

    def kinetic_energy_fn(p):
      return jnp.abs(p) ** 3.0, p

    integrator_fn = lambda state, num_steps: fun_mc.hamiltonian_integrator(  # pylint: disable=g-long-lambda
        state,
        num_steps=num_steps,
        integrator_step_fn=lambda state: fun_mc.leapfrog_step(  # pylint: disable=g-long-lambda
            state,
            step_size=0.1,
            target_log_prob_fn=target_log_prob_fn,
            kinetic_energy_fn=kinetic_energy_fn,
        ),
        kinetic_energy_fn=kinetic_energy_fn,
        integrator_trace_fn=lambda state, extra: (state, extra),
    )

    state = jnp.zeros([2], dtype=self._dtype)
    momentum = jnp.ones([2], dtype=self._dtype)
    target_log_prob, _, state_grads = fun_mc.call_potential_fn_with_grads(
        target_log_prob_fn, state
    )

    start_state = fun_mc.IntegratorState(
        target_log_prob=target_log_prob,
        momentum=momentum,
        state=state,
        state_grads=state_grads,
        state_extra=state,
    )

    state_1 = integrator_fn(start_state, 1)
    state_2 = integrator_fn(start_state, 2)
    state_1_2 = integrator_fn(start_state, [1, 2])

    # Make sure integrators actually integrated to different points.
    self.assertFalse(np.all(state_1[0].state == state_2[0].state))

    # Ragged integration should be consistent with the non-ragged equivalent.
    def get_batch(state, idx):
      # For the integrator trace, we'll grab the final value.
      return util.map_tree(
          lambda x: x[idx] if len(x.shape) == 1 else x[-1, idx], state
      )

    self.assertAllClose(get_batch(state_1, 0), get_batch(state_1_2, 0))
    self.assertAllClose(get_batch(state_2, 0), get_batch(state_1_2, 1))

    # Ragged traces should be equal up to the number of steps for the batch
    # element.
    def get_slice(state, num, idx):
      return util.map_tree(lambda x: x[:num, idx], state[1].integrator_trace)

    self.assertAllClose(get_slice(state_1, 1, 0), get_slice(state_1_2, 1, 0))
    self.assertAllClose(get_slice(state_2, 2, 0), get_slice(state_1_2, 2, 1))

  def testRaggedIntegratorMaxSteps(self):
    def target_log_prob_fn(q):
      return -(q**2), q

    def kinetic_energy_fn(p):
      return jnp.abs(p) ** 3.0, p

    integrator_fn = lambda state, num_steps: fun_mc.hamiltonian_integrator(  # pylint: disable=g-long-lambda
        state,
        num_steps=num_steps,
        integrator_step_fn=lambda state: fun_mc.leapfrog_step(  # pylint: disable=g-long-lambda
            state,
            step_size=0.1,
            target_log_prob_fn=target_log_prob_fn,
            kinetic_energy_fn=kinetic_energy_fn,
        ),
        kinetic_energy_fn=kinetic_energy_fn,
        max_num_steps=3,
        integrator_trace_fn=lambda state, extra: (state, extra),
    )

    state = jnp.zeros([2], dtype=self._dtype)
    momentum = jnp.ones([2], dtype=self._dtype)
    target_log_prob, _, state_grads = fun_mc.call_potential_fn_with_grads(
        target_log_prob_fn, state
    )

    start_state = fun_mc.IntegratorState(
        target_log_prob=target_log_prob,
        momentum=momentum,
        state=state,
        state_grads=state_grads,
        state_extra=state,
    )

    state_1 = integrator_fn(start_state, 1)
    state_2 = integrator_fn(start_state, 2)
    state_1_2 = integrator_fn(start_state, [1, 2])

    # Make sure integrators actually integrated to different points.
    self.assertFalse(np.all(state_1[0].state == state_2[0].state))

    # Ragged integration should be consistent with the non-ragged equivalent.
    def get_batch(state, idx):
      # For the integrator trace, we'll grab the final value.
      return util.map_tree(
          lambda x: x[idx] if len(x.shape) == 1 else x[-1, idx], state
      )

    self.assertAllClose(get_batch(state_1, 0), get_batch(state_1_2, 0))
    self.assertAllClose(get_batch(state_2, 0), get_batch(state_1_2, 1))

    # Ragged traces should be equal up to the number of steps for the batch
    # element.
    def get_slice(state, num, idx):
      return util.map_tree(lambda x: x[:num, idx], state[1].integrator_trace)

    self.assertAllClose(get_slice(state_1, 1, 0), get_slice(state_1_2, 1, 0))
    self.assertAllClose(get_slice(state_2, 2, 0), get_slice(state_1_2, 2, 1))

    self.assertAllEqual(
        3, util.flatten_tree(state_1[1].integrator_trace)[0].shape[0]
    )
    self.assertAllEqual(
        3, util.flatten_tree(state_2[1].integrator_trace)[0].shape[0]
    )
    self.assertAllEqual(
        3, util.flatten_tree(state_1_2[1].integrator_trace)[0].shape[0]
    )

  def testAdam(self):
    def loss_fn(x, y):
      return jnp.square(x - 1.0) + jnp.square(y - 2.0), []

    _, [(x, y), loss] = fun_mc.trace(
        fun_mc.adam_init([self._constant(0.0), self._constant(0.0)]),
        lambda adam_state: fun_mc.adam_step(  # pylint: disable=g-long-lambda
            adam_state, loss_fn, learning_rate=self._constant(0.01)
        ),
        num_steps=1000,
        trace_fn=lambda state, extra: [state.state, extra.loss],
    )

    self.assertAllClose(1.0, x[-1], atol=1e-3)
    self.assertAllClose(2.0, y[-1], atol=1e-3)
    self.assertAllClose(0.0, loss[-1], atol=1e-3)

  def testGradientDescent(self):
    def loss_fn(x, y):
      return jnp.square(x - 1.0) + jnp.square(y - 2.0), []

    _, [(x, y), loss] = fun_mc.trace(
        fun_mc.GradientDescentState([self._constant(0.0), self._constant(0.0)]),
        lambda gd_state: fun_mc.gradient_descent_step(  # pylint: disable=g-long-lambda
            gd_state, loss_fn, learning_rate=self._constant(0.01)
        ),
        num_steps=1000,
        trace_fn=lambda state, extra: [state.state, extra.loss],
    )

    self.assertAllClose(1.0, x[-1], atol=1e-3)
    self.assertAllClose(2.0, y[-1], atol=1e-3)
    self.assertAllClose(0.0, loss[-1], atol=1e-3)

  def testSimpleDualAverages(self):
    def loss_fn(x, y):
      return jnp.square(x - 1.0) + jnp.square(y - 2.0), []

    def kernel(sda_state, rms_state):
      sda_state, _ = fun_mc.simple_dual_averages_step(sda_state, loss_fn, 1.0)
      rms_state, _ = fun_mc.running_mean_step(rms_state, sda_state.state)
      return (sda_state, rms_state), rms_state.mean

    _, (x, y) = fun_mc.trace(
        (
            fun_mc.simple_dual_averages_init(
                [self._constant(0.0), self._constant(0.0)]
            ),
            fun_mc.running_mean_init([[], []], [self._dtype, self._dtype]),
        ),
        kernel,
        num_steps=1000,
    )

    self.assertAllClose(1.0, x[-1], atol=1e-1)
    self.assertAllClose(2.0, y[-1], atol=1e-1)

  def testGaussianProposal(self):
    num_samples = 1000

    state = {
        'x': jnp.zeros([num_samples, 1], dtype=self._dtype) + self._constant(
            [0.0, 1.0]
        ),
        'y': jnp.zeros([num_samples], dtype=self._dtype) + self._constant(3.0),
    }
    scale = 2.0

    state_samples, _ = fun_mc.gaussian_proposal(
        state, scale=scale, seed=self._make_seed(_test_seed())
    )

    _, p_val_x0 = scipy.stats.kstest(
        state_samples['x'][:, 0],
        lambda x: scipy.stats.norm.cdf(x, loc=0.0, scale=2.0),
    )
    _, p_val_x1 = scipy.stats.kstest(
        state_samples['x'][:, 1],
        lambda x: scipy.stats.norm.cdf(x, loc=1.0, scale=2.0),
    )
    _, p_val_y = scipy.stats.kstest(
        state_samples['y'],
        lambda x: scipy.stats.norm.cdf(x, loc=3.0, scale=2.0),
    )
    self.assertGreater(p_val_x0, 1e-3)
    self.assertGreater(p_val_x1, 1e-3)
    self.assertGreater(p_val_y, 1e-3)

    mean = util.map_tree(lambda x: jnp.mean(x, 0), state_samples)
    variance = util.map_tree(lambda x: jnp.var(x, 0), state_samples)

    self.assertAllClose(state['x'][0], mean['x'], atol=0.2)
    self.assertAllClose(state['y'][0], mean['y'], atol=0.2)
    self.assertAllClose(
        scale**2 * jnp.ones_like(mean['x']), variance['x'], atol=0.5
    )
    self.assertAllClose(
        scale**2 * jnp.ones_like(mean['y']), variance['y'], atol=0.5
    )

  def testGaussianProposalNamedAxis(self):
    if BACKEND != 'backend_jax':
      self.skipTest('JAX-only')

    state = {
        'sharded': jnp.zeros([4], self._dtype),
        'shared': jnp.zeros([], self._dtype),
    }
    in_axes = {
        'sharded': 0,
        'shared': None,
    }
    named_axis = {
        'sharded': 'named_axis',
        'shared': None,
    }

    @functools.partial(
        real_jax.pmap, in_axes=(in_axes, None), axis_name='named_axis'
    )
    def proposal_fn(state, seed):
      samples, _ = fun_mc.gaussian_proposal(
          state, scale=1.0, named_axis=named_axis, seed=seed
      )
      return samples

    samples = proposal_fn(state, self._make_seed(_test_seed()))

    self.assertAllClose(samples['shared'][0], samples['shared'][1])
    self.assertTrue(
        np.any(np.abs(samples['sharded'][0] - samples['sharded'][1]) > 1e-3)
    )

  def testMaximalReflectiveProposal(self):
    state = {
        'x': self._constant([[0.0, 1.0], [2.0, 3.0]]),
        'y': self._constant([3.0, 4.0]),
    }
    scale = 2.0

    def kernel(seed):
      proposal_seed, seed = util.split_seed(seed, 2)
      new_state, (extra, _) = fun_mc.maximal_reflection_coupling_proposal(
          state, scale=scale, seed=proposal_seed
      )
      return seed, (new_state, extra.coupling_proposed)

    # Simulate an MCMC run.
    _, (state_samples, coupling_proposed) = fun_mc.trace(
        self._make_seed(_test_seed()), kernel, 1000
    )

    mean = util.map_tree(lambda x: jnp.mean(x, 0), state_samples)
    variance = util.map_tree(lambda x: jnp.var(x, 0), state_samples)

    _, p_val_x00 = scipy.stats.kstest(
        state_samples['x'][:, 0, 0],
        lambda x: scipy.stats.norm.cdf(x, loc=0.0, scale=2.0),
    )
    _, p_val_x01 = scipy.stats.kstest(
        state_samples['x'][:, 0, 1],
        lambda x: scipy.stats.norm.cdf(x, loc=1.0, scale=2.0),
    )
    _, p_val_x10 = scipy.stats.kstest(
        state_samples['x'][:, 1, 0],
        lambda x: scipy.stats.norm.cdf(x, loc=2.0, scale=2.0),
    )
    _, p_val_x11 = scipy.stats.kstest(
        state_samples['x'][:, 1, 1],
        lambda x: scipy.stats.norm.cdf(x, loc=3.0, scale=2.0),
    )
    _, p_val_y0 = scipy.stats.kstest(
        state_samples['y'][:, 0],
        lambda x: scipy.stats.norm.cdf(x, loc=3.0, scale=2.0),
    )
    _, p_val_y1 = scipy.stats.kstest(
        state_samples['y'][:, 1],
        lambda x: scipy.stats.norm.cdf(x, loc=4.0, scale=2.0),
    )
    self.assertGreater(p_val_x00, 1e-3)
    self.assertGreater(p_val_x01, 1e-3)
    self.assertGreater(p_val_x10, 1e-3)
    self.assertGreater(p_val_x11, 1e-3)
    self.assertGreater(p_val_y0, 1e-3)
    self.assertGreater(p_val_y1, 1e-3)

    self.assertAllClose(state['x'], mean['x'], atol=0.2)
    self.assertAllClose(state['y'], mean['y'], atol=0.2)
    self.assertAllClose(
        scale**2 * jnp.ones_like(mean['x']), variance['x'], atol=0.5
    )
    self.assertAllClose(
        scale**2 * jnp.ones_like(mean['y']), variance['y'], atol=0.5
    )

    coupled = coupling_proposed & np.array(
        util.flatten_tree(
            util.map_tree(
                lambda x: jnp.all(  # pylint: disable=g-long-lambda
                    (x[:, :1] == x[:, 1:]), tuple(range(2, len(x.shape)))
                ),
                state_samples,
            )
        )
    ).all(0)
    self.assertAllClose(coupling_proposed, coupled)

  def testMaximalReflectiveProposalNamedAxis(self):
    if BACKEND != 'backend_jax':
      self.skipTest('JAX-only')

    state = {
        # The trailing shape is [coupled axis, independent chains].
        'sharded': (
            jnp.zeros([4, 2, 1024], self._dtype)
            + self._constant([0.0, 1.0])[:, jnp.newaxis]
        ),
        'shared': jnp.zeros([2, 1024], self._dtype),
    }
    in_axes = {
        'sharded': 0,
        'shared': None,
    }
    named_axis = {
        'sharded': 'named_axis',
        'shared': None,
    }

    @functools.partial(
        real_jax.pmap, in_axes=(in_axes, None), axis_name='named_axis'
    )
    def proposal_fn(state, seed):
      samples, (extra, _) = fun_mc.maximal_reflection_coupling_proposal(
          state, chain_ndims=1, scale=1.0, named_axis=named_axis, seed=seed
      )
      return samples, extra

    samples, extra = proposal_fn(state, self._make_seed(_test_seed()))

    self.assertAllClose(samples['shared'][0], samples['shared'][1])
    self.assertAllClose(extra.log_couple_ratio[0], extra.log_couple_ratio[1])
    self.assertAllClose(extra.coupling_proposed[0], extra.coupling_proposed[1])
    self.assertTrue(
        np.any(np.abs(samples['sharded'][0] - samples['sharded'][1]) > 1e-3)
    )

  def testHMCNamedAxis(self):
    if BACKEND != 'backend_jax':
      self.skipTest('JAX-only')

    state = {
        'sharded': jnp.zeros([4, 1024], self._dtype),
        'shared': jnp.zeros([1024], self._dtype),
    }
    in_axes = {
        'sharded': 0,
        'shared': None,
    }
    named_axis = {
        'sharded': 'named_axis',
        'shared': None,
    }

    def target_log_prob_fn(sharded, shared):
      return (
          -(
              backend.distribute_lib.psum(jnp.square(sharded), 'named_axis')
              + jnp.square(shared)
          ),
          (),
      )

    @functools.partial(
        real_jax.pmap, in_axes=(in_axes, None), axis_name='named_axis'
    )
    def kernel(state, seed):
      hmc_state = fun_mc.hamiltonian_monte_carlo_init(
          state, target_log_prob_fn=target_log_prob_fn
      )
      hmc_state, hmc_extra = fun_mc.hamiltonian_monte_carlo_step(
          hmc_state,
          step_size=self._constant(0.2),
          num_integrator_steps=4,
          target_log_prob_fn=target_log_prob_fn,
          named_axis=named_axis,
          seed=seed,
      )
      return hmc_state, hmc_extra

    seed = self._make_seed(_test_seed())
    hmc_state, hmc_extra = kernel(state, seed)
    self.assertAllClose(
        hmc_state.state['shared'][0], hmc_state.state['shared'][1]
    )
    self.assertTrue(
        np.any(
            np.abs(
                hmc_state.state['sharded'][0] - hmc_state.state['sharded'][1]
            )
            > 1e-3
        )
    )
    self.assertAllClose(hmc_extra.is_accepted[0], hmc_extra.is_accepted[1])
    self.assertAllClose(
        hmc_extra.log_accept_ratio[0], hmc_extra.log_accept_ratio[1]
    )

  def testRandomWalkMetropolis(self):
    num_steps = 1000
    state = jnp.ones([16], dtype=jnp.int32)
    target_logits = self._constant([1.0, 2.0, 3.0, 4.0]) + 2.0
    proposal_logits = self._constant([4.0, 3.0, 2.0, 1.0]) + 2.0

    def target_log_prob_fn(x):
      return jnp.asarray(target_logits)[x], ()

    def proposal_fn(x, seed):
      current_logits = proposal_logits[x]
      proposal = util.random_categorical(
          proposal_logits[jnp.newaxis], x.shape[0], seed
      )[0]
      proposed_logits = proposal_logits[proposal]
      return jnp.array(proposal, x.dtype), (
          (),
          proposed_logits - current_logits,
      )

    def kernel(rwm_state, seed):
      rwm_seed, seed = util.split_seed(seed, 2)
      rwm_state, rwm_extra = fun_mc.random_walk_metropolis_step(
          rwm_state,
          target_log_prob_fn=target_log_prob_fn,
          proposal_fn=proposal_fn,
          seed=rwm_seed,
      )
      return (rwm_state, seed), rwm_extra

    seed = self._make_seed(_test_seed())

    _, chain = jax.jit(
        lambda state, seed: fun_mc.trace(  # pylint: disable=g-long-lambda
            state=(
                fun_mc.random_walk_metropolis_init(state, target_log_prob_fn),
                seed,
            ),
            fn=kernel,
            num_steps=num_steps,
            trace_fn=lambda state, extra: state[0].state,
        )
    )(state, seed)
    # Discard the warmup samples.
    chain = chain[500:]

    sample_mean = jnp.mean(jax.nn.one_hot(chain, 4), axis=[0, 1])
    self.assertAllClose(jax.nn.softmax(target_logits), sample_mean, atol=0.11)

  @parameterized.named_parameters(
      ('Basic', (10, 3), None),
      ('Batched', (10, 4, 3), None),
      ('Aggregated0', (10, 4, 3), 0),
      ('Aggregated1', (10, 4, 3), 1),
      ('Aggregated01', (10, 4, 3), (0, 1)),
      ('Aggregated02', (10, 4, 5, 3), (0, 2)),
  )
  def testRunningMean(self, shape, aggregation):
    rng = np.random.RandomState(_test_seed())
    data = self._constant(rng.randn(*shape))

    def kernel(rms, idx):
      rms, _ = fun_mc.running_mean_step(rms, data[idx], axis=aggregation)
      return (rms, idx + 1), ()

    true_aggregation = (0,) + (
        ()
        if aggregation is None
        else tuple([a + 1 for a in util.flatten_tree(aggregation)])
    )
    true_mean = np.mean(data, true_aggregation)

    (rms, _), _ = fun_mc.trace(
        state=(fun_mc.running_mean_init(true_mean.shape, data.dtype), 0),
        fn=kernel,
        num_steps=len(data),
        trace_fn=lambda *args: (),
    )

    self.assertAllClose(true_mean, rms.mean)

  def testRunningMeanMaxPoints(self):
    window_size = 100
    rng = np.random.RandomState(_test_seed())
    data = self._constant(
        np.concatenate(
            [rng.randn(window_size), 1.0 + 2.0 * rng.randn(window_size * 10)],
            axis=0,
        )
    )

    def kernel(rms, idx):
      rms, _ = fun_mc.running_mean_step(rms, data[idx], window_size=window_size)
      return (rms, idx + 1), rms.mean

    _, mean = fun_mc.trace(
        state=(fun_mc.running_mean_init([], data.dtype), 0),
        fn=kernel,
        num_steps=len(data),
    )
    # Up to window_size, we compute the running mean exactly.
    self.assertAllClose(np.mean(data[:window_size]), mean[window_size - 1])
    # After window_size, we're doing exponential moving average, and pick up the
    # mean after the change in the distribution. Since the moving average is
    # computed only over ~window_size points, this test is rather noisy.
    self.assertAllClose(1.0, mean[-1], atol=0.2)

  @parameterized.named_parameters(
      ('Basic', (10, 3), None),
      ('Batched', (10, 4, 3), None),
      ('Aggregated0', (10, 4, 3), 0),
      ('Aggregated1', (10, 4, 3), 1),
      ('Aggregated01', (10, 4, 3), (0, 1)),
      ('Aggregated02', (10, 4, 5, 3), (0, 2)),
  )
  def testRunningVariance(self, shape, aggregation):
    rng = np.random.RandomState(_test_seed())
    data = self._constant(rng.randn(*shape))

    true_aggregation = (0,) + (
        ()
        if aggregation is None
        else tuple([a + 1 for a in util.flatten_tree(aggregation)])
    )
    true_mean = np.mean(data, true_aggregation)
    true_var = np.var(data, true_aggregation)

    def kernel(rvs, idx):
      rvs, _ = fun_mc.running_variance_step(rvs, data[idx], axis=aggregation)
      return (rvs, idx + 1), ()

    (rvs, _), _ = fun_mc.trace(
        state=(fun_mc.running_variance_init(true_mean.shape, data[0].dtype), 0),
        fn=kernel,
        num_steps=len(data),
        trace_fn=lambda *args: (),
    )
    self.assertAllClose(true_mean, rvs.mean)
    self.assertAllClose(true_var, rvs.variance)

  def testRunningVarianceMaxPoints(self):
    window_size = 100
    rng = np.random.RandomState(_test_seed())
    data = self._constant(
        np.concatenate(
            [rng.randn(window_size), 1.0 + 2.0 * rng.randn(window_size * 10)],
            axis=0,
        )
    )

    def kernel(rvs, idx):
      rvs, _ = fun_mc.running_variance_step(
          rvs, data[idx], window_size=window_size
      )
      return (rvs, idx + 1), (rvs.mean, rvs.variance)

    _, (mean, var) = fun_mc.trace(
        state=(fun_mc.running_variance_init([], data.dtype), 0),
        fn=kernel,
        num_steps=len(data),
    )
    # Up to window_size, we compute the running mean/variance exactly.
    self.assertAllClose(np.mean(data[:window_size]), mean[window_size - 1])
    self.assertAllClose(np.var(data[:window_size]), var[window_size - 1])
    # After window_size, we're doing exponential moving average, and pick up the
    # mean/variance after the change in the distribution. Since the moving
    # average is computed only over ~window_size points, this test is rather
    # noisy.
    self.assertAllClose(1.0, mean[-1], atol=0.2)
    self.assertAllClose(4.0, var[-1], atol=0.8)

  @parameterized.named_parameters(
      ('Basic', (10, 3), None),
      ('Batched', (10, 4, 3), None),
      ('Aggregated0', (10, 4, 3), 0),
      ('Aggregated01', (10, 4, 5, 3), (0, 1)),
  )
  def testRunningCovariance(self, shape, aggregation):
    rng = np.random.RandomState(_test_seed())
    data = self._constant(rng.randn(*shape))

    true_aggregation = (0,) + (
        ()
        if aggregation is None
        else tuple([a + 1 for a in util.flatten_tree(aggregation)])
    )
    true_mean = np.mean(data, true_aggregation)
    true_cov = _gen_cov(data, true_aggregation)

    def kernel(rcs, idx):
      rcs, _ = fun_mc.running_covariance_step(rcs, data[idx], axis=aggregation)
      return (rcs, idx + 1), ()

    (rcs, _), _ = fun_mc.trace(
        state=(
            fun_mc.running_covariance_init(true_mean.shape, data[0].dtype),
            0,
        ),
        fn=kernel,
        num_steps=len(data),
        trace_fn=lambda *args: (),
    )
    self.assertAllClose(true_mean, rcs.mean)
    self.assertAllClose(true_cov, rcs.covariance)

  def testRunningCovarianceMaxPoints(self):
    window_size = 100
    rng = np.random.RandomState(_test_seed())
    data = self._constant(
        np.concatenate(
            [
                rng.randn(window_size, 2),
                np.array([1.0, 2.0])
                + np.array([2.0, 3.0]) * rng.randn(window_size * 10, 2),
            ],
            axis=0,
        )
    )

    def kernel(rvs, idx):
      rvs, _ = fun_mc.running_covariance_step(
          rvs, data[idx], window_size=window_size
      )
      return (rvs, idx + 1), (rvs.mean, rvs.covariance)

    _, (mean, cov) = fun_mc.trace(
        state=(fun_mc.running_covariance_init([2], data.dtype), 0),
        fn=kernel,
        num_steps=len(data),
    )
    # Up to window_size, we compute the running mean/variance exactly.
    self.assertAllClose(
        np.mean(data[:window_size], axis=0), mean[window_size - 1]
    )
    self.assertAllClose(
        _gen_cov(data[:window_size], axis=0), cov[window_size - 1]
    )
    # After window_size, we're doing exponential moving average, and pick up the
    # mean/variance after the change in the distribution. Since the moving
    # average is computed only over ~window_size points, this test is rather
    # noisy.
    self.assertAllClose(np.array([1.0, 2.0]), mean[-1], atol=0.2)
    self.assertAllClose(np.array([[4.0, 0.0], [0.0, 9.0]]), cov[-1], atol=1.0)

  @parameterized.named_parameters(
      ('BasicScalar', (10, 20), 1),
      ('BatchedScalar', (10, 5, 20), 2),
      ('BasicVector', (10, 5, 20), 1),
      ('BatchedVector', (10, 5, 20, 7), 2),
  )
  def testPotentialScaleReduction(self, chain_shape, independent_chain_ndims):
    rng = np.random.RandomState(_test_seed())
    chain_means = rng.randn(*((1,) + chain_shape[1:])).astype(np.float32)
    chains = 0.4 * rng.randn(*chain_shape).astype(np.float32) + chain_means

    true_rhat = tfp.mcmc.potential_scale_reduction(
        chains, independent_chain_ndims=independent_chain_ndims
    )

    chains = self._constant(chains)
    psrs, _ = fun_mc.trace(
        state=fun_mc.potential_scale_reduction_init(
            chain_shape[1:], self._dtype
        ),
        fn=lambda psrs: fun_mc.potential_scale_reduction_step(  # pylint: disable=g-long-lambda
            psrs, chains[psrs.num_points]
        ),
        num_steps=chain_shape[0],
        trace_fn=lambda *_: (),
    )

    running_rhat = fun_mc.potential_scale_reduction_extract(
        psrs, independent_chain_ndims=independent_chain_ndims
    )
    self.assertAllClose(true_rhat, running_rhat)

  @parameterized.named_parameters(
      ('Basic', (), None, None),
      ('Batched1', (2,), 1, None),
      ('Batched2', (3, 2), 1, None),
      ('Aggregated0', (3, 2), 1, 0),
      ('Aggregated01', (3, 4, 2), 1, (0, 1)),
  )
  def testRunningApproximateAutoCovariance(
      self, state_shape, event_ndims, aggregation
  ):
    # We'll use HMC as the source of our chain.
    # While HMC is being sampled, we also compute the running autocovariance.
    step_size = 0.2
    num_steps = 1000
    num_leapfrog_steps = 10
    max_lags = 300

    state = jnp.zeros(state_shape, dtype=self._dtype)

    def target_log_prob_fn(x):
      lp = -0.5 * jnp.square(x)
      if event_ndims is None:
        return lp, ()
      else:
        return jnp.sum(lp, -1), ()

    def kernel(hmc_state, raac_state, seed):
      hmc_seed, seed = util.split_seed(seed, 2)
      hmc_state, hmc_extra = fun_mc.hamiltonian_monte_carlo_step(
          hmc_state,
          step_size=step_size,
          num_integrator_steps=num_leapfrog_steps,
          target_log_prob_fn=target_log_prob_fn,
          seed=hmc_seed,
      )
      raac_state, _ = fun_mc.running_approximate_auto_covariance_step(
          raac_state, hmc_state.state, axis=aggregation
      )
      return (hmc_state, raac_state, seed), hmc_extra

    seed = self._make_seed(_test_seed())

    # Subtle: Unlike TF, JAX needs a data dependency from the inputs to outputs
    # for the jit to do anything.
    (_, raac_state, _), chain = jax.jit(
        lambda state, seed: fun_mc.trace(  # pylint: disable=g-long-lambda
            state=(
                fun_mc.hamiltonian_monte_carlo_init(state, target_log_prob_fn),
                fun_mc.running_approximate_auto_covariance_init(
                    max_lags=max_lags,
                    state_shape=state_shape,
                    dtype=state.dtype,
                    axis=aggregation,
                ),
                seed,
            ),
            fn=kernel,
            num_steps=num_steps,
            trace_fn=lambda state, extra: state[0].state,
        )
    )(state, seed)

    true_aggregation = (0,) + (
        ()
        if aggregation is None
        else tuple([a + 1 for a in util.flatten_tree(aggregation)])
    )
    true_variance = np.array(jnp.var(np.array(chain), true_aggregation))
    true_autocov = np.array(
        tfp.stats.auto_correlation(np.array(chain), axis=0, max_lags=max_lags)
    )
    if aggregation is not None:
      true_autocov = jnp.mean(
          true_autocov, [a + 1 for a in util.flatten_tree(aggregation)]
      )

    self.assertAllClose(true_variance, raac_state.auto_covariance[0], 1e-5)
    self.assertAllClose(
        true_autocov,
        raac_state.auto_covariance / raac_state.auto_covariance[0],
        atol=0.1,
    )

  @parameterized.named_parameters(
      ('Positional1', 0.0),
      ('Positional2', (0.0, 1.0)),
      ('Named1', {'a': 0.0}),
      ('Named2', {'a': 0.0, 'b': 1.0}),
  )
  def testSurrogateLossFn(self, state):
    def grad_fn(*args, **kwargs):
      # This is uglier than user code due to the parameterized test...
      new_state = util.unflatten_tree(state, util.flatten_tree((args, kwargs)))
      return util.map_tree(lambda x: x + 1.0, new_state), new_state

    loss_fn = fun_mc.make_surrogate_loss_fn(grad_fn)

    # Mutate the state to make sure we didn't capture anything.
    state = util.map_tree(lambda x: self._constant(x + 1.0), state)
    ret, extra, grads = fun_mc.call_potential_fn_with_grads(loss_fn, state)
    # The default is 0.
    self.assertAllClose(0.0, ret)
    # The gradients of the surrogate loss are state + 1.
    self.assertAllCloseNested(util.map_tree(lambda x: x + 1.0, state), grads)
    self.assertAllCloseNested(state, extra)

  def testSurrogateLossFnDecorator(self):
    @fun_mc.make_surrogate_loss_fn(loss_value=1.0)
    def loss_fn(_):
      return 3.0, 2.0

    ret, extra, grads = fun_mc.call_potential_fn_with_grads(loss_fn, 0.0)
    self.assertAllClose(1.0, ret)
    self.assertAllClose(2.0, extra)
    self.assertAllClose(3.0, grads)

  @parameterized.named_parameters(
      ('Probability', True),
      ('Loss', False),
  )
  def testReparameterizeFn(self, track_volume):
    def potential_fn(x, y):
      return -(x**2) + -(y**2), ()

    def transport_map_fn(x, y):
      return [2 * x, 3 * y], ((), jnp.log(2.0) + jnp.log(3.0))

    def inverse_map_fn(x, y):
      return [x / 2, y / 3], ((), -jnp.log(2.0) - jnp.log(3.0))

    transport_map_fn.inverse = inverse_map_fn

    (transformed_potential_fn, transformed_init_state) = (
        fun_mc.reparameterize_potential_fn(
            potential_fn,
            transport_map_fn,
            [self._constant(2.0), self._constant(3.0)],
            track_volume=track_volume,
        )
    )

    self.assertIsInstance(transformed_init_state, list)
    self.assertAllClose([1.0, 1.0], transformed_init_state)
    transformed_potential, (orig_space, _, _) = transformed_potential_fn(
        1.0, 1.0
    )
    potential = potential_fn(2.0, 3.0)[0]
    if track_volume:
      potential += jnp.log(2.0) + jnp.log(3.0)

    self.assertAllClose([2.0, 3.0], orig_space)
    self.assertAllClose(potential, transformed_potential)

  def testPersistentMH(self):
    def target_log_prob_fn(x):
      return -(x**2) / 2, ()

    def kernel(pmh_state, rwm_state, seed):
      seed, rwm_seed = util.split_seed(seed, 2)
      # RWM is used to create a valid sequence of energy changes. The
      # correctness of the algorithm relies on the energy changes to be
      # symmetric about 0.
      rwm_state, rwm_extra = fun_mc.random_walk_metropolis_step(
          rwm_state,
          target_log_prob_fn=target_log_prob_fn,
          proposal_fn=lambda state, seed: fun_mc.gaussian_proposal(  # pylint: disable=g-long-lambda
              state, seed=seed
          ),
          seed=rwm_seed,
      )
      pmh_state, pmh_extra = fun_mc.persistent_metropolis_hastings_step(
          pmh_state,
          # Use dummy states for testing.
          current_state=self._constant(0.0),
          proposed_state=self._constant(1.0),
          # Coprime with 1000 below.
          drift=0.127,
          energy_change=-rwm_extra.log_accept_ratio,
      )
      return (pmh_state, rwm_state, seed), (
          pmh_extra.is_accepted,
          pmh_extra.accepted_state,
          rwm_extra.is_accepted,
      )

    _, (pmh_is_accepted, pmh_accepted_state, rwm_is_accepted) = fun_mc.trace(
        (
            fun_mc.persistent_metropolis_hastings_init([], self._dtype),
            fun_mc.random_walk_metropolis_init(
                self._constant(0.0), target_log_prob_fn
            ),
            self._make_seed(_test_seed()),
        ),
        kernel,
        1000,
    )

    pmh_is_accepted = jnp.array(pmh_is_accepted, self._dtype)
    rwm_is_accepted = jnp.array(rwm_is_accepted, self._dtype)
    self.assertAllClose(
        jnp.mean(rwm_is_accepted), jnp.mean(pmh_is_accepted), atol=0.05
    )
    self.assertAllClose(pmh_is_accepted, pmh_accepted_state)

  @parameterized.named_parameters(
      ('ScalarLarge', -1.0, lambda x: x**2, True, -1.0),
      ('ScalarSmall', -0.1, lambda x: x**2, True, -0.2),
      (
          'Vector',
          np.array([-3, -4.0]),
          lambda x: jnp.sum(x**2),
          True,
          np.array([-3.0 / 5.0, -4.0 / 5.0]),
      ),
      (
          'List',
          [
              -3,
              -4.0,
          ],
          lambda x: x[0] ** 2 + x[1] ** 2,
          True,
          [
              -3.0 / 5.0,
              -4.0 / 5.0,
          ],
      ),
      (
          'Dict',
          {
              'a': -3,
              'b': -4.0,
          },
          lambda x: x['a'] ** 2 + x['b'] ** 2,
          True,
          {
              'a': -3.0 / 5.0,
              'b': -4.0 / 5.0,
          },
      ),
      (
          'NaNToZero',
          [
              -3,
              np.float32('nan'),
          ],
          lambda x: x[0] ** 2 + x[1] ** 2,
          True,
          [
              -1.0,
              0.0,
          ],
      ),
      (
          'NaNKept',
          [-3, np.float32('nan')],
          lambda x: x[0] ** 2 + x[1] ** 2,
          False,
          [np.float32('nan'), np.float32('nan')],
      ),
  )
  def testClipGradients(self, x, fn, zero_out_nan, expected_grad):
    x = util.map_tree(self._constant, x)
    max_global_norm = self._constant(1.0)
    expected_grad = util.map_tree(self._constant, expected_grad)

    def eval_fn(x):
      x = fun_mc.clip_grads(
          x, max_global_norm=max_global_norm, zero_out_nan=zero_out_nan
      )
      return fn(x), ()

    value, _, (grad,) = fun_mc.call_potential_fn_with_grads(eval_fn, (x,))

    self.assertAllCloseNested(value, fn(x))
    self.assertAllCloseNested(expected_grad, grad)


@test_util.multi_backend_test(globals(), 'fun_mc_test')
class FunMCTest32(FunMCTest):

  @property
  def _dtype(self):
    return jnp.float32


@test_util.multi_backend_test(globals(), 'fun_mc_test')
class FunMCTest64(FunMCTest):

  @property
  def _dtype(self):
    return jnp.float64


del FunMCTest

if __name__ == '__main__':
  tfp_test_util.main()
