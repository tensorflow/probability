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
"""Tests for `MetropolisHastings` `TransitionKernel`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import warnings
# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.mcmc.internal.util import is_list_like

InnerKernelResultsWithoutCorrection = collections.namedtuple(

    'InnerKernelResultsWithoutCorrection',
    [
        'target_log_prob',        # For "next_state".
        'grads_target_log_prob',  # For "next_state".
        # We add a "bogus" field just to ensure that the automatic introspection
        # works as intended.
        'extraneous',
    ])


InnerKernelResultsWithCorrection = collections.namedtuple(
    'InnerKernelResultsWithCorrection',
    [
        'log_acceptance_correction',
        'target_log_prob',        # For "next_state".
        'grads_target_log_prob',  # For "next_state".
        # We add a "bogus" field just to ensure that the automatic introspection
        # works as intended.
        'extraneous',
    ])


class FakeTransitionKernel(tfp.mcmc.TransitionKernel):
  """Fake TransitionKernel for testing MetropolisHastings."""

  def __init__(self, is_calibrated, one_step_fn, bootstrap_fn):
    self._is_calibrated = is_calibrated
    self._one_step_fn = one_step_fn
    self._bootstrap_fn = bootstrap_fn
    self._call_count = collections.Counter()

  @property
  def call_count(self):
    return self._call_count

  @property
  def is_calibrated(self):
    self.call_count['is_calibrated'] += 1
    return self._is_calibrated

  def one_step(self, current_state, previous_kernel_results):
    self.call_count['one_step'] += 1
    return self._one_step_fn(current_state, previous_kernel_results)

  def bootstrap_results(self, init_state):
    self.call_count['bootstrap_results'] += 1
    return self._bootstrap_fn(init_state)


def make_one_step_fn(dtype):
  def one_step(current_state, previous_kernel_results):
    # Make next_state.
    if is_list_like(current_state):
      next_state = []
      for i, s in enumerate(current_state):
        next_state.append(tf.identity(s * dtype(i + 2),
                                      name='next_state'))
    else:
      next_state = tf.identity(2. * current_state,
                               name='next_state')
    # Make kernel_results.
    kernel_results = {}
    for fn in sorted(previous_kernel_results._fields):
      if fn == 'grads_target_log_prob':
        kernel_results['grads_target_log_prob'] = [
            tf.identity(0.5 * g, name='grad_target_log_prob')
            for g in previous_kernel_results.grads_target_log_prob]
      elif fn == 'extraneous':
        kernel_results[fn] = getattr(previous_kernel_results, fn, None)
      else:
        kernel_results[fn] = tf.identity(
            0.5 * getattr(previous_kernel_results, fn, None),
            name=fn)
    kernel_results = type(previous_kernel_results)(**kernel_results)
    # Done.
    return next_state, kernel_results

  return one_step


def make_bootstrap_results_fn(true_kernel_results):
  kernel_results_cls = type(true_kernel_results)
  def bootstrap_results(_):
    fake_kernel_results = {}
    for fn in sorted(kernel_results_cls._fields):
      if fn == 'grads_target_log_prob':
        fake_kernel_results['grads_target_log_prob'] = [
            tf.identity(g, name='grad_target_log_prob')
            for g in true_kernel_results.grads_target_log_prob]
      else:
        fake_kernel_results[fn] = tf.identity(
            getattr(true_kernel_results, fn, None),
            name=fn)
    fake_kernel_results = kernel_results_cls(**fake_kernel_results)
    return fake_kernel_results
  return bootstrap_results


@test_util.test_all_tf_execution_regimes
class MetropolisHastingsTest(test_util.TestCase):

  def setUp(self):
    self.dtype = np.float32

  def testCorrectlyWorksWithoutCorrection(self):
    current_state_ = [self.dtype([1, 2]),
                      self.dtype([3, 4])]
    current_state = [tf.convert_to_tensor(s) for s in current_state_]
    expected_inner_init_kernel_results = InnerKernelResultsWithoutCorrection(
        target_log_prob=self.dtype([
            +100.,
            -100.,
        ]),
        grads_target_log_prob=[self.dtype([1.25, 1.5]),
                               self.dtype([2.25, 2.5])],
        extraneous=self.dtype([1.75, 2.]))

    one_step_fn = make_one_step_fn(dtype=self.dtype)
    bootstrap_fn = make_bootstrap_results_fn(
        expected_inner_init_kernel_results)

    # Collect expected results.
    expected_init_inner_kernel_results = bootstrap_fn(current_state)
    _, expected_inner_kernel_results = one_step_fn(
        current_state, expected_init_inner_kernel_results)

    # Collect actual results.
    mh = tfp.mcmc.MetropolisHastings(
        FakeTransitionKernel(
            is_calibrated=False,
            one_step_fn=one_step_fn,
            bootstrap_fn=bootstrap_fn),
        seed=1)
    init_kernel_results = mh.bootstrap_results(current_state)
    next_state, kernel_results = mh.one_step(
        current_state, init_kernel_results)

    # Unmodified state is passed through unmodified.
    self.assertIs(kernel_results.accepted_results.extraneous,
                  init_kernel_results.accepted_results.extraneous)
    self.assertIs(kernel_results.proposed_results.extraneous,
                  init_kernel_results.accepted_results.extraneous)

    # Check correct types and call pattern.
    self.assertEqual(
        dict(is_calibrated=1,
             one_step=1,
             bootstrap_results=1),
        mh.inner_kernel.call_count)

    for kr in [init_kernel_results.accepted_results,
               init_kernel_results.proposed_results]:
      self.assertEqual(type(expected_init_inner_kernel_results), type(kr))
    for kr in [kernel_results.accepted_results,
               kernel_results.proposed_results]:
      self.assertEqual(type(expected_inner_kernel_results), type(kr))

    # Now check actual values.
    [
        expected_init_inner_kernel_results_,
        expected_inner_kernel_results_,
        init_kernel_results_,
        kernel_results_,
        next_state_,
    ] = self.evaluate([
        expected_init_inner_kernel_results,
        expected_inner_kernel_results,
        init_kernel_results,
        kernel_results,
        next_state,
    ])

    # Check that the bootstrapped kernel results are correctly initialized.
    for fn in expected_inner_init_kernel_results._fields:
      self.assertAllClose(
          getattr(expected_init_inner_kernel_results_, fn, np.nan),
          getattr(init_kernel_results_.accepted_results, fn, np.nan),
          atol=0.,
          rtol=1e-5)

    # Check that the proposal is correctly computed.
    self.assertAllClose([2 * current_state_[0],
                         3 * current_state_[1]],
                        kernel_results_.proposed_state,
                        atol=0., rtol=1e-5)
    for fn in expected_inner_kernel_results._fields:
      self.assertAllClose(
          getattr(expected_inner_kernel_results_, fn, np.nan),
          getattr(kernel_results_.proposed_results, fn, np.nan),
          atol=0.,
          rtol=1e-5)

    # Extremely high start prob means first will be rejected.
    # Extremely low start prob means second will be accepted.
    self.assertAllEqual([False, True],
                        kernel_results_.is_accepted)
    self.assertAllEqual([(0.5 * 100.) - (100.),
                         (0.5 * -100.) - (-100.)],
                        kernel_results_.log_accept_ratio)
    self.assertAllClose([self.dtype([1, 0 + 2]) * current_state_[0],
                         self.dtype([1, 1 + 2]) * current_state_[1]],
                        next_state_)

  def testCorrectlyWorksWithCorrection(self):
    current_state_ = [self.dtype([1, 2]),
                      self.dtype([3, 4])]
    current_state = [tf.convert_to_tensor(s) for s in current_state_]

    expected_inner_init_kernel_results = InnerKernelResultsWithCorrection(
        log_acceptance_correction=self.dtype([+300., -300.]),
        target_log_prob=self.dtype([100., -100.]),
        grads_target_log_prob=[self.dtype([1.25, 1.5]),
                               self.dtype([2.25, 2.5])],
        extraneous=self.dtype([1.75, 2.]))

    one_step_fn = make_one_step_fn(dtype=self.dtype)
    bootstrap_fn = make_bootstrap_results_fn(
        expected_inner_init_kernel_results)

    # Collect expected results.
    expected_init_inner_kernel_results = bootstrap_fn(current_state)
    _, expected_inner_kernel_results = one_step_fn(
        current_state, expected_init_inner_kernel_results)

    # Collect actual results.
    mh = tfp.mcmc.MetropolisHastings(
        FakeTransitionKernel(
            is_calibrated=False,
            one_step_fn=one_step_fn,
            bootstrap_fn=bootstrap_fn),
        seed=1)
    init_kernel_results = mh.bootstrap_results(current_state)
    next_state, kernel_results = mh.one_step(
        current_state, init_kernel_results)

    # Unmodified state is passed through unmodified.
    self.assertIs(kernel_results.accepted_results.extraneous,
                  init_kernel_results.accepted_results.extraneous)
    self.assertIs(kernel_results.proposed_results.extraneous,
                  init_kernel_results.accepted_results.extraneous)

    # Check correct types and call pattern.
    self.assertEqual(
        dict(is_calibrated=1,
             one_step=1,
             bootstrap_results=1),
        mh.inner_kernel.call_count)
    for kr in [init_kernel_results.accepted_results,
               init_kernel_results.proposed_results]:
      self.assertEqual(type(expected_init_inner_kernel_results), type(kr))
    for kr in [kernel_results.accepted_results,
               kernel_results.proposed_results]:
      self.assertEqual(type(expected_inner_kernel_results), type(kr))

    # Now check actual values.
    [
        expected_init_inner_kernel_results_,
        expected_inner_kernel_results_,
        init_kernel_results_,
        kernel_results_,
        next_state_,
    ] = self.evaluate([
        expected_init_inner_kernel_results,
        expected_inner_kernel_results,
        init_kernel_results,
        kernel_results,
        next_state,
    ])

    # Check that the bootstrapped kernel results are correctly initialized.
    for fn in expected_inner_init_kernel_results._fields:
      self.assertAllClose(
          getattr(expected_init_inner_kernel_results_, fn, np.nan),
          getattr(init_kernel_results_.accepted_results, fn, np.nan),
          atol=0.,
          rtol=1e-5)

    # Check that the proposal is correctly computed.
    self.assertAllClose([2 * current_state_[0],
                         3 * current_state_[1]],
                        kernel_results_.proposed_state,
                        atol=0., rtol=1e-5)
    for fn in expected_inner_kernel_results._fields:
      self.assertAllClose(
          getattr(expected_inner_kernel_results_, fn, np.nan),
          getattr(kernel_results_.proposed_results, fn, np.nan),
          atol=0.,
          rtol=1e-5)

    # First: Extremely high correction means proposed will be accepted, despite
    #        high prob initial state.
    # Second: Extremely low correction means proposed will be rejected, despite
    #         low prob initial state.
    self.assertAllEqual([True, False],
                        kernel_results_.is_accepted)
    self.assertAllEqual([(0.5 * 100.) - (100.) + (0.5 * 300.),
                         (0.5 * -100.) - (-100.) + (0.5 * -300.)],
                        kernel_results_.log_accept_ratio)
    self.assertAllClose([self.dtype([0 + 2, 1]) * current_state_[0],
                         self.dtype([1 + 2, 1]) * current_state_[1]],
                        next_state_)

  def testWarnings(self):
    current_state_ = [self.dtype([1, 2]),
                      self.dtype([3, 4])]
    current_state = [tf.convert_to_tensor(s) for s in current_state_]
    expected_inner_init_kernel_results = InnerKernelResultsWithoutCorrection(
        target_log_prob=self.dtype([100., -100.]),
        grads_target_log_prob=[
            self.dtype([1.25, 1.5]),
            self.dtype([2.25, 2.5]),
        ],
        extraneous=self.dtype([1.75, 2.]))

    one_step_fn = make_one_step_fn(dtype=self.dtype)
    bootstrap_fn = make_bootstrap_results_fn(
        expected_inner_init_kernel_results)

    with warnings.catch_warnings(record=True) as w:
      mh = tfp.mcmc.MetropolisHastings(
          FakeTransitionKernel(
              is_calibrated=True,
              one_step_fn=one_step_fn,
              bootstrap_fn=bootstrap_fn),
          seed=1)
      init_kernel_results = mh.bootstrap_results(current_state)
      _, _ = mh.one_step(current_state, init_kernel_results)
    w = sorted(w, key=lambda w: str(w.message))
    self.assertRegexpMatches(
        str(w[0].message),
        r'`TransitionKernel` is already calibrated')
    self.assertRegexpMatches(
        str(w[1].message),
        r'`TransitionKernel` does not have a `log_acceptance_correction`')


if __name__ == '__main__':
  tf.test.main()
