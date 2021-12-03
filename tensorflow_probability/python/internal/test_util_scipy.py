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
"""Utilities for testing TFP code that depend on scipy."""

import numpy as np
import scipy.optimize as optimize
import scipy.stats as stats

__all__ = [
    'binomial_confidence_interval',
]


# TODO(axch): Test this independently of its use in
# distributions/internal/correlation_matrix_volumes_lib
def binomial_confidence_interval(successes, trials, error_rate):
  """Computes a confidence interval on the true p of a binomial.

  Assumes:
  - The given `successes` count outcomes of an iid Bernoulli trial
    with unknown probability p, that was repeated `trials` times.

  Guarantees:
  - The probability (over the randomness of drawing the given sample)
    that the true p is outside the returned interval is no more than
    the given `error_rate`.

  Args:
    successes: Python or numpy `int` number of successes.
    trials: Python or numpy `int` number of trials.
    error_rate: Python `float` admissible rate of mistakes.

  Returns:
    low_p: Lower bound of confidence interval.
    high_p: Upper bound of confidence interval.

  Raises:
    ValueError: If scipy is not available.

  """
  def p_small_enough(p):
    log_prob = stats.binom.logcdf(successes, trials, p)
    return log_prob - np.log(error_rate / 2.)
  def p_big_enough(p):
    log_prob = stats.binom.logsf(successes, trials, p)
    return log_prob - np.log(error_rate / 2.)
  if successes < trials:
    high_p = optimize.brentq(
        p_small_enough, successes / float(trials), 1., rtol=1e-9)
  else:
    high_p = 1.
  if successes > 0:
    low_p = optimize.brentq(
        p_big_enough, 0., successes / float(trials), rtol=1e-9)
  else:
    low_p = 0.
  return low_p, high_p
