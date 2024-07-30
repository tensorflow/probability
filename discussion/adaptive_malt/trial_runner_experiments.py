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
"""Defines experiments for the trial runner."""

import os
from typing import Any, Callable, Dict, List, Tuple


def get_experiment(
    name: str, output_dir: str, inits_dir: str,
    ground_truth_dir: str) -> Tuple[Callable[..., str], List[Dict[str, Any]]]:
  """Returns the a list of hyperparameters to run a trial."""
  # pylint: disable=function-redefined
  all_job_args = []
  args_to_hparams = None

  if False:  # pylint: disable=using-constant-test
    # Just to make the syntax more uniform below...
    pass
  elif name == 'trial.1.4':
    # No learning rate decay, NUTS also traces mean num leapfrog steps.
    # 1.3 had a broken ESS computation

    def args_to_hparams(target, method, adapt_power, jitter_style):
      if method in ['hmc', 'malt']:
        ap_suffix = '.ap' if adapt_power else '.nap'
      else:
        ap_suffix = ''
      if jitter_style == 'halton':
        jitter_suffix = ''
      else:
        jitter_suffix = '.' + jitter_style
      save_warmup = target not in ['stochastic_volatility']
      return {
          'experiment.output_dir':
              os.path.join(output_dir, method + ap_suffix + jitter_suffix,
                           target),
          'run_trial.target_name':
              target,
          'run_trial.num_replicas':
              20,
          'run_trial.method':
              method,
          'run_trial.inits_dir':
              os.path.join(inits_dir, '1.8_vi'),
          'run_trial.ground_truth_dir':
              ground_truth_dir,
          'run_trial.adapt_normalization_power':
              adapt_power,
          'run_trial.num_chains':
              128,
          'run_trial.num_adaptation_steps':
              5000,
          'run_trial.num_results':
              2000 if save_warmup else 1000,
          'run_trial.step_size_adaptation_rate_decay':
              'none',
          'run_trial.trajectory_length_adaptation_rate_decay':
              'none',
          'run_trial.save_warmup':
              save_warmup,
          'run_trial.jitter_style':
              jitter_style,
      }

    for [target] in [
        ('radon_indiana',),
        ('german_credit_numeric_sparse_logistic_regression',),
        ('item_response_theory',),
        ('german_credit_numeric_logistic_regression',),
        ('brownian_motion',),
        ('banana',),
        ('stochastic_volatility',),
    ]:
      for method, adapt_power_vals, jitter_style_vals in [
          ('hmc', (True, False), (
              'halton',
              'halton_exponential',
          )),
          ('malt', (True, False), ('halton',)),
          ('meads', (True,), ('halton',)),
          ('nuts', (True,), ('halton',)),
      ]:
        for adapt_power in adapt_power_vals:
          for jitter_style in jitter_style_vals:
            all_job_args.append({
                'target': target,
                'method': method,
                'adapt_power': adapt_power,
                'jitter_style': jitter_style,
            })
  elif name == 'trial.1.2':

    def args_to_hparams(target, method, adapt_power):
      if method in ['hmc', 'malt']:
        ap_suffix = '.ap' if adapt_power else '.nap'
      else:
        ap_suffix = ''
      return {
          'experiment.output_dir':
              os.path.join(output_dir, method + ap_suffix, target),
          'run_trial.target_name':
              target,
          'run_trial.num_replicas':
              20,
          'run_trial.method':
              method,
          'run_trial.inits_dir':
              os.path.join(inits_dir, '1.8_vi'),
          'run_trial.ground_truth_dir':
              ground_truth_dir,
          'run_trial.adapt_normalization_power':
              adapt_power,
          'run_trial.num_chains':
              128,
          'run_trial.num_adaptation_steps':
              5000,
          'run_trial.num_results':
              2000,
      }

    for [target] in [
        ('test_gaussian_2',),
        ('test_gaussian_1',),
        ('radon_indiana',),
        ('german_credit_numeric_sparse_logistic_regression',),
        ('item_response_theory',),
        ('german_credit_numeric_logistic_regression',),
        ('brownian_motion',),
    ]:
      for method, adapt_power_vals in [
          ('hmc', (True, False)),
          ('malt', (True, False)),
          ('meads', (True,)),
          ('nuts', (True,)),
      ]:
        for adapt_power in adapt_power_vals:
          all_job_args.append({
              'target': target,
              'method': method,
              'adapt_power': adapt_power,
          })
  else:
    raise ValueError(f'Unknown experiment: {name}')

  def wrapped_args_to_hparams(**job_args):
    return ', '.join(
        f'{k}: {v}' for k, v in args_to_hparams(**job_args).items())

  return wrapped_args_to_hparams, all_job_args
