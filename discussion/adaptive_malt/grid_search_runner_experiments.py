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
"""Defines experiments for the grid search runner."""

import os
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


def get_experiment(
    name: str, output_dir: str,
    inits_dir: str) -> Tuple[Callable[..., str], List[Dict[str, Any]]]:
  """Returns the a list of hyperparameters to run a grid search."""
  # pylint: disable=function-redefined
  all_job_args = []
  args_to_hparams = None

  if False:  # pylint: disable=using-constant-test
    # Just to make the syntax more uniform below...
    pass
  elif name == 'grid_search.1.6.malt.lower_accept':
    # Like 1.5, but even lower target accept for MALT.

    def args_to_hparams(target, mean_trajectory_length_vals, damping,
                        grid_index):
      return {
          'experiment.output_dir': os.path.join(output_dir, target),
          'experiment.grid_index': grid_index,
          'experiment.mean_trajectory_length_vals': mean_trajectory_length_vals,
          'run_grid_element.damping': damping,
          'run_grid_element.target_name': target,
          'run_grid_element.num_replicas': 10,
          'run_grid_element.method': 'malt',
          'run_grid_element.inits_dir': inits_dir,
          'run_grid_element.target_accept_prob': 0.5,
      }

    for target, mean_trajectory_length_vals, damping_vals in [
        (
            'test_gaussian_2',
            np.linspace(0.1, 4., 25),
            np.linspace(0., 2., 25),
        ),
        (
            'test_gaussian_1',
            np.linspace(0.1, 4., 25),
            np.linspace(0., 2., 25),
        ),
        (
            'radon_indiana',
            np.linspace(0.1, 10., 25),
            np.linspace(0., 1., 25),
        ),
        (
            'german_credit_numeric_sparse_logistic_regression',
            np.linspace(0.1, 12., 25),
            np.linspace(0., 1., 25),
        ),
        (
            'item_response_theory',
            np.linspace(0.1, 6., 25),
            np.linspace(0., 2., 25),
        ),
        (
            'german_credit_numeric_logistic_regression',
            np.linspace(0.05, 2., 25),
            np.linspace(0., 5., 25),
        ),
    ]:
      for i, damping in enumerate(damping_vals):
        all_job_args.append({
            'target': target,
            'grid_index': i,
            'mean_trajectory_length_vals': list(mean_trajectory_length_vals),
            'damping': damping,
        })
  elif name == 'grid_search.1.5.malt.low_accept':
    # Like 1.2, but lower target accept for MALT.

    def args_to_hparams(target, mean_trajectory_length_vals, damping,
                        grid_index):
      return {
          'experiment.output_dir': os.path.join(output_dir, target),
          'experiment.grid_index': grid_index,
          'experiment.mean_trajectory_length_vals': mean_trajectory_length_vals,
          'run_grid_element.damping': damping,
          'run_grid_element.target_name': target,
          'run_grid_element.num_replicas': 10,
          'run_grid_element.method': 'malt',
          'run_grid_element.inits_dir': inits_dir,
          'run_grid_element.target_accept_prob': 0.6,
      }

    for target, mean_trajectory_length_vals, damping_vals in [
        (
            'test_gaussian_2',
            np.linspace(0.1, 4., 25),
            np.linspace(0., 2., 25),
        ),
        (
            'test_gaussian_1',
            np.linspace(0.1, 4., 25),
            np.linspace(0., 2., 25),
        ),
        (
            'radon_indiana',
            np.linspace(0.1, 10., 25),
            np.linspace(0., 1., 25),
        ),
        (
            'german_credit_numeric_sparse_logistic_regression',
            np.linspace(0.1, 12., 25),
            np.linspace(0., 1., 25),
        ),
        (
            'item_response_theory',
            np.linspace(0.1, 6., 25),
            np.linspace(0., 2., 25),
        ),
        (
            'german_credit_numeric_logistic_regression',
            np.linspace(0.05, 2., 25),
            np.linspace(0., 5., 25),
        ),
    ]:
      for i, damping in enumerate(damping_vals):
        all_job_args.append({
            'target': target,
            'grid_index': i,
            'mean_trajectory_length_vals': list(mean_trajectory_length_vals),
            'damping': damping,
        })
  elif name == 'grid_search.1.4_exp_hmc':
    # HMC-only, with exponential jittering.

    def args_to_hparams(target, mean_trajectory_length_vals,
                        grid_index):
      return {
          'experiment.output_dir': os.path.join(output_dir, target),
          'experiment.grid_index': grid_index,
          'experiment.mean_trajectory_length_vals': mean_trajectory_length_vals,
          'run_grid_element.damping': 1.,
          'run_grid_element.target_name': target,
          'run_grid_element.num_replicas': 10,
          'run_grid_element.method': 'hmc',
          'run_grid_element.inits_dir': inits_dir,
          'run_grid_element.jitter_style': 'exponential',
      }

    for target, mean_trajectory_length_vals in [
        (
            'test_gaussian_2',
            np.linspace(0.1, 4., 25),
        ),
        (
            'test_gaussian_1',
            np.linspace(0.1, 4., 25),
        ),
        (
            'radon_indiana',
            np.linspace(0.1, 10., 25),
        ),
        (
            'german_credit_numeric_sparse_logistic_regression',
            np.linspace(0.1, 12., 25),
        ),
        (
            'item_response_theory',
            np.linspace(0.1, 6., 25),
        ),
        (
            'german_credit_numeric_logistic_regression',
            np.linspace(0.05, 2., 25),
        ),
    ]:
      all_job_args.append({
          'target': target,
          'grid_index': 0,
          'mean_trajectory_length_vals': list(mean_trajectory_length_vals),
      })
  elif name == 'grid_search.1.3_single_chain':

    def args_to_hparams(target, method, mean_trajectory_length_vals, damping,
                        grid_index):
      return {
          'experiment.output_dir': os.path.join(output_dir, target, method),
          'experiment.grid_index': grid_index,
          'experiment.mean_trajectory_length_vals': mean_trajectory_length_vals,
          'run_grid_element.damping': damping,
          'run_grid_element.target_name': target,
          'run_grid_element.num_replicas': 10,
          'run_grid_element.method': method,
          'run_grid_element.inits_dir': inits_dir,
          'run_grid_element.num_chains': 1,
          'run_grid_element.num_adaptation_steps': 10000,
          'run_grid_element.num_results': 10000,
      }

    for target, mean_trajectory_length_vals, damping_vals in [
        (
            'test_gaussian_1',
            np.linspace(0.1, 4., 25),
            np.linspace(0., 2., 25),
        ),
        (
            'radon_indiana',
            np.linspace(0.1, 10., 25),
            np.linspace(0., 1., 25),
        ),
        (
            'german_credit_numeric_sparse_logistic_regression',
            np.linspace(0.1, 12., 25),
            np.linspace(0., 1., 25),
        ),
    ]:
      for method, cur_damping_vals in zip(
          ['hmc', 'malt'], [[0.], damping_vals]):
        for i, damping in enumerate(cur_damping_vals):
          all_job_args.append({
              'target': target,
              'method': method,
              'grid_index': i,
              'mean_trajectory_length_vals': list(mean_trajectory_length_vals),
              'damping': damping,
          })
  elif name == 'grid_search.1.2':
    # Everything fixed.

    def args_to_hparams(target, method, mean_trajectory_length_vals, damping,
                        grid_index):
      return {
          'experiment.output_dir': os.path.join(output_dir, target, method),
          'experiment.grid_index': grid_index,
          'experiment.mean_trajectory_length_vals': mean_trajectory_length_vals,
          'run_grid_element.damping': damping,
          'run_grid_element.target_name': target,
          'run_grid_element.num_replicas': 10,
          'run_grid_element.method': method,
          'run_grid_element.inits_dir': inits_dir,
      }

    for target, mean_trajectory_length_vals, damping_vals in [
        (
            'test_gaussian_2',
            np.linspace(0.1, 4., 25),
            np.linspace(0., 2., 25),
        ),
        (
            'test_gaussian_1',
            np.linspace(0.1, 4., 25),
            np.linspace(0., 2., 25),
        ),
        (
            'radon_indiana',
            np.linspace(0.1, 10., 25),
            np.linspace(0., 1., 25),
        ),
        (
            'german_credit_numeric_sparse_logistic_regression',
            np.linspace(0.1, 12., 25),
            np.linspace(0., 1., 25),
        ),
        (
            'item_response_theory',
            np.linspace(0.1, 6., 25),
            np.linspace(0., 2., 25),
        ),
        (
            'german_credit_numeric_logistic_regression',
            np.linspace(0.05, 2., 25),
            np.linspace(0., 5., 25),
        ),
    ]:
      for method, cur_damping_vals in zip(
          ['hmc', 'malt'], [[0.], damping_vals]):
        for i, damping in enumerate(cur_damping_vals):
          all_job_args.append({
              'target': target,
              'method': method,
              'grid_index': i,
              'mean_trajectory_length_vals': list(mean_trajectory_length_vals),
              'damping': damping,
          })
  elif name == 'grid_search.1.1':
    # Broken.

    def args_to_hparams(target, method, mean_trajectory_length, damping,
                        grid_index):
      return {
          'experiment.output_dir': os.path.join(output_dir, target, method),
          'experiment.grid_index': grid_index,
          'run_grid_element.mean_trajectory_length': mean_trajectory_length,
          'run_grid_element.damping': damping,
          'run_grid_element.target_name': target,
          'run_grid_element.num_replicas': 10,
          'run_grid_element.method': method,
          'run_grid_element.inits_dir': inits_dir,
      }

    for target, mean_trajectory_length_vals, damping_vals in [
        (
            'test_gaussian_1',
            np.linspace(0.1, 4., 25),
            np.linspace(0., 1., 25),
        ),
        (
            'radon_indiana',
            np.linspace(0.1, 10., 25),
            np.linspace(0., 1., 25),
        ),
        (
            'german_credit_numeric_sparse_logistic_regression',
            np.linspace(0.1, 12., 25),
            np.linspace(0., 1., 25),
        ),
    ]:
      for method, cur_damping_vals in zip(
          ['hmc', 'malt'], [[0.], damping_vals]):
        for i, damping in enumerate(cur_damping_vals):
          for j, mean_trajectory_length in enumerate(
              mean_trajectory_length_vals):
            all_job_args.append({
                'target': target,
                'method': method,
                'grid_index': [i, j],
                'mean_trajectory_length': mean_trajectory_length,
                'damping': damping,
            })
  elif name == 'grid_search.1.0':
    # Broken.

    def args_to_hparams(target, method, mean_trajectory_length, damping,
                        grid_index):
      return {
          'experiment.output_dir': os.path.join(output_dir, target, method),
          'experiment.grid_index': grid_index,
          'run_grid_element.mean_trajectory_length': mean_trajectory_length,
          'run_grid_element.damping': damping,
          'run_grid_element.target_name': target,
          'run_grid_element.num_replicas': 10,
          'run_grid_element.method': method,
          'run_grid_element.inits_dir': inits_dir,
      }

    for target, mean_trajectory_length_vals in [
        (
            'test_gaussian_1',
            np.linspace(0.1, 4., 10),
        ),
        (
            'radon_indiana',
            np.linspace(0.1, 10., 10),
        ),
        (
            'german_credit_numeric_sparse_logistic_regression',
            np.linspace(0.1, 12., 10),
        ),
    ]:
      for method, damping_vals in zip(
          ['hmc', 'malt'], [[0.], np.cos(np.linspace(0., np.pi / 2, 10))]):
        for i, damping in enumerate(damping_vals):
          for j, mean_trajectory_length in enumerate(
              mean_trajectory_length_vals):
            all_job_args.append({
                'target': target,
                'method': method,
                'grid_index': [i, j],
                'mean_trajectory_length': mean_trajectory_length,
                'damping': damping,
            })
  else:
    raise ValueError(f'Unknown experiment: {name}')

  def wrapped_args_to_hparams(**job_args):
    return ', '.join(
        f'{k}: {v}' for k, v in args_to_hparams(**job_args).items())

  return wrapped_args_to_hparams, all_job_args
