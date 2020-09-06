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
"""Utilities for parameter constraints for surrogate posterior construction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb

JAX_MODE = False

# TODO(kateslin): Delete this file and refactor code for bijector constraint
#  lookup when this when cl/325157147 gets checked in.


def constrain_between_eps_and_one_minus_eps(eps=1e-6):
  return lambda x: eps + (1. - 2. * eps) * tf.sigmoid(x)


def fix_lkj(d):
  return dict(d, concentration=d['concentration'] + 1., dimension=3)


def fix_spherical_uniform(d):
  return dict(d, dimension=5, batch_shape=[])


def constraint_for(dist=None, param=None):
  """Get bijector constraint for a given distribution's parameter."""

  constraints = {
      'atol':
          tfb.Softplus(),
      'rtol':
          tfb.Softplus(),
      'concentration':
          tfb.Softplus(),
      'GeneralizedPareto.concentration':  # Permits +ve and -ve concentrations.
          lambda x: tf.math.tanh(x) * 0.24,
      'concentration0':
          tfb.Softplus(),
      'concentration1':
          tfb.Softplus(),
      'df':
          tfb.Softplus(),
      'InverseGaussian.loc':
          tfb.Softplus(),
      'JohnsonSU.tailweight':
          tfb.Softplus(),
      'PowerSpherical.mean_direction':
          lambda x: tf.math.l2_normalize(tf.math.sigmoid(x) + 1e-6, -1),
      'ContinuousBernoulli.probs':
          tfb.Sigmoid(),
      'Geometric.logits':  # TODO(b/128410109): re-enable down to -50
          # Capping at 15. so that probability is less than 1, and entropy is
          # defined. b/147394924
          lambda x: tf.minimum(tf.maximum(x, -16.), 15.
                              ),  # works around the bug
      'Geometric.probs':
          constrain_between_eps_and_one_minus_eps(),
      'Binomial.probs':
          tfb.Sigmoid(),
      'NegativeBinomial.probs':
          tfb.Sigmoid(),
      'Bernoulli.probs':
          tfb.Sigmoid(),
      'PlackettLuce.scores':
          tfb.Softplus(),
      'ProbitBernoulli.probs':
          tfb.Sigmoid(),
      'RelaxedBernoulli.probs':
          tfb.Sigmoid(),
      'cutpoints':  # Permit values that aren't too large
          lambda x: tfb.Ordered().inverse(10. * tf.math.tanh(x)),
      'log_rate':
          lambda x: tf.maximum(x, -16.),
      'mixing_concentration':
          tfb.Softplus(),
      'mixing_rate':
          tfb.Softplus(),
      'rate':
          tfb.Softplus(),
      'scale':
          tfb.Softplus(),
      'scale_diag':
          tfb.Softplus(),
      'scale_identity_multiplier':
          tfb.Softplus(),
      'tailweight':
          tfb.Softplus(),
      'temperature':
          tfb.Softplus(),
      'total_count':
          lambda x: tf.floor(tfb.Sigmoid()(x / 100.) * 100.) + 1.,
      'Bernoulli':
          lambda d: dict(d, dtype=tf.float32),
      'CholeskyLKJ':
          fix_lkj,
      'LKJ':
          fix_lkj,
      'Zipf':
          lambda d: dict(d, dtype=tf.float32),
      'GeneralizedNormal.power':
          tfb.Softplus(),
  }

  if param is not None:
    return constraints.get('{}.{}'.format(dist, param),
                           constraints.get(param, tfb.Identity()))
  return constraints.get(dist, tfb.Identity())
