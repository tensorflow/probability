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
"""Contains utilities for the `plate` transformation.

A plate is a term in graphical models that is used to designate independent
random variables. In Oryx, `plate` is a transformation that converts a program
into one that produces independent samples. Ordinarily, this can be done with
`jax.vmap`, where we could split several random keys and map a program over
them. Unlike `jax.vmap`, `plate` operates using named axes. A `plate`-ed
program will specialize the random seed to the particular index of the axis
being mapped over. Taking the `log_prob` of a `plate` program will reduce over
the named axis. In design, `plate` resembles the `Sharded` meta-distribution
from TensorFlow Probability.

In implementation, `plate` is an Oryx `HigherOrderPrimitive` (i.e. a JAX
`CallPrimitive` with a `log_prob` rule that reduces over a named axis at the
end.
"""
import functools

from jax import lax
from jax import random

from oryx.core import primitive
from oryx.core.interpreters import log_prob
from oryx.core.interpreters import propagate


__all__ = [
    'make_plate',
]


plate_p = primitive.HigherOrderPrimitive('plate')


def plate_log_prob_rule(incells, outcells, *, plate, **params):
  incells, outcells, lp = propagate.call_rule(
      plate_p, incells, outcells, plate=plate, **params)
  return incells, outcells, lax.psum(lp, plate)


log_prob.log_prob_rules[plate_p] = plate_log_prob_rule
log_prob.log_prob_registry.add(plate_p)


def make_plate(f, *, name):
  """Wraps a probabilistic program in a plate with a named axis."""

  @functools.wraps(f)
  def plate_fun(key, *args, **kwargs):
    key = random.fold_in(key, lax.axis_index(name))
    return f(key, *args, **kwargs)

  def wrapped(key, *args, **kwargs):
    return primitive.call_bind(
        plate_p, plate=name)(plate_fun)(key, *args, **kwargs)

  return wrapped
