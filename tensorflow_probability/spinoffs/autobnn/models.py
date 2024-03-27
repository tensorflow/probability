# Copyright 2023 The TensorFlow Probability Authors.
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
"""BNF models.

The "combo" model is a simple sum of linear and periodic components. The sum of
products is the smallest example of a sum of two products over two leaves each,
where each leaf is a continuous relaxiation (using WeightedSum) of periodic and
linear components.
"""
import functools
from typing import Sequence, Union
import jax.numpy as jnp
from tensorflow_probability.spinoffs.autobnn import bnn
from tensorflow_probability.spinoffs.autobnn import bnn_tree
from tensorflow_probability.spinoffs.autobnn import kernels
from tensorflow_probability.spinoffs.autobnn import likelihoods
from tensorflow_probability.spinoffs.autobnn import operators


Array = jnp.ndarray


def make_sum_of_operators_of_relaxed_leaves(
    time_series_xs: Array,
    width: int = 5,
    periods: Sequence[float] = (0.1,),
    use_mul: bool = True,
    num_outputs: int = 1,
) -> bnn.BNN:
  """Returns BNN model consisting of a sum of products or changeponts of leaves.

  Each leaf is a continuous relaxation over base kernels.

  Args:
    time_series_xs: The x-values of the training data.
    width: Width of the leaf BNNs.
    periods: Periods for the PeriodicBNN kernel.
    use_mul: If true, use Multiply as the depth 1 operator.  If false, use
      a LearnableChangepoint instead.
    num_outputs: Number of outputs on the BNN.
  """
  del num_outputs
  def _make_continuous_relaxation(
      width: int,
      periods: Sequence[float],
      include_eq_and_poly: bool) -> bnn.BNN:
    leaves = [kernels.PeriodicBNN(
        width=width, period=p, going_to_be_multiplied=use_mul) for p in periods]
    leaves.append(kernels.LinearBNN(
        width=width, going_to_be_multiplied=use_mul))
    if include_eq_and_poly:
      leaves.extend([
          kernels.ExponentiatedQuadraticBNN(
              width=width, going_to_be_multiplied=use_mul),
          kernels.PolynomialBNN(width=width, going_to_be_multiplied=use_mul),
          kernels.IdentityBNN(width=width, going_to_be_multiplied=use_mul),
      ])
    return operators.WeightedSum(bnns=tuple(leaves), num_outputs=1,
                                 going_to_be_multiplied=use_mul)

  leaf1 = _make_continuous_relaxation(width, periods, include_eq_and_poly=False)
  leaf2 = _make_continuous_relaxation(width, periods, include_eq_and_poly=False)

  if use_mul:
    op = operators.Multiply
  else:
    op = functools.partial(operators.LearnableChangePoint,
                           time_series_xs=time_series_xs)

  bnn1 = op(bnns=(leaf1, leaf2))

  leaf3 = _make_continuous_relaxation(width, periods, include_eq_and_poly=True)
  leaf4 = _make_continuous_relaxation(width, periods, include_eq_and_poly=True)
  bnn2 = op(bnns=(leaf3, leaf4))

  net = operators.Add(bnns=(bnn1, bnn2))
  return net


def make_sum_of_products(
    time_series_xs: Array,
    width: int = 5,
    periods: Sequence[float] = (0.1,),
    num_outputs: int = 1,
) -> bnn.BNN:
  return make_sum_of_operators_of_relaxed_leaves(
      time_series_xs, width, periods, use_mul=True, num_outputs=num_outputs)


def make_sum_of_changepoints(
    time_series_xs: Array,
    width: int = 5,
    periods: Sequence[float] = (0.1,),
    num_outputs: int = 1,
) -> bnn.BNN:
  return make_sum_of_operators_of_relaxed_leaves(
      time_series_xs, width, periods, use_mul=False, num_outputs=num_outputs)


def make_linear_plus_periodic(
    time_series_xs: Array,
    width: int = 5,
    periods: Sequence[float] = (0.1,),
    num_outputs: int = 1,
) -> bnn.BNN:
  """Returns Combo model, consisting of linear and periodic leafs.

  Args:
    time_series_xs: The x-values of the training data.
    width: Width of the leaf BNNs.
    periods: Periods for the PeriodicBNN kernel.
    num_outputs: Number of outputs on the BNN.
  """
  del num_outputs
  del time_series_xs
  leaves = [kernels.PeriodicBNN(width=width, period=p) for p in periods]
  leaves.append(kernels.LinearBNN(width=width))
  return operators.Add(bnns=tuple(leaves))


def make_sum_of_stumps(
    time_series_xs: Array,
    width: int = 5,
    periods: Sequence[float] = (0.1,),
    num_outputs: int = 1,
) -> bnn.BNN:
  """Return a sum of depth 0 trees."""
  stumps = bnn_tree.list_of_all(time_series_xs, 0, width, periods=periods)

  return operators.WeightedSum(bnns=tuple(stumps), num_outputs=num_outputs)


def make_sum_of_stumps_and_products(
    time_series_xs: Array,
    width: int = 5,
    periods: Sequence[float] = (0.1,),
    num_outputs: int = 1,
) -> bnn.BNN:
  """Return a sum of depth 0 and depth 1 product-only trees."""
  stumps = bnn_tree.list_of_all(time_series_xs, 0, width, periods=periods)
  products = bnn_tree.list_of_all(
      time_series_xs,
      1,
      width,
      periods=periods,
      include_sums=False,
      include_changepoints=False,
  )

  return operators.WeightedSum(
      bnns=tuple(stumps + products), num_outputs=num_outputs)


def make_sum_of_shallow(
    time_series_xs: Array,
    width: int = 5,
    periods: Sequence[float] = (0.1,),
    num_outputs: int = 1,
) -> bnn.BNN:
  """Return a sum of depth 0 and 1 trees."""
  stumps = bnn_tree.list_of_all(time_series_xs, 0, width, periods=periods)
  depth1 = bnn_tree.list_of_all(
      time_series_xs, 1, width, periods=periods, include_sums=False
  )

  return operators.WeightedSum(
      bnns=tuple(stumps + depth1), num_outputs=num_outputs)


def make_sum_of_safe_shallow(
    time_series_xs: Array,
    width: int = 5,
    periods: Sequence[float] = (0.1,),
    num_outputs: int = 1,
) -> bnn.BNN:
  """Return a sum of depth 0 and 1 trees, but not unsafe products."""
  stumps = bnn_tree.list_of_all(time_series_xs, 0, width, periods=periods)
  depth1 = bnn_tree.list_of_all(
      time_series_xs,
      1,
      width,
      periods=periods,
      include_sums=False,
      only_safe_products=True,
  )

  return operators.WeightedSum(
      bnns=tuple(stumps + depth1), num_outputs=num_outputs)


def make_changepoint_of_safe_products(
    time_series_xs: Array,
    width: int = 5,
    periods: Sequence[float] = (0.1,),
    num_outputs: int = 1,
) -> bnn.BNN:
  """Return a changepoint over two Multiply(Linear, WeightedSum(kernels))'s."""
  # By varying the weights inside the WeightedSum (and by relying on the
  # identity Changepoint(A, A) = A), this model can express
  # * all base kernels,
  # * all "safe" multiplies over two base kernels (i.e., one of the terms
  #   has a very low effective parameter count to avoid overfitting noise), and
  # * all single changepoints over two of the above.

  all_kernels = [
      kernels.PeriodicBNN(width=width, period=p, going_to_be_multiplied=True)
      for p in periods
  ]
  all_kernels.extend(
      [
          k(width=width, going_to_be_multiplied=True)
          for k in [
              kernels.ExponentiatedQuadraticBNN,
              kernels.MaternBNN,
              kernels.LinearBNN,
              kernels.QuadraticBNN,
          ]
      ]
  )

  safe_product = operators.Multiply(
      bnns=(
          operators.WeightedSum(
              num_outputs=num_outputs,
              bnns=(
                  kernels.IdentityBNN(width=width, going_to_be_multiplied=True),
                  kernels.LinearBNN(width=width, going_to_be_multiplied=True),
                  kernels.QuadraticBNN(
                      width=width, going_to_be_multiplied=True),
              ),
              going_to_be_multiplied=True
          ),
          operators.WeightedSum(bnns=tuple(all_kernels),
                                going_to_be_multiplied=True,
                                num_outputs=num_outputs),
      ),
  )

  return operators.LearnableChangePoint(
      time_series_xs=time_series_xs,
      bnns=(safe_product, safe_product.clone(_deep_clone=True)),
  )


def make_mlp(num_layers: int):
  """Return a make function for the MultiLayerBNN of the given depth."""

  def make_multilayer(
      time_series_xs: Array,
      width: int = 5,
      periods: Sequence[float] = (0.1,),
      num_outputs: int = 1,
  ):
    del num_outputs
    del time_series_xs
    assert len(periods) == 1
    return kernels.MultiLayerBNN(
        num_layers=num_layers,
        width=width,
        period=periods[0],
    )

  return make_multilayer


MODEL_NAME_TO_MAKE_FUNCTION = {
    'sum_of_products': make_sum_of_products,
    'sum_of_changepoints': make_sum_of_changepoints,
    'linear_plus_periodic': make_linear_plus_periodic,
    'sum_of_stumps': make_sum_of_stumps,
    'sum_of_stumps_and_products': make_sum_of_stumps_and_products,
    'sum_of_shallow': make_sum_of_shallow,
    'sum_of_safe_shallow': make_sum_of_safe_shallow,
    'changepoint_of_safe_products': make_changepoint_of_safe_products,
    'mlp_depth2': make_mlp(2),
    'mlp_depth3': make_mlp(3),
    'mlp_depth4': make_mlp(4),
    'mlp_depth5': make_mlp(5),
}


def make_model(
    model_name: Union[str, bnn.BNN],
    likelihood_model: likelihoods.LikelihoodModel,
    time_series_xs: Array,
    width: int = 5,
    periods: Sequence[float] = (0.1,),
) -> bnn.BNN:
  """Create a BNN model by name."""
  if isinstance(model_name, str):
    m = MODEL_NAME_TO_MAKE_FUNCTION[model_name](
        time_series_xs=time_series_xs,
        width=width,
        periods=periods,
        num_outputs=likelihood_model.num_outputs(),
    )
  else:
    m = model_name

  m.set_likelihood_model(likelihood_model)
  return m
