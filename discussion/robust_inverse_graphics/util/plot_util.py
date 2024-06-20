# Copyright 2024 The TensorFlow Probability Authors.
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
"""Plotting utilities."""
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability.spinoffs.fun_mc.using_jax as fun_mc

__all__ = [
    'COLORS',
    'polkagram_horiz',
    'polkagram_vert',
    'trace_plot',
]

# From
# https://mikemol.github.io/technique/colorblind/2018/02/11/color-safe-palette.html,
# ordered by luminocity.
COLORS = ('#009E73', '#0072B2', '#D55E00', '#E69F00', '#56B4E9', '#F0E442')


# TODO(siege): Move this to some sort of utils directory?
def _exp_mean(vals: jax.Array, window_size: float) -> jax.Array:
  """Exponential moving average."""
  vals = jnp.asarray(vals)

  def kernel(rm_state, i):
    v = vals[i]
    cand_rm_state, _ = fun_mc.running_mean_step(
        rm_state, v, window_size=window_size
    )
    rm_state = fun_mc.choose(jnp.isfinite(v), cand_rm_state, rm_state)
    return (rm_state, i + 1), rm_state.mean

  _, exp_mean = fun_mc.trace(
      (fun_mc.running_mean_init(vals.shape[1:], jnp.float32), 0),
      kernel,
      vals.shape[0],
  )
  return exp_mean


def trace_plot(ax: plt.Axes, vals: jax.Array, color: str = COLORS[0]):
  """Plots a trace, with overlaid EMA trace."""
  ax.plot(vals, color=color)
  ax.plot(_exp_mean(vals, window_size=10), color='k')

  # Drop the extremes.
  y_min = jnp.nanpercentile(vals, 1)
  y_max = jnp.nanpercentile(vals, 99)
  if y_min == y_max:
    y_min = y_min - 0.1
    y_max = y_max + 0.1
  ax.set_ylim(y_min, y_max)


class HistBoxes(matplotlib.collections.PatchCollection):
  """Collection for the histogram parts."""

  def __init__(self, *args, scatter, **scatter_kwargs):
    super().__init__(*args, **scatter_kwargs)
    self.scatter = scatter


class HistBoxesHandler:
  """Legend handler for histograms."""

  def legend_artist(
      self,
      legend: matplotlib.legend.Legend,
      orig_handle: matplotlib.patches.Patch,
      fontsize: int,
      handlebox: matplotlib.offsetbox.DrawingArea,
  ) -> matplotlib.patches.Patch:
    """Creates the legend artist."""
    del fontsize, legend  # Unused
    x0, y0 = handlebox.xdescent, handlebox.ydescent
    width, height = handlebox.width, handlebox.height

    scatter_kwargs = {}
    ec = orig_handle.get_edgecolor()
    fc = orig_handle.get_facecolor()
    if ec is not None and len(ec) > 0:  # pylint: disable=g-explicit-length-test
      scatter_kwargs.update(ec=ec[0])
    else:
      scatter_kwargs.update(ec='none')
    if fc is not None and len(fc) > 0:  # pylint: disable=g-explicit-length-test
      scatter_kwargs.update(fc=fc[0])
    else:
      scatter_kwargs.update(fc='none')

    patch = matplotlib.patches.Rectangle(
        [x0, y0],
        width,
        height,
        lw=orig_handle.get_linewidth()[0],
        transform=handlebox.get_transform(),
        **scatter_kwargs,
    )
    handlebox.add_artist(patch)
    handlebox.add_artist(
        matplotlib.collections.PathCollection(
            orig_handle.scatter.get_paths(),
            offsets=(x0 + width / 2, y0 + height / 2),
            fc=orig_handle.scatter.get_fc(),
            ec=orig_handle.scatter.get_ec(),
            sizes=5 * np.array([min(width, height)]),
        )
    )
    return patch


matplotlib.legend.Legend.update_default_handler_map(
    {HistBoxes: HistBoxesHandler()}
)


def polkagram_vert(
    ys: Sequence[float] | np.ndarray,
    x: float | np.generic = 0.0,
    bins: int = 20,
    width: float = 1.0,
    rng: np.random.RandomState = np.random,
    draw_boxes: bool = True,
    box_ec: str = 'none',
    box_fc: str = 'lightgray',
    ax: plt.Axes | None = None,
    center: bool = True,
    max_point_density: float | None = None,
    min_points: int = 10,
    **scatter_kwargs: Any,
) -> tuple[
    matplotlib.collections.PatchCollection,
    matplotlib.collections.PatchCollection | None,
]:
  """Draw a histogram/scatter combo.

  Args:
    ys: Y-coordinates of the datapoints.
    x: X-coordinate of the datapoints.
    bins: Number of bins to use for the histogram.
    width: Maximum bin width.
    rng: PRNG for the scatter.
    draw_boxes: Whether to add the boxes behind the scatter points.
    box_ec: Box edge color.
    box_fc: Box face color.
    ax: Axis to draw on.
    center: Whether to center the bins around the X axis.
    max_point_density: Maximum number of points per bin size.
    min_points: Minimum number of points to keep per bin. Only relevant when
      `max_point_density` is not `None`.
    **scatter_kwargs: Passed to ax.scatter.

  Returns:
    A pair of collections for the scatter points and boxes.
  """
  if ax is None:
    ax = plt.gca()
  range_ = scatter_kwargs.pop('range', None)

  ys = np.array(ys)
  ys = ys[np.isfinite(ys)]
  if range_ is None:
    range_ = (ys.min(), ys.max())

  heights, edges = np.histogram(ys, bins=bins, range=range_, density=True)

  ids = np.digitize(ys, edges[:-1]) - 1
  heights /= heights.max()

  if center:
    start_frac = -0.5
    end_frac = 0.5
  else:
    start_frac = 0.0
    end_frac = 1.0

  patches = []
  xs = np.empty(ys.shape)
  for bin_id, bin_height in enumerate(heights):
    mask = ids == bin_id
    bin_xs = xs[mask]
    if len(bin_xs) == 0:  # pylint: disable=g-explicit-length-test
      continue
    elif len(bin_xs) == 1:
      bin_xs = np.zeros_like(bin_xs)
    else:
      bin_xs = (
          width * bin_height * np.linspace(start_frac, end_frac, len(bin_xs))
      )
      rng.shuffle(bin_xs)
    xs[mask] = bin_xs
    if max_point_density is not None:
      max_points = max(int(max_point_density * bin_height), min_points)
      if len(bin_xs) > max_points:
        idxs = np.where(mask)[0]
        rng.shuffle(idxs)
        nan_idxs = idxs[max_points:]
        ys[nan_idxs] = np.nan
    patches.append(
        matplotlib.patches.Rectangle(
            (x + width * bin_height * start_frac, edges[bin_id]),
            width * bin_height,
            edges[bin_id + 1] - edges[bin_id],
        )
    )

  if draw_boxes:
    label = scatter_kwargs.pop('label', None)

  isfinite = np.isfinite(xs) | np.isfinite(ys)
  xs = xs[isfinite]
  ys = ys[isfinite]
  scatter = ax.scatter(x + xs, ys, **scatter_kwargs)

  if draw_boxes:
    boxes = ax.add_collection(
        HistBoxes(
            patches,
            facecolor=box_fc,
            edgecolor=box_ec,
            label=label,
            zorder=scatter.zorder - 1,
            scatter=scatter,
        )
    )
  else:
    boxes = None
  return scatter, boxes


def polkagram_horiz(
    xs: Sequence[float] | np.ndarray,
    y: float | np.generic = 0.0,
    bins: int = 20,
    height: float = 1.0,
    rng: np.random.RandomState = np.random,
    draw_boxes: bool = True,
    box_ec: str = 'none',
    box_fc: str = 'lightgray',
    ax: plt.Axes | None = None,
    center: bool = True,
    max_point_density: float | None = None,
    min_points: int = 10,
    **scatter_kwargs: Any,
) -> tuple[
    matplotlib.collections.PatchCollection,
    matplotlib.collections.PatchCollection | None,
]:
  """Draw a histogram/scatter combo.

  Args:
    xs: X-coordinates of the datapoints.
    y: Y-coordinate of the datapoints.
    bins: Number of bins to use for the histogram.
    height: Maximum bin height.
    rng: PRNG for the scatter.
    draw_boxes: Whether to add the boxes behind the scatter points.
    box_ec: Box edge color.
    box_fc: Box face color.
    ax: Axis to draw on.
    center: Whether to center the bins around the Y axis.
    max_point_density: Maximum number of points per bin size.
    min_points: Minimum number of points to keep per bin. Only relevant when
      `max_point_density` is not `None`.
    **scatter_kwargs: Passed to ax.scatter.

  Returns:
    A pair of collections for the scatter points and boxes.
  """
  if ax is None:
    ax = plt.gca()
  range_ = scatter_kwargs.pop('range', None)

  xs = np.array(xs)
  xs = xs[np.isfinite(xs)]
  if range_ is None:
    range_ = (xs.min(), xs.max())

  heights, edges = np.histogram(xs, bins=bins, range=range_, density=True)

  ids = np.digitize(xs, edges[:-1]) - 1
  heights /= heights.max()

  if center:
    start_frac = -0.5
    end_frac = 0.5
  else:
    start_frac = 0.0
    end_frac = 1.0

  patches = []
  ys = np.empty(xs.shape)
  for bin_id, bin_height in enumerate(heights):
    mask = ids == bin_id
    bin_ys = ys[mask]
    if len(bin_ys) == 0:  # pylint: disable=g-explicit-length-test
      continue
    elif len(bin_ys) == 1:
      bin_ys = np.zeros_like(bin_ys)
    else:
      bin_ys = (
          height * bin_height * np.linspace(start_frac, end_frac, len(bin_ys))
      )
      rng.shuffle(bin_ys)
    ys[mask] = bin_ys
    if max_point_density is not None:
      max_points = max(int(max_point_density * bin_height), min_points)
      if len(bin_ys) > max_points:
        idxs = np.where(mask)[0]
        rng.shuffle(idxs)
        nan_idxs = idxs[max_points:]
        xs[nan_idxs] = np.nan
    patches.append(
        matplotlib.patches.Rectangle(
            (edges[bin_id], y + height * bin_height * start_frac),
            edges[bin_id + 1] - edges[bin_id],
            height * bin_height,
        )
    )

  if draw_boxes:
    label = scatter_kwargs.pop('label', None)

  isfinite = np.isfinite(xs) | np.isfinite(ys)
  xs = xs[isfinite]
  ys = ys[isfinite]
  scatter = ax.scatter(xs, y + ys, **scatter_kwargs)

  if draw_boxes:
    boxes = ax.add_collection(
        HistBoxes(
            patches,
            facecolor=box_fc,
            edgecolor=box_ec,
            label=label,
            zorder=scatter.zorder - 1,
            scatter=scatter,
        )
    )
  else:
    boxes = None
  return scatter, boxes
