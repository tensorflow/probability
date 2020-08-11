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
# Lint as: python3
"""Utility functions for tfp.experimental.lazybones."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

from tensorflow_probability.python.experimental.lazybones import deferred


# Since neither `networkx` or `matplotlib.plt` are official TFP dependencies, we
# lazily import them only as needed using the `_import_once_*` functions defined
# below.
nx = None
plt = None


__all__ = [
    'get_leaves',
    'get_roots',
    'is_any_ancestor',
    'iter_edges',
    'plot_graph',
]


def get_leaves(ancestor):
  """Returns all childless descendants ("leaves") of (iterable) ancestors."""
  return _get_relation(ancestor, 'children')


def get_roots(descendant):
  """Returns all parentless ancestors ("roots") of (iterable) descendants."""
  return _get_relation(descendant, 'parents')


def is_any_ancestor(v, ancestor):
  """Returns `True` if any member of `v` has an ancestor in `ancestor`s."""
  v = set(_prep_arg(v))
  ancestor = set(_prep_arg(ancestor))
  return (any(v_ in ancestor for v_ in v) or
          any(is_any_ancestor(v_.parents, ancestor) for v_ in v))


def iter_edges(leaves, from_roots=False):
  """Returns iter over all `(parent, child)` edges in `leaves`' ancestors."""
  leaves = _prep_arg(leaves)
  for child in leaves:
    parents = child.children if from_roots else child.parents
    for parent in parents:
      yield (parent, child)
    # The following is only supported in >= Python3.3:
    # yield from iter_edges(parents, from_roots=from_roots)
    for e in iter_edges(parents, from_roots=from_roots):
      yield e


def plot_graph(leaves,
               pos=None,
               with_labels=True,
               arrowsize=10,
               node_size=1200,
               fig=None,
               labels=lambda node: node.name,
               seed=42,
               **kwargs):
  """Plots `leaves` and ancestors. (See `help(nx.draw_networkx)` for kwargs)."""
  _import_once_nx()
  _import_once_plt()

  if isinstance(leaves, nx.Graph):
    nx_graph = leaves
  else:
    nx_graph = nx.DiGraph()
    nx_graph.add_edges_from(iter_edges(leaves))

  if pos is None:
    pos = lambda g: nx.spring_layout(g, seed=seed)
  if callable(pos):
    pos = pos(nx_graph)

  if fig is None:
    fig = plt.figure()  # or, f,ax=plt.subplots()
  if isinstance(fig, plt.Figure):
    fig = fig.add_axes((0, 0, 1, 1))  # or, f.gca()
  if not isinstance(fig, plt.Axes):
    raise ValueError()

  if callable(labels):
    labels = dict((v, labels(v)) for v in nx_graph.nodes())

  nx.draw(nx_graph,
          pos=pos,
          arrows=kwargs.pop('arrows', arrowsize > 0),
          with_labels=with_labels,
          arrowsize=arrowsize,
          node_size=node_size,
          ax=fig,
          labels=labels)

  return nx_graph, fig


def _get_relation(vertices, attr):
  vertices = set(_prep_arg(vertices))
  relations = set()
  for v in vertices:
    near_relation = getattr(v, attr)
    relations.update(_get_relation(near_relation, attr) if near_relation
                     else (v,))
  return relations


def _import_once_nx():
  global nx
  if nx is None:
    import networkx as nx  # pylint: disable=g-import-not-at-top,redefined-outer-name


def _import_once_plt():
  global plt
  if plt is None:
    import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top,redefined-outer-name


def _prep_arg(x):
  if isinstance(x, deferred.DeferredBase):
    return (x,)
  return x
