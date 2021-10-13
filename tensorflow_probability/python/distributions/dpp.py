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
"""The determinantal point process (DPP) distribution class."""

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import bernoulli
from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow.python.ops import control_flow_util  # pylint: disable=g-direct-tensorflow-import


__all__ = ['DeterminantalPointProcess']


FAST_PATH_ENABLED = True  # Enables correctness tests w/ and w/o optimization.
JAX_MODE = False


def _orthogonal_complement_e_i(vectors, i, gram_schmidt_iters):
  """Computes a basis for the orthogonal complement to `e_i` in `span(vectors)`.

  The orthogonal complement of the coordinate vector `e_i` of the vector space
  `V` is the set of all vectors in `V` that are orthogonal to `e_i`.

  We compute this by first choosing a column `j` of `vectors` with non-zero in
  coordinate `i`. This vector (`col_j`) is subtracted from all other vectors
  with an appropriate weight to zero out row `i`. Finally, we orthonormalize
  using (modified) Gram-Schmidt. For performance reasons, the calling code
  specifies the G-S iteration count.

  For example, suppose we start with the matrix of column vectors:

  ```none
  [ 2  4  7 ]
  [ 4  2  4 ]
  [ 6  6  3 ]
  ```

  If we suppose `i = 1`, we are being asked to zero-out the middle row, i.e.
  orthogonalize with respect to the coordinate vector `e_1 = [0, 1, 0]^T`. We
  can do so by picking `j = argmax(mat[i, :])`, so `j = 0` in this case. Then,
  compute the appropriate weights that would zero out the row, i.e.
  `w=[1, 0.5, 1]` and subtract `mat[:, j:j+1] * w = [2, 4, 6]^T * [1, .5, 1]`.
  This yields the intermediate:

  ```none
  [ 2  4  7 ]   [ 2  1  2 ]   [ 0  3  5 ]
  [ 4  2  4 ] - [ 4  2  4 ] = [ 0  0  0 ]
  [ 6  6  3 ]   [ 6  3  6 ]   [ 0  3 -3 ]
  ```

  We rotate the zero column to the end, and finally return the result of
  applying Gram-Schmidt orthogonalization, i.e.

  ```none
  [ sqrt(.5)  sqrt(.5) 0 ]
  [     0        0     0 ]
  [ sqrt(.5) -sqrt(.5) 0 ]
  ```

  Args:
    vectors: A Tensor of vectors of shape `[..., d, n]` we are orthogonalizing.
    i: The coordinate (against dimension `d`) w.r.t. which we orthogonalize.
    gram_schmidt_iters: Number of iterations of Gram-Schmidt orthonormalization
      to run, generally `n_vectors - iter_num`. Since each iteration of sampling
      reduces the number of nonzero columns by one (in the `n` dim), this allows
      us to save iterations of orthonormalization work.

  Returns:
    orthogonal: A Tensor of shape `[..., d, n]` representing the subspace
      spanned by `vectors` that is orthogonal to `e_i`, the `i`-th coordinate
      vector. The tensor is orthonormalized. It contains at least one more zero
      row (`i`) and zero column than the input vectors (exactly one more if all
      nonzero columns of `vectors` are linearly independent).
  """
  i = tf.convert_to_tensor(i, dtype_hint=tf.int32)
  row_i = tf.gather(vectors, i, axis=-2, batch_dims=len(i.shape))
  j = tf.argmax(tf.abs(row_i), axis=-1)  # Max for numerical stability.
  col_j = tf.gather(vectors, j, axis=-1, batch_dims=len(j.shape))
  val_i_j = tf.gather(row_i, j, axis=-1, batch_dims=len(j.shape))
  weights = row_i / val_i_j[..., tf.newaxis]
  delta = weights[..., tf.newaxis, :] * col_j[..., :, tf.newaxis]
  result = (vectors - delta)
  # Rotate the new zero column to the end.
  d = ps.shape(vectors)[-2]
  n = ps.shape(vectors)[-1]
  mask_d = tf.not_equal(tf.range(d, dtype=i.dtype),
                        i[..., tf.newaxis])[..., tf.newaxis]
  shift_indices = tf.range(n, dtype=j.dtype)
  shift_indices = shift_indices + tf.cast(
      shift_indices >= j[..., tf.newaxis], j.dtype)
  shift_indices = tf.where(
      shift_indices >= tf.cast(n, j.dtype), j[..., tf.newaxis], shift_indices)
  result = tf.gather(
      result, shift_indices, axis=-1, batch_dims=len(shift_indices.shape) - 1)
  mask_n = tf.not_equal(tf.range(n), n - 1)
  result = tf.where(mask_d & mask_n, result, 0)  # Make exactly zero.
  # Orthonormalize. This is equivalent, but faster than tf.linalg.qr(result).q
  return tfp_math.gram_schmidt(result, gram_schmidt_iters)


def _reconstruct_matrix(eigenvalues, eigenvectors, indices=None):
  """Builds submatrix w/ corresponding eigendecomposition at position `indices`.

  Args:
    eigenvalues: A Tensor of shape `[batch_shape, n]`.
    eigenvectors: A Tensor of shape `[batch_shape, d, n]`.
    indices: A boolean Tensor of shape `[d]`.

  Returns:
    matrix: A Tensor of shape `[batch_shape, d, d]` with `k` rows and
      columns not pinned to the identity, where `k` is the number of `True`
      elements in `indices`. (The remaining `d - k` rows/columns have `1`'s on
      the diagonal and `0`'s elsewhere.)
  """
  offset = 0
  if indices is not None:
    mask = tf.cast(indices, eigenvectors.dtype)
    eigenvectors = mask[..., tf.newaxis] * eigenvectors
    offset = 1. - mask
    # TODO(bjp): Consider similar fast path to _sample_from_edpp: gather active
    #     coordinates to the front, padding with diag(ones), then slicing to the
    #     size of the largest submatrix in the batch.
  result = tf.matmul(
      eigenvectors * eigenvalues[..., tf.newaxis, :], eigenvectors,
      transpose_b=True)
  # To ensure the logdet is not -inf, we add `1`s along the diagonal of the
  # masked rows/columns.
  return tf.linalg.set_diag(result, tf.linalg.diag_part(result) + offset)


def _sample_from_edpp(eigenvectors, vector_onehot, seed):
  """Samples a batch of subsets from a DPP given pre-selected elementary DPPs.

  Recall that an elementary DPP is a DPP with eigenvalues all exactly 0 or 1.
  This function implements the second step of standard sampling algorithm for
  DPPs, by sampling subsets based on the E-DPPs obtained by selecting
  `vector_onehot` against the DPP's original eigenvectors.

  Args:
    eigenvectors: A Tensor of `float32` of shape `[..., num_points, num_vecs]`
      representing the eigenvectors of a DPP's L-ensemble matrix, eigenvectors
      in columns. Generally, `num_vecs == num_points`; we name separately to
      distinguish axes.
    vector_onehot:  A Tensor of shape `[..., n_vecs]` whose innermost
      dimension corresponds to 1-hot subset encodings. The subsets represent the
      subset of eigenvectors of the original DPP that define an elementary DPP.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

  Returns:
    samples: A many-hot `bool` Tensor of shape `[..., n_points]`
      representing a batch of 1-hot subset encodings.
  """
  with tf.name_scope('sample_from_edpp'):
    seed = samplers.sanitize_seed(seed)
    # Sort the 1's to the front, and sort corresponding eigenvectors, then mask.
    vector_onehot = tf.cast(vector_onehot, eigenvectors.dtype)
    vector_indices = tf.argsort(vector_onehot, axis=-1, direction='DESCENDING')
    vector_onehot = tf.gather(
        vector_onehot, vector_indices, axis=-1,
        batch_dims=len(vector_indices.shape) - 1)
    eigenvectors = tf.gather(
        eigenvectors, vector_indices, axis=-1,
        batch_dims=len(vector_indices.shape) - 1)
    eigenvectors = eigenvectors * vector_onehot[..., tf.newaxis, :]
    sample_size = tf.reduce_sum(tf.cast(vector_onehot, tf.int32), axis=-1)
    max_sample_size = tf.reduce_max(sample_size)

    d = ps.shape(eigenvectors)[-2]
    n = ps.shape(eigenvectors)[-1]

    # Slice eigvecs to do less work in eager/non-XLA modes.
    if FAST_PATH_ENABLED and not JAX_MODE and (
        tf.executing_eagerly() or
        not control_flow_util.GraphOrParentsInXlaContext(
            tf1.get_default_graph())):
      # We can save some work in non-XLA contexts by reducing the size of the
      # eigenvectors.
      eigenvectors = eigenvectors[..., :max_sample_size]
      n = max_sample_size

    def cond(i, *_):
      return i < max_sample_size

    def body(i, vecs, cur_sample, seed):
      sample_seed, next_seed = samplers.split_seed(seed)
      # squared norm at each coord across active subspace
      is_active = (i < sample_size)
      coord_prob = tf.reduce_sum(tf.square(vecs), axis=-1)
      coord_logits = tf.where(
          is_active[..., tf.newaxis], tf.math.log(coord_prob), 0.)

      idx = categorical.Categorical(logits=coord_logits).sample(
          seed=sample_seed)
      new_vecs = tf.where(
          (tf.range(n) < sample_size[..., tf.newaxis, tf.newaxis] - i - 1) &
          ~cur_sample[..., tf.newaxis],
          _orthogonal_complement_e_i(
              vecs, i=tf.where(is_active, idx, 0),
              gram_schmidt_iters=max_sample_size - i),
          0.)
      # Since range(n) may have unknown shape in the stmt above, we clarify.
      tensorshape_util.set_shape(new_vecs, vecs.shape)
      vecs = tf.where(is_active[..., tf.newaxis, tf.newaxis], new_vecs, vecs)
      cur_sample = (cur_sample |
                    (tf.equal(tf.range(d), idx[..., tf.newaxis]) &
                     is_active[..., tf.newaxis]))
      return i + 1, vecs, cur_sample, next_seed

    _, _, sample, _ = tf.while_loop(
        cond, body,
        (tf.zeros([], tf.int32, name='i'),
         eigenvectors,
         tf.zeros(ps.shape(eigenvectors)[:-1], dtype=tf.bool),
         seed))

    return tf.cast(sample, tf.int32)


class DeterminantalPointProcess(distribution.AutoCompositeTensorDistribution):
  """Determinantal point process (DPP) distribution.

  The DPP disribution parameterized by the eigenvalues and eigenvectors of the
  L-ensemble matrix. The L-ensemble matrix indicates the degree of "repulsion"
  between pairs of items.

  #### Mathematical details

  A Determinantal Point Process is a distribution over subsets of `n` items,
  called the *ground set*. The DPP is parameterized by a positive definite
  matrix of shape `n x n`, the L-ensemble matrix. It assigns to any subset `S`
  of `{1, ..., n}` the probability:

  ```none
  Pr(S) = det(L_S) / det(I + L)
  ```

  where:

  * `L` is the L-ensemble matrix parameterized by `eigenvalues` and
    `eigenvectors`, i.e. `L = U D U^T` for `U = eigenvectors` and
    `D = eigenvalues`.
  * `L_S` is the principal submatrix of `L` indexed by items in `S`. In Numpy
    slicing notation, `L_S = L[S, :][:, S]`.
  * `det` is the matrix determinant.

  Marginal probabilities, i.e. the probability that a sample from the DPP
  contains the subset S, are obtained by way of the marginal kernel:

  ```none
  K = L / (I + L)
  ```

  where `/` is the matrix inverse.

  When sampling a random set `A` from the DPP, the marginal probability of `S`,
  given by `exp(dpp.marginal_log_prob(S))`, is:

  ```none
  Pr(A is a superset of S) = det(K_S)
  ```

  This is a marginal probability in the following sense. If we think of the
  DPP as a joint distribution over `n` binary indicator variables, each telling
  whether a given element is in a given subset `S`, then we can consider the
  marginal distribution obtained by "summing" out some of these binary
  indicators. The resulting marginal distribution happens also to be a DPP. What
  is referred to as the `marginal_log_prob` of `S` (under the original DPP) is
  just the `log_prob` of `S` under the marginal DPP, obtained by summing out the
  indicators of the *complement* of S. This tells us the (log) probability that
  a sample from the full DPP includes `S` as a subset.

  Written in terms of sets, with each `S'` a subset of the complement of `S`:

  ```none
  det(K_S) = sum_{S' s.t. S' intersect S is empty} [ Pr(S union S') ]
  ```

  where `Pr(S union S')` is the probability of sampling exactly `S union S'`
  from the DPP.

  For further detail, see Theorem 2.2 of [3].

  #### Repulsion

  Rewriting `L = B B^T` (which in particular can be done using `B = U sqrt(D)`,
  where `D` are the eigenvalues and `U` the eigenvectors), we have

  ```none
  Pr(S) = Vol^2(b_s1, b_s2, ..., b_sk)
  ```

  where `b_s1, ...` is the `s1`th column of `B`. Hence, the probability of
  sampling two points simultaneously decreases as a function of how colinear
  their corresponding eigenvectors are.

  #### Sampling

  Sampling is implemented following the algorithm introduced in [2] (see also
  [3], Algorithm 1), and proceeds in two phases.

  Given an orthonormalization `L = U D U^T`:

  * First, an elementary DPP (E-DPP) is built by sampling a subset of
    eigenvectors `S` from a Bernoulli distribution with probs equal to
    `D / (D + 1)`. This E-DPP has the same eigenvectors `U` as `L`, but its
    eigenvalues are `1` iff the corresponding Bernoulli trial was succesful,
    `0` otherwise.

  * Then, a number of points `k` equal to the number of selected eigenvalues is
    selected iteratively from the elementary DPP. After sampling a point `i`,
    the kernel is updated by projecting it onto the subspace of eigenvectors
    orthogonal to the `i`th basis vector.

  #### Examples

  Sample points on the unit square grid:

  ```python
  import itertools
  import tensorflow as tf
  import tensorflow_probability as tfp
  import matplotlib.pyplot as plt

  tfd = tfp.distributions
  tfpk = tfp.math.psd_kernels

  grid_size = 16
  # Generate grid_size**2 pts on the unit square.
  grid = np.arange(0, 1, 1./grid_size)
  points = np.array(list(itertools.product(grid, grid)))

  # Create the kernel L that parameterizes the DPP.
  kernel_amplitude = 2.
  kernel_lengthscale = 2. / grid_size
  kernel = tfpk.ExponentiatedQuadratic(kernel_amplitude, kernel_lengthscale)
  kernel_matrix = kernel.matrix(points, points)

  eigenvalues, eigenvectors = tf.linalg.eigh(kernel_matrix)
  dpp = tfd.DeterminantalPointProcess(eigenvalues, eigenvectors)

  # The inner-most dimension of the result of `dpp.sample` is a multi-hot
  # encoding of a subset of {1, ..., ground_set_size}.

  plt.figure(figsize=(6, 6))
  for i, samp in enumerate(dpp.sample(4, seed=(1, 2))):  # 4 x grid_size**2
    plt.subplot(221 + i)
    plt.scatter(*points[np.where(samp)].T)
    plt.xticks([])
    plt.yticks([])
  plt.tight_layout()
  plt.show()

  # Like any TFP distribution, the DPP supports batching and shaped samples.

  kernel_amplitude = [2., 3, 4]  # Build a batch of 3 PSD kernels.
  kernel_lengthscale = 2. / grid_size
  kernel = tfpk.ExponentiatedQuadratic(kernel_amplitude, kernel_lengthscale)
  kernel_matrix = kernel.matrix(points, points)  # 3 x 256 x 256

  eigenvalues, eigenvectors = tf.linalg.eigh(kernel_matrix)
  dpp = tfd.DeterminantalPointProcess(eigenvalues, eigenvectors)
  print(dpp)  # batch shape: [3], event shape: [256]
  samps = dpp.sample(2, seed=(10, 20))
  print(samps.shape)  # shape: [2, 3, 256]
  print(dpp.log_prob(samps))  # tensor with shape [2, 3]
  ```

  #### References

  [1]: Odile Macchi. The coincidence approach to stochastic point processes.
       _Advances in Applied Probability_, 1975.

  [2]: J. Ben Hough, Manjunath Krishnapur, Yuval Peres, Balint Virag.
       Determinantal point processes and independence. _Probability Surveys_,
       2006. https://arxiv.org/abs/math/0503110

  [3]: Alex Kulesza, Ben Taskar. Determinantal point processes for machine
       learning. _Foundations and Trends in Machine Learning_, 2012.
       https://arxiv.org/abs/1207.6083
  """

  def __init__(self,
               eigenvalues,
               eigenvectors,
               validate_args=False,
               allow_nan_stats=False,
               name='DeterminantalPointProcess'):
    """Instantiate a `DeterminantalPointProcess` distribution.

    Args:
      eigenvalues: `float` `Tensor` representing the eigenvalues of the DPP
        kernel (a.k.a. "L"). All eigenvalues must be > 0. Shape has the form
        `[b1, ..., bB, n]` where `n` is the number of points in the ground set.
      eigenvectors: `float` `Tensor` representing the column eigenvectors of the
        DPP kernel ("L"), provided in the same order as the eigenvalues. Shape
        has the form `[b1, ..., bB, n, n]` where `n` is the number of points in
        the ground set. The batch shape components need not be identical to
        those of `eigenvalues`, but must be broadcast compatible with them.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs. Default value: `False`.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or more
        of the statistic's batch members are undefined. Default value: `False`.
      name: Python `str` name prefixed to ops created by this class.
    """
    parameters = dict(locals())

    with tf.name_scope(name) as name:
      param_dtype = dtype_util.common_dtype([eigenvalues, eigenvectors],
                                            dtype_hint=tf.float32)
      self._eigenvalues = tensor_util.convert_nonref_to_tensor(
          eigenvalues, dtype_hint=param_dtype, name='eigenvalues')
      self._eigenvectors = tensor_util.convert_nonref_to_tensor(
          eigenvectors, dtype_hint=param_dtype, name='eigenvectors')

      super(DeterminantalPointProcess, self).__init__(
          dtype=tf.int32,  # sample dtype
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        eigenvalues=parameter_properties.ParameterProperties(
            event_ndims=1,
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        eigenvectors=parameter_properties.ParameterProperties(
            event_ndims=2,
            default_constraining_bijector_fn=(
                # TODO(b/171872834): Add tfb.Expm()(tfb.FillSkewSymmetric(..))
                parameter_properties.BIJECTOR_NOT_IMPLEMENTED)))

  def _default_event_space_bijector(self, *args, **kwargs):
    return  # Distribution is discrete.

  def _event_shape_tensor(self):
    return ps.shape(self._eigenvectors)[-2:-1]

  def _event_shape(self):
    return self._eigenvectors.shape[-2:-1]

  @property
  def eigenvalues(self):
    return self._eigenvalues

  @property
  def eigenvectors(self):
    return self._eigenvectors

  def l_ensemble_matrix(self):
    """Returns the L-ensemble parameterization of the DPP."""
    return _reconstruct_matrix(self.eigenvalues, self.eigenvectors)

  def _log_normalization(self, eigvals=None):
    eigvals = self._eigenvalues if eigvals is None else eigvals
    return tf.reduce_sum(tf.math.log1p(eigvals), axis=-1)

  def _log_prob(self, value):
    eigvals = tf.convert_to_tensor(self.eigenvalues)
    eigvecs = tf.convert_to_tensor(self.eigenvectors)
    submatrix = _reconstruct_matrix(eigvals, eigvecs, value)
    return (tf.linalg.logdet(submatrix) -
            self._log_normalization(eigvals=eigvals))

  def marginal_kernel(self):
    """Returns the marginal kernel that defines the DPP."""
    eigvals = tf.convert_to_tensor(self.eigenvalues)
    return _reconstruct_matrix(eigvals / (eigvals + 1.), self.eigenvectors)

  def marginal_log_prob(self, value):
    """Computes the marginal log probability of an event.

    The marginal log probability is the log-probability that a set sampled from
    the DPP will include `value` as a subset. By contrast, `log_prob` returns
    the log-probability of sampling exactly `value`.

    Args:
      value: Tensor broadcastable to `[batch_shape, n_points]` corresponding
        to the one-hot encoding of a subset of points.

    Returns:
      The log marginal probability of `value` according to the DPP.
    """
    eigvals = tf.convert_to_tensor(self.eigenvalues)
    eigvals = eigvals / (eigvals + 1.)
    eigvecs = tf.convert_to_tensor(self.eigenvectors)
    submatrix = _reconstruct_matrix(eigvals, eigvecs, value)
    return tf.linalg.logdet(submatrix)

  # TODO(b/172913602): Faster sampler, e.g. https://arxiv.org/abs/1811.03717
  def _sample_n(self, n, seed=None):
    indices_seed, edpp_seed = samplers.split_seed(seed)
    eigvals = tf.convert_to_tensor(self.eigenvalues)
    eigvecs = tf.convert_to_tensor(self.eigenvectors)

    batch_shape = self._batch_shape_tensor(
        eigenvalues=eigvals, eigenvectors=eigvecs)
    ground_set_size = ps.shape(eigvecs)[-2]
    vecs_size = ps.shape(eigvecs)[-1]

    # First, we select an elementary DPP to construct an elementary DPP kernel.
    # An elementary DPP (E-DPP) is a DPP whose kernel's eigenvalues are in
    # `{0, 1}`. Any DPP is a mixture of E-DPPs. The standard DPP sampling
    # algorithms first selects an E-DPP (this algorithm) before sampling from
    # the E-DPP.
    batch_eigvals_shape = ps.concat([batch_shape, [vecs_size]], axis=0)
    logits = tf.broadcast_to(tf.math.log(eigvals), batch_eigvals_shape)
    # Shape: [n, batch_shape, vecs_size]
    edpp_indices = bernoulli.Bernoulli(logits=logits).sample(
        n, seed=indices_seed)

    # Shape: [n, batch_shape, ground_set_size, vecs_size]
    n_batch_eigvecs_shape = ps.concat(
        [[n], batch_shape, [ground_set_size, vecs_size]],
        axis=0)
    eigvecs = tf.broadcast_to(eigvecs, n_batch_eigvecs_shape)

    # Shape: [n, batch_shape, ground_set_size]
    return _sample_from_edpp(eigvecs, edpp_indices, seed=edpp_seed)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    checks = []
    if is_init != tensor_util.is_ref(self.eigenvectors):
      eigvecs = tf.convert_to_tensor(self.eigenvectors)
      checks += [
          tf.debugging.assert_near(
              tf.eye(ps.shape(eigvecs)[-1], dtype=eigvecs.dtype),
              tf.matmul(eigvecs, eigvecs, adjoint_b=True),
              message='`eigenvectors` must be orthonormal.'),
      ]
    if is_init != tensor_util.is_ref(self.eigenvalues):
      checks += [
          tf.debugging.assert_positive(
              self.eigenvalues,
              message='`eigenvalues` must be positive.'),
      ]

    return checks
