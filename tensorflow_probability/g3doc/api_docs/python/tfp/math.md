<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.math

TensorFlow Probability math functions.



Defined in [`python/math/__init__.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/__init__.py).

<!-- Placeholder for "Used in" -->


## Modules

[`ode`](../tfp/math/ode.md) module: TensorFlow Probability ODE solvers.

## Functions

[`batch_interp_regular_1d_grid(...)`](../tfp/math/batch_interp_regular_1d_grid.md): Linear `1-D` interpolation on a regular (constant spacing) grid.

[`batch_interp_regular_nd_grid(...)`](../tfp/math/batch_interp_regular_nd_grid.md): Multi-linear interpolation on a regular (constant spacing) grid.

[`cholesky_concat(...)`](../tfp/math/cholesky_concat.md): Concatenates `chol @ chol.T` with additional rows and columns.

[`clip_by_value_preserve_gradient(...)`](../tfp/math/clip_by_value_preserve_gradient.md): Clips values to a specified min and max while leaving gradient unaltered.

[`custom_gradient(...)`](../tfp/math/custom_gradient.md): Embeds a custom gradient into a `Tensor`.

[`dense_to_sparse(...)`](../tfp/math/dense_to_sparse.md): Converts dense `Tensor` to `SparseTensor`, dropping `ignore_value` cells.

[`diag_jacobian(...)`](../tfp/math/diag_jacobian.md): Computes diagonal of the Jacobian matrix of `ys=fn(xs)` wrt `xs`.

[`interp_regular_1d_grid(...)`](../tfp/math/interp_regular_1d_grid.md): Linear `1-D` interpolation on a regular (constant spacing) grid.

[`log1psquare(...)`](../tfp/math/log1psquare.md): Numerically stable calculation of `log(1 + x**2)` for small or large `|x|`.

[`lu_matrix_inverse(...)`](../tfp/math/lu_matrix_inverse.md): Computes a matrix inverse given the matrix's LU decomposition.

[`lu_reconstruct(...)`](../tfp/math/lu_reconstruct.md): The inverse LU decomposition, `X == lu_reconstruct(*tf.linalg.lu(X))`.

[`lu_solve(...)`](../tfp/math/lu_solve.md): Solves systems of linear eqns `A X = RHS`, given LU factorizations.

[`matrix_rank(...)`](../tfp/math/matrix_rank.md): Compute the matrix rank; the number of non-zero SVD singular values.

[`pinv(...)`](../tfp/math/pinv.md): Compute the Moore-Penrose pseudo-inverse of a matrix.

[`pivoted_cholesky(...)`](../tfp/math/pivoted_cholesky.md): Computes the (partial) pivoted cholesky decomposition of `matrix`.

[`random_rademacher(...)`](../tfp/math/random_rademacher.md): Generates `Tensor` consisting of `-1` or `+1`, chosen uniformly at random.

[`random_rayleigh(...)`](../tfp/math/random_rayleigh.md): Generates `Tensor` of positive reals drawn from a Rayleigh distributions.

[`secant_root(...)`](../tfp/math/secant_root.md): Finds root(s) of a function of single variable using the secant method.

[`soft_threshold(...)`](../tfp/math/soft_threshold.md): Soft Thresholding operator.

[`sparse_or_dense_matmul(...)`](../tfp/math/sparse_or_dense_matmul.md): Returns (batched) matmul of a SparseTensor (or Tensor) with a Tensor.

[`sparse_or_dense_matvecmul(...)`](../tfp/math/sparse_or_dense_matvecmul.md): Returns (batched) matmul of a (sparse) matrix with a column vector.

[`value_and_gradient(...)`](../tfp/math/value_and_gradient.md): Computes `f(*xs)` and its gradients wrt to `*xs`.

