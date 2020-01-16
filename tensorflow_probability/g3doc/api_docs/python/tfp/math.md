<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.math


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/math/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



TensorFlow Probability math functions.

<!-- Placeholder for "Used in" -->


## Modules

[`ode`](../tfp/math/ode.md) module: TensorFlow Probability ODE solvers.

[`psd_kernels`](../tfp/math/psd_kernels.md) module: Positive-semidefinite kernels package.

## Functions

[`batch_interp_regular_1d_grid(...)`](../tfp/math/batch_interp_regular_1d_grid.md): Linear `1-D` interpolation on a regular (constant spacing) grid.

[`batch_interp_regular_nd_grid(...)`](../tfp/math/batch_interp_regular_nd_grid.md): Multi-linear interpolation on a regular (constant spacing) grid.

[`cholesky_concat(...)`](../tfp/math/cholesky_concat.md): Concatenates `chol @ chol.T` with additional rows and columns.

[`clip_by_value_preserve_gradient(...)`](../tfp/math/clip_by_value_preserve_gradient.md): Clips values to a specified min and max while leaving gradient unaltered.

[`custom_gradient(...)`](../tfp/math/custom_gradient.md): Embeds a custom gradient into a `Tensor`.

[`dense_to_sparse(...)`](../tfp/math/dense_to_sparse.md): Converts dense `Tensor` to `SparseTensor`, dropping `ignore_value` cells.

[`diag_jacobian(...)`](../tfp/math/diag_jacobian.md): Computes diagonal of the Jacobian matrix of `ys=fn(xs)` wrt `xs`.

[`fill_triangular(...)`](../tfp/math/fill_triangular.md): Creates a (batch of) triangular matrix from a vector of inputs.

[`fill_triangular_inverse(...)`](../tfp/math/fill_triangular_inverse.md): Creates a vector from a (batch of) triangular matrix.

[`interp_regular_1d_grid(...)`](../tfp/math/interp_regular_1d_grid.md): Linear `1-D` interpolation on a regular (constant spacing) grid.

[`log1psquare(...)`](../tfp/math/log1psquare.md): Numerically stable calculation of `log(1 + x**2)` for small or large `|x|`.

[`log_add_exp(...)`](../tfp/math/log_add_exp.md): Computes `log(exp(x) + exp(y))` in a numerically stable way.

[`log_combinations(...)`](../tfp/math/log_combinations.md): Multinomial coefficient.

[`log_sub_exp(...)`](../tfp/math/log_sub_exp.md): Compute `log(exp(max(x, y)) - exp(min(x, y)))` in a numerically stable way.

[`lu_matrix_inverse(...)`](../tfp/math/lu_matrix_inverse.md): Computes a matrix inverse given the matrix's LU decomposition.

[`lu_reconstruct(...)`](../tfp/math/lu_reconstruct.md): The inverse LU decomposition, `X == lu_reconstruct(*tf.linalg.lu(X))`.

[`lu_solve(...)`](../tfp/math/lu_solve.md): Solves systems of linear eqns `A X = RHS`, given LU factorizations.

[`matrix_rank(...)`](../tfp/math/matrix_rank.md): Compute the matrix rank of one or more matrices. (deprecated)

[`minimize(...)`](../tfp/math/minimize.md): Minimize a loss function using a provided optimizer.

[`pinv(...)`](../tfp/math/pinv.md): Compute the Moore-Penrose pseudo-inverse of one or more matrices. (deprecated)

[`pivoted_cholesky(...)`](../tfp/math/pivoted_cholesky.md): Computes the (partial) pivoted cholesky decomposition of `matrix`.

[`random_rademacher(...)`](../tfp/math/random_rademacher.md): Generates `Tensor` consisting of `-1` or `+1`, chosen uniformly at random.

[`random_rayleigh(...)`](../tfp/math/random_rayleigh.md): Generates `Tensor` of positive reals drawn from a Rayleigh distributions.

[`reduce_logmeanexp(...)`](../tfp/math/reduce_logmeanexp.md): Computes `log(mean(exp(input_tensor)))`.

[`reduce_weighted_logsumexp(...)`](../tfp/math/reduce_weighted_logsumexp.md): Computes `log(abs(sum(weight * exp(elements across tensor dimensions))))`.

[`secant_root(...)`](../tfp/math/secant_root.md): Finds root(s) of a function of single variable using the secant method.

[`smootherstep(...)`](../tfp/math/smootherstep.md): Computes a sigmoid-like interpolation function on the unit-interval.

[`soft_threshold(...)`](../tfp/math/soft_threshold.md): Soft Thresholding operator.

[`softplus_inverse(...)`](../tfp/math/softplus_inverse.md): Computes the inverse softplus, i.e., x = softplus_inverse(softplus(x)).

[`sparse_or_dense_matmul(...)`](../tfp/math/sparse_or_dense_matmul.md): Returns (batched) matmul of a SparseTensor (or Tensor) with a Tensor.

[`sparse_or_dense_matvecmul(...)`](../tfp/math/sparse_or_dense_matvecmul.md): Returns (batched) matmul of a (sparse) matrix with a column vector.

[`value_and_gradient(...)`](../tfp/math/value_and_gradient.md): Computes `f(*xs)` and its gradients wrt to `*xs`.

