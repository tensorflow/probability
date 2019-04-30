<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.math

TensorFlow Probability math functions.

## Functions

[`batch_interp_regular_1d_grid(...)`](../tfp/math/batch_interp_regular_1d_grid.md): Linear `1-D` interpolation on a regular (constant spacing) grid.

[`custom_gradient(...)`](../tfp/math/custom_gradient.md): Embeds a custom gradient into a `Tensor`.

[`diag_jacobian(...)`](../tfp/math/diag_jacobian.md): Computes diagonal of the Jacobian matrix of `ys=fn(xs)` wrt `xs`.

[`interp_regular_1d_grid(...)`](../tfp/math/interp_regular_1d_grid.md): Linear `1-D` interpolation on a regular (constant spacing) grid.

[`log1psquare(...)`](../tfp/math/log1psquare.md): A numerically stable implementation of log(1 + x**2).

[`lu_matrix_inverse(...)`](../tfp/math/lu_matrix_inverse.md): Computes a matrix inverse given the matrix's LU decomposition.

[`lu_reconstruct(...)`](../tfp/math/lu_reconstruct.md): The inverse LU decomposition, `X == lu_reconstruct(*tf.linalg.lu(X))`.

[`lu_solve(...)`](../tfp/math/lu_solve.md): Solves systems of linear eqns `A X = RHS`, given LU factorizations.

[`pinv(...)`](../tfp/math/pinv.md): Compute the Moore-Penrose pseudo-inverse of a matrix.

[`random_rademacher(...)`](../tfp/math/random_rademacher.md): Generates `Tensor` consisting of `-1` or `+1`, chosen uniformly at random.

[`random_rayleigh(...)`](../tfp/math/random_rayleigh.md): Generates `Tensor` of positive reals drawn from a Rayleigh distributions.

[`secant_root(...)`](../tfp/math/secant_root.md): Finds root(s) of a function of single variable using the secant method.

