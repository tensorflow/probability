<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.substrates.numpy.math.linalg" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.experimental.substrates.numpy.math.linalg


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/numpy/math/linalg.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Functions for common linear algebra operations.

<!-- Placeholder for "Used in" -->

Note: Many of these functions will eventually be migrated to core TensorFlow.

## Functions

[`cholesky_concat(...)`](../../../../../tfp/experimental/substrates/numpy/math/linalg/cholesky_concat.md): Concatenates `chol @ chol.T` with additional rows and columns.

[`fill_triangular(...)`](../../../../../tfp/experimental/substrates/numpy/math/fill_triangular.md): Creates a (batch of) triangular matrix from a vector of inputs.

[`fill_triangular_inverse(...)`](../../../../../tfp/experimental/substrates/numpy/math/fill_triangular_inverse.md): Creates a vector from a (batch of) triangular matrix.

[`lu_matrix_inverse(...)`](../../../../../tfp/experimental/substrates/numpy/math/linalg/lu_matrix_inverse.md): Computes a matrix inverse given the matrix's LU decomposition.

[`lu_reconstruct(...)`](../../../../../tfp/experimental/substrates/numpy/math/linalg/lu_reconstruct.md): The inverse LU decomposition, `X == lu_reconstruct(*tf.linalg.lu(X))`.

[`lu_reconstruct_assertions(...)`](../../../../../tfp/experimental/substrates/numpy/math/linalg/lu_reconstruct_assertions.md): Returns list of assertions related to `lu_reconstruct` assumptions.

[`lu_solve(...)`](../../../../../tfp/experimental/substrates/numpy/math/linalg/lu_solve.md): Solves systems of linear eqns `A X = RHS`, given LU factorizations.

[`matrix_rank(...)`](../../../../../tfp/experimental/substrates/numpy/math/linalg/matrix_rank.md): DEPRECATED FUNCTION

[`pinv(...)`](../../../../../tfp/experimental/substrates/numpy/math/linalg/pinv.md): DEPRECATED FUNCTION

[`pivoted_cholesky(...)`](../../../../../tfp/experimental/substrates/numpy/math/linalg/pivoted_cholesky.md): Computes the (partial) pivoted cholesky decomposition of `matrix`.

[`sparse_or_dense_matmul(...)`](../../../../../tfp/experimental/substrates/numpy/math/linalg/sparse_or_dense_matmul.md): Returns (batched) matmul of a SparseTensor (or Tensor) with a Tensor.

[`sparse_or_dense_matvecmul(...)`](../../../../../tfp/experimental/substrates/numpy/math/linalg/sparse_or_dense_matvecmul.md): Returns (batched) matmul of a (sparse) matrix with a column vector.

