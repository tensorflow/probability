<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.bijectors" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.bijectors

Bijective transformations.



Defined in [`python/bijectors/__init__.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/bijectors/__init__.py).

<!-- Placeholder for "Used in" -->


## Classes

[`class AbsoluteValue`](../tfp/bijectors/AbsoluteValue.md): Computes `Y = g(X) = Abs(X)`, element-wise.

[`class Affine`](../tfp/bijectors/Affine.md): Compute `Y = g(X; shift, scale) = scale @ X + shift`.

[`class AffineLinearOperator`](../tfp/bijectors/AffineLinearOperator.md): Compute `Y = g(X; shift, scale) = scale @ X + shift`.

[`class AffineScalar`](../tfp/bijectors/AffineScalar.md): Compute `Y = g(X; shift, scale) = scale * X + shift`.

[`class AutoregressiveLayer`](../tfp/bijectors/AutoregressiveLayer.md): Masked Autoencoder for Distribution Estimation [Germain et al. (2015)].

[`class AutoregressiveNetwork`](../tfp/bijectors/AutoregressiveNetwork.md): Masked Autoencoder for Distribution Estimation [Germain et al. (2015)][1].

[`class BatchNormalization`](../tfp/bijectors/BatchNormalization.md): Compute `Y = g(X) s.t. X = g^-1(Y) = (Y - mean(Y)) / std(Y)`.

[`class Bijector`](../tfp/bijectors/Bijector.md): Interface for transformations of a `Distribution` sample.

[`class Blockwise`](../tfp/bijectors/Blockwise.md): Bijector which applies a list of bijectors to blocks of a `Tensor`.

[`class Chain`](../tfp/bijectors/Chain.md): Bijector which applies a sequence of bijectors.

[`class CholeskyOuterProduct`](../tfp/bijectors/CholeskyOuterProduct.md): Compute `g(X) = X @ X.T`; X is lower-triangular, positive-diagonal matrix.

[`class CholeskyToInvCholesky`](../tfp/bijectors/CholeskyToInvCholesky.md): Maps the Cholesky factor of `M` to the Cholesky factor of `M^{-1}`.

[`class ConditionalBijector`](../tfp/bijectors/ConditionalBijector.md): Conditional Bijector is a Bijector that allows intrinsic conditioning.

[`class Cumsum`](../tfp/bijectors/Cumsum.md): Computes the cumulative sum of a tensor along a specified axis.

[`class DiscreteCosineTransform`](../tfp/bijectors/DiscreteCosineTransform.md): Compute `Y = g(X) = DCT(X)`, where DCT type is indicated by the `type` arg.

[`class Exp`](../tfp/bijectors/Exp.md): Compute `Y = g(X) = exp(X)`.

[`class Expm1`](../tfp/bijectors/Expm1.md): Compute `Y = g(X) = exp(X) - 1`.

[`class FillTriangular`](../tfp/bijectors/FillTriangular.md): Transforms vectors to triangular.

[`class Gumbel`](../tfp/bijectors/Gumbel.md): Compute `Y = g(X) = exp(-exp(-(X - loc) / scale))`.

[`class Identity`](../tfp/bijectors/Identity.md): Compute Y = g(X) = X.

[`class Inline`](../tfp/bijectors/Inline.md): Bijector constructed from custom callables.

[`class Invert`](../tfp/bijectors/Invert.md): Bijector which inverts another Bijector.

[`class IteratedSigmoidCentered`](../tfp/bijectors/IteratedSigmoidCentered.md): Bijector which applies a Stick Breaking procedure.

[`class Kumaraswamy`](../tfp/bijectors/Kumaraswamy.md): Compute `Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a), X in [0, 1]`.

[`class MaskedAutoregressiveFlow`](../tfp/bijectors/MaskedAutoregressiveFlow.md): Affine MaskedAutoregressiveFlow bijector.

[`class MatrixInverseTriL`](../tfp/bijectors/MatrixInverseTriL.md): Computes `g(L) = inv(L)`, where `L` is a lower-triangular matrix.

[`class MatvecLU`](../tfp/bijectors/MatvecLU.md): Matrix-vector multiply using LU decomposition.

[`class NormalCDF`](../tfp/bijectors/NormalCDF.md): Compute `Y = g(X) = NormalCDF(x)`.

[`class Ordered`](../tfp/bijectors/Ordered.md): Maps a vector of increasing elements to an unconstrained vector.

[`class Permute`](../tfp/bijectors/Permute.md): Permutes the rightmost dimension of a `Tensor`.

[`class PowerTransform`](../tfp/bijectors/PowerTransform.md): Compute `Y = g(X) = (1 + X * c)**(1 / c), X >= -1 / c`.

[`class RealNVP`](../tfp/bijectors/RealNVP.md): RealNVP "affine coupling layer" for vector-valued events.

[`class Reciprocal`](../tfp/bijectors/Reciprocal.md): A `Bijector` that computes the reciprocal `b(x) = 1. / x` entrywise.

[`class Reshape`](../tfp/bijectors/Reshape.md): Reshapes the `event_shape` of a `Tensor`.

[`class ScaleTriL`](../tfp/bijectors/ScaleTriL.md): Transforms unconstrained vectors to TriL matrices with positive diagonal.

[`class Sigmoid`](../tfp/bijectors/Sigmoid.md): Bijector which computes `Y = g(X) = 1 / (1 + exp(-X))`.

[`class SinhArcsinh`](../tfp/bijectors/SinhArcsinh.md): Compute `Y = g(X) = Sinh( (Arcsinh(X) + skewness) * tailweight )`.

[`class Softfloor`](../tfp/bijectors/Softfloor.md): Compute a differentiable approximation to `tf.math.floor`.

[`class SoftmaxCentered`](../tfp/bijectors/SoftmaxCentered.md): Bijector which computes `Y = g(X) = exp([X 0]) / sum(exp([X 0]))`.

[`class Softplus`](../tfp/bijectors/Softplus.md): Bijector which computes `Y = g(X) = Log[1 + exp(X)]`.

[`class Softsign`](../tfp/bijectors/Softsign.md): Bijector which computes `Y = g(X) = X / (1 + |X|)`.

[`class Square`](../tfp/bijectors/Square.md): Compute `g(X) = X^2`; X is a positive real number.

[`class Tanh`](../tfp/bijectors/Tanh.md): Bijector that computes `Y = tanh(X)`, therefore `Y in (-1, 1)`.

[`class TransformDiagonal`](../tfp/bijectors/TransformDiagonal.md): Applies a Bijector to the diagonal of a matrix.

[`class Transpose`](../tfp/bijectors/Transpose.md): Compute `Y = g(X) = transpose_rightmost_dims(X, rightmost_perm)`.

[`class Weibull`](../tfp/bijectors/Weibull.md): Compute `Y = g(X) = 1 - exp((-X / scale) ** concentration), X >= 0`.

## Functions

[`masked_autoregressive_default_template(...)`](../tfp/bijectors/masked_autoregressive_default_template.md): Build the Masked Autoregressive Density Estimator (Germain et al., 2015).

[`masked_dense(...)`](../tfp/bijectors/masked_dense.md): A autoregressively masked dense layer. Analogous to `tf.layers.dense`.

[`real_nvp_default_template(...)`](../tfp/bijectors/real_nvp_default_template.md): Build a scale-and-shift function using a multi-layer neural network.

