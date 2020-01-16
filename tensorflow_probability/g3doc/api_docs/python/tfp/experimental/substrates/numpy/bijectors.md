<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.substrates.numpy.bijectors" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.experimental.substrates.numpy.bijectors


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/numpy/bijectors/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Bijective transformations.

<!-- Placeholder for "Used in" -->


## Classes

[`class AbsoluteValue`](../../../../tfp/experimental/substrates/numpy/bijectors/AbsoluteValue.md): Computes `Y = g(X) = Abs(X)`, element-wise.

[`class Affine`](../../../../tfp/experimental/substrates/numpy/bijectors/Affine.md): Compute `Y = g(X; shift, scale) = scale @ X + shift`.

[`class AffineLinearOperator`](../../../../tfp/experimental/substrates/numpy/bijectors/AffineLinearOperator.md): Compute `Y = g(X; shift, scale) = scale @ X + shift`.

[`class AffineScalar`](../../../../tfp/experimental/substrates/numpy/bijectors/AffineScalar.md): Compute `Y = g(X; shift, scale) = scale * X + shift`.

[`class BatchNormalization`](../../../../tfp/experimental/substrates/numpy/bijectors/BatchNormalization.md): Compute `Y = g(X) s.t. X = g^-1(Y) = (Y - mean(Y)) / std(Y)`.

[`class Bijector`](../../../../tfp/experimental/substrates/numpy/bijectors/Bijector.md): Interface for transformations of a `Distribution` sample.

[`class Blockwise`](../../../../tfp/experimental/substrates/numpy/bijectors/Blockwise.md): Bijector which applies a list of bijectors to blocks of a `Tensor`.

[`class Chain`](../../../../tfp/experimental/substrates/numpy/bijectors/Chain.md): Bijector which applies a sequence of bijectors.

[`class CholeskyOuterProduct`](../../../../tfp/experimental/substrates/numpy/bijectors/CholeskyOuterProduct.md): Compute `g(X) = X @ X.T`; X is lower-triangular, positive-diagonal matrix.

[`class CholeskyToInvCholesky`](../../../../tfp/experimental/substrates/numpy/bijectors/CholeskyToInvCholesky.md): Maps the Cholesky factor of `M` to the Cholesky factor of `M^{-1}`.

[`class CorrelationCholesky`](../../../../tfp/experimental/substrates/numpy/bijectors/CorrelationCholesky.md): Maps unconstrained reals to Cholesky-space correlation matrices.

[`class Cumsum`](../../../../tfp/experimental/substrates/numpy/bijectors/Cumsum.md): Computes the cumulative sum of a tensor along a specified axis.

[`class DiscreteCosineTransform`](../../../../tfp/experimental/substrates/numpy/bijectors/DiscreteCosineTransform.md): Compute `Y = g(X) = DCT(X)`, where DCT type is indicated by the `type` arg.

[`class Exp`](../../../../tfp/experimental/substrates/numpy/bijectors/Exp.md): Compute `Y = g(X) = exp(X)`.

[`class Expm1`](../../../../tfp/experimental/substrates/numpy/bijectors/Expm1.md): Compute `Y = g(X) = exp(X) - 1`.

[`class FillTriangular`](../../../../tfp/experimental/substrates/numpy/bijectors/FillTriangular.md): Transforms vectors to triangular.

[`class Gumbel`](../../../../tfp/experimental/substrates/numpy/bijectors/Gumbel.md): Compute `Y = g(X) = exp(-exp(-(X - loc) / scale))`.

[`class Identity`](../../../../tfp/experimental/substrates/numpy/bijectors/Identity.md): Compute Y = g(X) = X.

[`class Inline`](../../../../tfp/experimental/substrates/numpy/bijectors/Inline.md): Bijector constructed from custom callables.

[`class Invert`](../../../../tfp/experimental/substrates/numpy/bijectors/Invert.md): Bijector which inverts another Bijector.

[`class IteratedSigmoidCentered`](../../../../tfp/experimental/substrates/numpy/bijectors/IteratedSigmoidCentered.md): Bijector which applies a Stick Breaking procedure.

[`class Kumaraswamy`](../../../../tfp/experimental/substrates/numpy/bijectors/Kumaraswamy.md): Compute `Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a), X in [0, 1]`.

[`class MatrixInverseTriL`](../../../../tfp/experimental/substrates/numpy/bijectors/MatrixInverseTriL.md): Computes `g(L) = inv(L)`, where `L` is a lower-triangular matrix.

[`class NormalCDF`](../../../../tfp/experimental/substrates/numpy/bijectors/NormalCDF.md): Compute `Y = g(X) = NormalCDF(x)`.

[`class Ordered`](../../../../tfp/experimental/substrates/numpy/bijectors/Ordered.md): Maps a vector of increasing elements to an unconstrained vector.

[`class Pad`](../../../../tfp/experimental/substrates/numpy/bijectors/Pad.md): Pads a value to the `event_shape` of a `Tensor`.

[`class Permute`](../../../../tfp/experimental/substrates/numpy/bijectors/Permute.md): Permutes the rightmost dimension of a `Tensor`.

[`class PowerTransform`](../../../../tfp/experimental/substrates/numpy/bijectors/PowerTransform.md): Compute `Y = g(X) = (1 + X * c)**(1 / c), X >= -1 / c`.

[`class RationalQuadraticSpline`](../../../../tfp/experimental/substrates/numpy/bijectors/RationalQuadraticSpline.md): A piecewise rational quadratic spline, as developed in [1].

[`class Reciprocal`](../../../../tfp/experimental/substrates/numpy/bijectors/Reciprocal.md): A `Bijector` that computes the reciprocal `b(x) = 1. / x` entrywise.

[`class Reshape`](../../../../tfp/experimental/substrates/numpy/bijectors/Reshape.md): Reshapes the `event_shape` of a `Tensor`.

[`class ScaleTriL`](../../../../tfp/experimental/substrates/numpy/bijectors/ScaleTriL.md): Transforms unconstrained vectors to TriL matrices with positive diagonal.

[`class Sigmoid`](../../../../tfp/experimental/substrates/numpy/bijectors/Sigmoid.md): Bijector which computes `Y = g(X) = 1 / (1 + exp(-X))`.

[`class SinhArcsinh`](../../../../tfp/experimental/substrates/numpy/bijectors/SinhArcsinh.md): `Y = g(X) = Sinh( (Arcsinh(X) + skewness) * tailweight ) * multiplier`.

[`class Softfloor`](../../../../tfp/experimental/substrates/numpy/bijectors/Softfloor.md): Compute a differentiable approximation to `tf.math.floor`.

[`class SoftmaxCentered`](../../../../tfp/experimental/substrates/numpy/bijectors/SoftmaxCentered.md): Bijector which computes `Y = g(X) = exp([X 0]) / sum(exp([X 0]))`.

[`class Softplus`](../../../../tfp/experimental/substrates/numpy/bijectors/Softplus.md): Bijector which computes `Y = g(X) = Log[1 + exp(X)]`.

[`class Softsign`](../../../../tfp/experimental/substrates/numpy/bijectors/Softsign.md): Bijector which computes `Y = g(X) = X / (1 + |X|)`.

[`class Square`](../../../../tfp/experimental/substrates/numpy/bijectors/Square.md): Compute `g(X) = X^2`; X is a positive real number.

[`class Tanh`](../../../../tfp/experimental/substrates/numpy/bijectors/Tanh.md): Bijector that computes `Y = tanh(X)`, therefore `Y in (-1, 1)`.

[`class TransformDiagonal`](../../../../tfp/experimental/substrates/numpy/bijectors/TransformDiagonal.md): Applies a Bijector to the diagonal of a matrix.

[`class Transpose`](../../../../tfp/experimental/substrates/numpy/bijectors/Transpose.md): Compute `Y = g(X) = transpose_rightmost_dims(X, rightmost_perm)`.

[`class Weibull`](../../../../tfp/experimental/substrates/numpy/bijectors/Weibull.md): Compute `Y = g(X) = 1 - exp((-X / scale) ** concentration), X >= 0`.

