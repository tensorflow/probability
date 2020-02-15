"""FillScaleDiagonal bijector."""

from __future__ import absolute_import, division, print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import chain
from tensorflow_probability.python.bijectors import fill_diagonal
from tensorflow_probability.python.bijectors import shift
from tensorflow_probability.python.bijectors import softplus
from tensorflow_probability.python.bijectors import transform_diagonal
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util

__all__ = ["FillScaleDiagonal"]


class FillScaleDiagonal(chain.Chain):
  """Transforms unconstrained vectors to Diag matrices with positive diagonal.
  This is implemented as a simple `tfb.Chain` of `tfb.FillDiagonal`
  followed by `tfb.TransformDiagonal`, and provided mostly as a
  convenience. The default setup is somewhat opinionated, using a
  Softplus transformation followed by a small shift (`1e-5`) which
  attempts to avoid numerical issues from zeros on the diagonal.
  #### Examples
  ```python
  tfb = tfp.distributions.bijectors
  b = tfb.FillScaleDiagonal(
       diag_bijector=tfb.Exp(),
       diag_shift=None)
  b.forward(x=[0., 0.])
  # Result: [[1., 0.],
  #          [0., 1.]]
  b.inverse(y=[[1., 0],
               [0, 2]])
  # Result: [log(1), log(2)]
  # Define a distribution over PSD matrices of shape `[3, 3]`,
  # with `3` degrees of freedom.
  dist = tfd.TransformedDistribution(
          tfd.Normal(tf.zeros(3), tf.ones(3)),
          tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleDiagonal()]))
  # Using an identity transformation, FillScaleDiagonal is equivalent to
  # tfb.FillDiagonal.
  b = tfb.FillScaleDiagonal(
       diag_bijector=tfb.Identity(),
       diag_shift=None)
  # For greater control over initialization, one can manually encode
  # pre- and post- shifts inside of `diag_bijector`.
  b = tfb.FillScaleDiagonal(
       diag_bijector=tfb.Chain([
         tfb.Shift(1e-3),
         tfb.Softplus(),
         tfb.Shift(0.5413)]),  # tfp.math.softplus_inverse(1.)
                               #  = log(expm1(1.)) = 0.5413
       diag_shift=None)
  ```
  """
  def __init__(self,
               diag_bijector=None,
               diag_shift=1e-5,
               validate_args=False,
               name="fill_scale_diagonal"):
    """Instantiates the `FillScaleDiagonal` bijector.
    Args:
      diag_bijector: `Bijector` instance, used to transform the output diagonal
        to be positive.
        Default value: `None` (i.e., `tfb.Softplus()`).
      diag_shift: Float value broadcastable and added to all diagonal entries
        after applying the `diag_bijector`. Setting a positive
        value forces the output diagonal entries to be positive, but
        prevents inverting the transformation for matrices with
        diagonal entries less than this value.
        Default value: `1e-5`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
        Default value: `False` (i.e., arguments are not validated).
      name: Python `str` name given to ops managed by this object.
        Default value: `fill_scale_diagonal`.
    """
    with tf.name_scope(name) as name:
      if diag_bijector is None:
        diag_bijector = softplus.Softplus(validate_args=validate_args)

      if diag_shift is not None:
        dtype = dtype_util.common_dtype([diag_bijector, diag_shift],
                                        tf.float32)
        diag_shift = tensor_util.convert_nonref_to_tensor(
            diag_shift, name="diag_shift", dtype=dtype)
        diag_bijector = chain.Chain(
            [shift.Shift(shift=diag_shift), diag_bijector])

      super(FillScaleDiagonal, self).__init__(
          [
              transform_diagonal.TransformDiagonal(
                  diag_bijector=diag_bijector),
              fill_diagonal.FillDiagonal()
          ],
          validate_args=validate_args,
          name=name,
      )
