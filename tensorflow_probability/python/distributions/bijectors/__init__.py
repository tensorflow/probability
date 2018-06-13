# Copyright 2018 The TensorFlow Probability Authors.
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
"""Bijector Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import,line-too-long,g-importing-member

from tensorflow_probability.python.distributions.bijectors.absolute_value import AbsoluteValue
from tensorflow_probability.python.distributions.bijectors.affine import Affine
from tensorflow_probability.python.distributions.bijectors.affine_linear_operator import AffineLinearOperator
from tensorflow_probability.python.distributions.bijectors.affine_scalar import AffineScalar
from tensorflow_probability.python.distributions.bijectors.batch_normalization import BatchNormalization
from tensorflow_probability.python.distributions.bijectors.chain import Chain
from tensorflow_probability.python.distributions.bijectors.cholesky_outer_product import CholeskyOuterProduct
from tensorflow_probability.python.distributions.bijectors.conditional_bijector import ConditionalBijector
from tensorflow_probability.python.distributions.bijectors.exp import Exp
from tensorflow_probability.python.distributions.bijectors.fill_triangular import FillTriangular
from tensorflow_probability.python.distributions.bijectors.gumbel import Gumbel
from tensorflow_probability.python.distributions.bijectors.inline import Inline
from tensorflow_probability.python.distributions.bijectors.invert import Invert
from tensorflow_probability.python.distributions.bijectors.kumaraswamy import Kumaraswamy
from tensorflow_probability.python.distributions.bijectors.masked_autoregressive import masked_autoregressive_default_template
from tensorflow_probability.python.distributions.bijectors.masked_autoregressive import masked_dense
from tensorflow_probability.python.distributions.bijectors.masked_autoregressive import MaskedAutoregressiveFlow
from tensorflow_probability.python.distributions.bijectors.matrix_inverse_tril import MatrixInverseTriL
from tensorflow_probability.python.distributions.bijectors.permute import Permute
from tensorflow_probability.python.distributions.bijectors.power_transform import PowerTransform
from tensorflow_probability.python.distributions.bijectors.real_nvp import real_nvp_default_template
from tensorflow_probability.python.distributions.bijectors.real_nvp import RealNVP
from tensorflow_probability.python.distributions.bijectors.reshape import Reshape
from tensorflow_probability.python.distributions.bijectors.scale_tril import ScaleTriL
from tensorflow_probability.python.distributions.bijectors.sigmoid import Sigmoid
from tensorflow_probability.python.distributions.bijectors.sinh_arcsinh import SinhArcsinh
from tensorflow_probability.python.distributions.bijectors.softmax_centered import SoftmaxCentered
from tensorflow_probability.python.distributions.bijectors.softplus import Softplus
from tensorflow_probability.python.distributions.bijectors.softsign import Softsign
from tensorflow_probability.python.distributions.bijectors.square import Square
from tensorflow_probability.python.distributions.bijectors.transform_diagonal import TransformDiagonal
from tensorflow_probability.python.distributions.bijectors.weibull import Weibull
from tensorflow.python.ops.distributions.bijector import Bijector
from tensorflow.python.ops.distributions.identity_bijector import Identity

# pylint: enable=unused-import,line-too-long,g-importing-member

from tensorflow.python.util.all_util import remove_undocumented

__all__ = [
    "AbsoluteValue", "Affine", "AffineLinearOperator", "AffineScalar",
    "Bijector", "BatchNormalization", "Chain", "CholeskyOuterProduct",
    "ConditionalBijector", "Exp", "FillTriangular", "Gumbel", "Identity",
    "Inline", "Invert", "Kumaraswamy", "MaskedAutoregressiveFlow",
    "MatrixInverseTriL", "Permute", "PowerTransform", "RealNVP", "Reshape",
    "ScaleTriL", "Sigmoid", "SinhArcsinh", "SoftmaxCentered", "Softplus",
    "Softsign", "Square", "TransformDiagonal", "Weibull",
    "masked_autoregressive_default_template", "masked_dense",
    "real_nvp_default_template"
]

remove_undocumented(__name__, __all__)
