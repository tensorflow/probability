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
"""Bijective transformations."""

# pylint: disable=unused-import,wildcard-import,line-too-long,g-importing-member

from tensorflow_probability.python.bijectors.absolute_value import AbsoluteValue
from tensorflow_probability.python.bijectors.ascending import Ascending
from tensorflow_probability.python.bijectors.batch_normalization import BatchNormalization
from tensorflow_probability.python.bijectors.bijector import AutoCompositeTensorBijector
from tensorflow_probability.python.bijectors.bijector import Bijector
from tensorflow_probability.python.bijectors.blockwise import Blockwise
from tensorflow_probability.python.bijectors.chain import Chain
from tensorflow_probability.python.bijectors.cholesky_outer_product import CholeskyOuterProduct
from tensorflow_probability.python.bijectors.cholesky_to_inv_cholesky import CholeskyToInvCholesky
from tensorflow_probability.python.bijectors.composition import Composition
from tensorflow_probability.python.bijectors.correlation_cholesky import CorrelationCholesky
from tensorflow_probability.python.bijectors.cumsum import Cumsum
from tensorflow_probability.python.bijectors.discrete_cosine_transform import DiscreteCosineTransform
from tensorflow_probability.python.bijectors.exp import Exp
from tensorflow_probability.python.bijectors.exp import Log
from tensorflow_probability.python.bijectors.expm1 import Expm1
from tensorflow_probability.python.bijectors.expm1 import Log1p
from tensorflow_probability.python.bijectors.ffjord import FFJORD
from tensorflow_probability.python.bijectors.fill_scale_tril import FillScaleTriL
from tensorflow_probability.python.bijectors.fill_triangular import FillTriangular
from tensorflow_probability.python.bijectors.frechet_cdf import FrechetCDF
from tensorflow_probability.python.bijectors.generalized_pareto import GeneralizedPareto
from tensorflow_probability.python.bijectors.gev_cdf import GeneralizedExtremeValueCDF
from tensorflow_probability.python.bijectors.glow import Glow
from tensorflow_probability.python.bijectors.glow import GlowDefaultExitNetwork
from tensorflow_probability.python.bijectors.glow import GlowDefaultNetwork
from tensorflow_probability.python.bijectors.gompertz_cdf import GompertzCDF
from tensorflow_probability.python.bijectors.gumbel_cdf import GumbelCDF
from tensorflow_probability.python.bijectors.householder import Householder
from tensorflow_probability.python.bijectors.identity import Identity
from tensorflow_probability.python.bijectors.inline import Inline
from tensorflow_probability.python.bijectors.invert import Invert
from tensorflow_probability.python.bijectors.iterated_sigmoid_centered import IteratedSigmoidCentered
from tensorflow_probability.python.bijectors.joint_map import JointMap
from tensorflow_probability.python.bijectors.kumaraswamy_cdf import KumaraswamyCDF
from tensorflow_probability.python.bijectors.lambertw_transform import LambertWTail
from tensorflow_probability.python.bijectors.masked_autoregressive import AutoregressiveNetwork
from tensorflow_probability.python.bijectors.masked_autoregressive import masked_autoregressive_default_template
from tensorflow_probability.python.bijectors.masked_autoregressive import masked_dense
from tensorflow_probability.python.bijectors.masked_autoregressive import MaskedAutoregressiveFlow
from tensorflow_probability.python.bijectors.matrix_inverse_tril import MatrixInverseTriL
from tensorflow_probability.python.bijectors.moyal_cdf import MoyalCDF
from tensorflow_probability.python.bijectors.normal_cdf import NormalCDF
from tensorflow_probability.python.bijectors.pad import Pad
from tensorflow_probability.python.bijectors.permute import Permute
from tensorflow_probability.python.bijectors.power import Power
from tensorflow_probability.python.bijectors.power_transform import PowerTransform
from tensorflow_probability.python.bijectors.rational_quadratic_spline import RationalQuadraticSpline
from tensorflow_probability.python.bijectors.rayleigh_cdf import RayleighCDF
from tensorflow_probability.python.bijectors.real_nvp import real_nvp_default_template
from tensorflow_probability.python.bijectors.real_nvp import RealNVP
from tensorflow_probability.python.bijectors.reciprocal import Reciprocal
from tensorflow_probability.python.bijectors.reshape import Reshape
from tensorflow_probability.python.bijectors.restructure import pack_sequence_as
from tensorflow_probability.python.bijectors.restructure import Restructure
from tensorflow_probability.python.bijectors.restructure import tree_flatten
from tensorflow_probability.python.bijectors.scale import Scale
from tensorflow_probability.python.bijectors.scale_matvec_diag import ScaleMatvecDiag
from tensorflow_probability.python.bijectors.scale_matvec_linear_operator import ScaleMatvecLinearOperator
from tensorflow_probability.python.bijectors.scale_matvec_linear_operator import ScaleMatvecLinearOperatorBlock
from tensorflow_probability.python.bijectors.scale_matvec_lu import MatvecLU
from tensorflow_probability.python.bijectors.scale_matvec_lu import ScaleMatvecLU
from tensorflow_probability.python.bijectors.scale_matvec_tril import ScaleMatvecTriL
from tensorflow_probability.python.bijectors.shift import Shift
from tensorflow_probability.python.bijectors.shifted_gompertz_cdf import ShiftedGompertzCDF
from tensorflow_probability.python.bijectors.sigmoid import Sigmoid
from tensorflow_probability.python.bijectors.sinh import Sinh
from tensorflow_probability.python.bijectors.sinh_arcsinh import SinhArcsinh
from tensorflow_probability.python.bijectors.soft_clip import SoftClip
from tensorflow_probability.python.bijectors.softfloor import Softfloor
from tensorflow_probability.python.bijectors.softmax_centered import SoftmaxCentered
from tensorflow_probability.python.bijectors.softplus import Softplus
from tensorflow_probability.python.bijectors.softsign import Softsign
from tensorflow_probability.python.bijectors.split import Split
from tensorflow_probability.python.bijectors.square import Square
from tensorflow_probability.python.bijectors.tanh import Tanh
from tensorflow_probability.python.bijectors.transform_diagonal import TransformDiagonal
from tensorflow_probability.python.bijectors.transpose import Transpose
from tensorflow_probability.python.bijectors.weibull_cdf import WeibullCDF

# pylint: enable=unused-import,line-too-long,g-importing-member

__all__ = [
    "AbsoluteValue",
    "Ascending",
    "AutoCompositeTensorBijector",
    "AutoregressiveNetwork",
    "BatchNormalization",
    "Bijector",
    "Blockwise",
    # "CategoricalToDiscrete",  # Omitted pending further discussion.
    "Chain",
    "CholeskyOuterProduct",
    "CholeskyToInvCholesky",
    "Composition",
    "CorrelationCholesky",
    "Cumsum",
    "DiscreteCosineTransform",
    "Exp",
    "Expm1",
    "FFJORD",
    "FillScaleTriL",
    "FillTriangular",
    "FrechetCDF",
    "GeneralizedPareto",
    "Glow",
    "GlowDefaultNetwork",
    "GlowDefaultExitNetwork",
    "GompertzCDF",
    "GumbelCDF",
    "GeneralizedExtremeValueCDF",
    "Householder",
    "Identity",
    "Inline",
    "Invert",
    "IteratedSigmoidCentered",
    "JointMap",
    "KumaraswamyCDF",
    "LambertWTail",
    "Log",
    "Log1p",
    "MaskedAutoregressiveFlow",
    "MatrixInverseTriL",
    "MatvecLU",
    "MoyalCDF",
    "NormalCDF",
    "Pad",
    "Permute",
    "Power",
    "PowerTransform",
    "RationalQuadraticSpline",
    "RayleighCDF",
    "RealNVP",
    "Reciprocal",
    "Reshape",
    "Restructure",
    "Scale",
    "ScaleMatvecDiag",
    "ScaleMatvecLinearOperator",
    "ScaleMatvecLinearOperatorBlock",
    "ScaleMatvecLU",
    "ScaleMatvecTriL",
    "Shift",
    "ShiftedGompertzCDF",
    "Sigmoid",
    "Sinh",
    "SinhArcsinh",
    "SoftClip",
    "Softfloor",
    "SoftmaxCentered",
    "Softplus",
    "Softsign",
    "Split",
    "Square",
    "Tanh",
    "TransformDiagonal",
    "Transpose",
    "WeibullCDF",
    "masked_autoregressive_default_template",
    "masked_dense",
    "pack_sequence_as",
    "real_nvp_default_template",
    "tree_flatten",
]
