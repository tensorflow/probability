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
"""The continuous Bernoulli distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import sigmoid as sigmoid_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util


class ContinuousBernoulli(distribution.Distribution):
    """Continuous Bernoulli distribution.

    The continuous Bernoulli distribution with `lam` in (0, 1) parameter,
    supported in [0, 1], the pdf is given by:
    pdf(x|lam) = lam^x * (1 - lam)^(1 - x) * C(lam)
    where C(lam) = 2 * atanh(1 - 2 * lam) / (1 - 2* lam) if lam != 0.5 and 2
    otherwise
    for more details, see:
    The continuous Bernoulli: fixing a pervasive error in variational
    autoencoders, Loaiza-Ganem and Cunningham,
    NeurIPS 2019, https://arxiv.org/abs/1907.06845
    NOTE: Unlike the Bernoulli, numerical instabilities can happen for lams
    very close to 0 or 1. Current implementation allows any value in (0,1),
    but this could be changed to (1e-6, 1-1e-6) to avoid these issues.

    """

    def __init__(
        self,
        logits=None,
        lams=None,
        lims=[0.499, 0.501],
        dtype=tf.float32,
        validate_args=False,
        allow_nan_stats=True,
        name="ContinuousBernoulli"
    ):
        """Construct Bernoulli distributions.

        Args:
          logits: An N-D `Tensor`. Each entry in the `Tensor` parameterizes
           an independent continuous Bernoulli distribution with parameter
           sigmoid(logits). Only one of `logits` or `lams` should be passed
           in. Note that this does not correspond to the log-odds as in the
           Bernoulli case.
          lams: An N-D `Tensor` representing the parameter of a continuous
           Bernoulli. Each entry in the `Tensor` parameterizes an independent
           continuous Bernoulli distribution. Only one of `logits` or `lams`
           should be passed in.
          lims: A list with two floats containing the lower and upper limits
           used to approximate the continuous Bernoulli around 0.5 for
           numerical stability purposes.
          dtype: The type of the event samples. Default: `float32`.
           validate_args: Python `bool`, default `False`. When `True`
           distribution parameters are checked for validity despite possibly
           degrading runtime performance. When `False` invalid inputs may
           silently render incorrect outputs.
          allow_nan_stats: Python `bool`, default `True`. When `True`,
            statistics (e.g., mean, mode, variance) use the value "`NaN`" to
            indicate the result is undefined. When `False`, an exception is
            raised if one or more of the statistic's batch members are
            undefined.
          name: Python `str` name prefixed to Ops created by this class.

        Raises:
          ValueError: If p and logits are passed, or if neither are passed.
        """
        parameters = dict(locals())
        if (lams is None) == (logits is None):
            raise ValueError("Must pass lams or logits, but not both.")
        self._lims = lims
        with tf.name_scope(name) as name:
            self._lams = tensor_util.convert_nonref_to_tensor(
                lams, dtype_hint=tf.float32, name="lams"
            )
            self._logits = tensor_util.convert_nonref_to_tensor(
                logits, dtype_hint=tf.float32, name="logits"
            )
        super(ContinuousBernoulli, self).__init__(
            dtype=dtype,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name
        )

    @staticmethod
    def _param_shapes(sample_shape):
        return {"logits": tf.convert_to_tensor(sample_shape, dtype=tf.int32)}

    @classmethod
    def _params_event_ndims(cls):
        return dict(logits=0, lams=0)

    @property
    def logits(self):
        """Input argument `logits`."""
        return self._logits

    @property
    def lams(self):
        """Input argument `lams`."""
        return self._lams

    def _batch_shape_tensor(self):
        x = self._lams if self._logits is None else self._logits
        return tf.shape(x)

    def _batch_shape(self):
        x = self._lams if self._logits is None else self._logits
        return x.shape

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])

    def _outside_unstable_region(self):
        lams = self._lams_parameter_no_checks()
        return lams < self._lims[0] | lams > self._lims[1]

    def _cut_lams(self):
        lams = self._lams_parameter_no_checks()
        return tf.where(
            self._outside_unstable_region(),
            lams,
            self._lims[0] * tf.ones_like(lams)
        )

    def _sample_n(self, n, seed=None):
        cut_lams = self._cut_lams()
        new_shape = tf.concat([[n], tf.shape(cut_lams)], 0)
        uniform = tf.random.uniform(new_shape, seed=seed, dtype=cut_lams.dtype)
        sample = tf.where(
            self._outside_unstable_region(),
            (
                tf.math.log1p(-cut_lams + uniform * (2.0 * cut_lams - 1.0))
                - tf.math.log1p(-cut_lams)
            )
            / (tf.math.log(cut_lams) - tf.math.log1p(-cut_lams)),
            uniform
        )
        return tf.cast(sample, self.dtype)

    def _cont_bern_log_norm(self):
        lams = self._lams_parameter_no_checks()
        cut_lams = self._cut_lams()
        log_norm = tf.math.log(
            tf.math.abs(tf.math.log1p(-cut_lams) - tf.math.log(cut_lams))
        ) - tf.math.log(tf.math.abs(1 - 2.0 * cut_lams))
        taylor = (
            tf.math.log(2.0)
            + 4.0 / 3.0 * tf.math.pow(lams - 0.5, 2)
            + 104.0 / 45.0 * tf.math.pow(lams - 0.5, 4)
        )
        return tf.where(self._outside_unstable_region(), log_norm, taylor)

    def _log_prob(self, event):
        log_lams0, log_lams1 = self._outcome_log_lams()
        event = tf.cast(event, log_lams0.dtype)
        tentative_log_pdf = (
            tf.math.multiply_no_nan(log_lams0, 1.0 - event)
            + tf.math.multiply_no_nan(log_lams1, event)
            + self._cont_bern_log_norm()
        )
        return tf.where(
            event < 0 | event > 1,
            -float("Inf") * tf.ones_like(tentative_log_pdf),
            tentative_log_pdf
        )

    def _cdf(self, x):
        cut_lams = self._cut_lams()
        cdfs = (
            tf.math.pow(cut_lams, x) * tf.math.pow(1.0 - cut_lams, 1.0 - x)
            + cut_lams
            - 1.0
        ) / (2.0 * cut_lams - 1.0)
        unbounded_cdfs = tf.where(self._outside_unstable_region(), cdfs, x)
        return tf.where(
            x < 0.0,
            tf.zeros_like(x),
            tf.where(x > 1.0, tf.ones_like(x), unbounded_cdfs)
        )

    def _outcome_log_lams(self):
        if self._logits is None:
            lam = tf.convert_to_tensor(self._lams)
            return tf.math.log1p(-lam), tf.math.log(lam)
        s = tf.convert_to_tensor(self._logits)
        # softplus(s) = -Log[1 - p]
        # -softplus(-s) = Log[p]
        # softplus(+inf) = +inf, softplus(-inf) = 0, so...
        #  logits = -inf ==> log_probs0 = 0, log_probs1 = -inf (as desired)
        #  logits = +inf ==> log_probs0 = -inf, log_probs1 = 0 (as desired)
        return -tf.math.softplus(s), -tf.math.softplus(-s)

    def _entropy(self):
        log_lams0, log_lams1 = self._outcome_log_lams()
        return (
            self._mean() * (log_lams0 - log_lams1)
            - self._cont_bern_log_norm()
            - log_lams0
        )

    def _mean(self):
        lams = self._lams_parameter_no_checks()
        cut_lams = self._cut_lams()
        mus = cut_lams / (2.0 * cut_lams - 1.0) + 1.0 / (
            tf.math.log1p(-cut_lams) - tf.math.log(cut_lams)
        )
        taylor = (
            0.5 + (lams - 0.5) / 3.0 + 16.0 / 45.0 * tf.math.pow(lams - 0.5, 3)
        )
        return tf.where(self._outside_unstable_region(), mus, taylor)

    def _variance(self):
        lams = self._lams_parameter_no_checks()
        cut_lams = self._cut_lams()
        vars = cut_lams * (cut_lams - 1.0) / tf.math.pow(
            1.0 - 2.0 * cut_lams, 2
        ) + 1.0 / tf.math.pow(
            tf.math.log1p(-cut_lams) - tf.math.log(cut_lams), 2
        )
        taylor = (
            1.0 / 12.0
            - tf.math.pow(lams - 0.5, 2) / 15.0
            - 128.0 / 945.0 * tf.math.pow(lams - 0.5, 4)
        )
        return tf.where(self._outside_unstable_region(), vars, taylor)

    def _quantile(self, p):
        cut_lams = self._cut_lams()
        return tf.where(
            self._outside_unstable_region(),
            (
                tf.math.log1p(-cut_lams + p * (2.0 * cut_lams - 1.0))
                - tf.math.log1p(-cut_lams)
            )
            / (tf.math.log(cut_lams) - tf.math.log1p(-cut_lams)),
            p
        )

    def _mode(self):
        """Returns `1` if `prob > 0.5` and `0` otherwise."""
        return tf.cast(self._lams_parameter_no_checks() > 0.5, self.dtype)

    def logits_parameter(self, name=None):
        """Logits computed from non-`None` input arg (`lams` or `logits`)."""
        with self._name_and_control_scope(name or "logits_parameter"):
            return self._logits_parameter_no_checks()

    def _logits_parameter_no_checks(self):
        if self._logits is None:
            lams = tf.convert_to_tensor(self._lams)
            return tf.math.log(lams) - tf.math.log1p(-lams)
        return tf.identity(self._logits)

    def lams_parameter(self, name=None):
        """Lams computed from non-`None` input arg (`lams` or `logits`)."""
        with self._name_and_control_scope(name or "lams_parameter"):
            return self._lams_parameter_no_checks()

    def _lams_parameter_no_checks(self):
        if self._logits is None:
            return tf.identity(self._lams)
        return tf.math.sigmoid(self._logits)

    def _default_event_space_bijector(self):
        return sigmoid_bijector.Sigmoid(validate_args=self.validate_args)

    def _parameter_control_dependencies(self, is_init):
        return maybe_assert_continuous_bernoulli_param_correctness(
            is_init, self.validate_args, self._lams, self._logits
        )

    def _sample_control_dependencies(self, x):
        assertions = []
        if not self.validate_args:
            return assertions
        assertions.append(
            assert_util.assert_positive(x, message="Sample must be positive.")
        )
        assertions.append(
            assert_util.assert_less(
                x,
                tf.ones([], dtype=x.dtype),
                message="Sample must be less than `1`."
            )
        )
        return assertions


def maybe_assert_continuous_bernoulli_param_correctness(
    is_init, validate_args, lams, logits
):
    """Return assertions for `Bernoulli`-type distributions."""
    if is_init:
        x, name = (lams, "lams") if logits is None else (logits, "logits")
        if not dtype_util.is_floating(x.dtype):
            raise TypeError(
                "Argument `{}` must having floating type.".format(name)
            )

    if not validate_args:
        return []

    assertions = []

    if lams is not None:
        if is_init != tensor_util.is_ref(lams):
            lams = tf.convert_to_tensor(lams)
            one = tf.constant(1.0, lams.dtype)
            assertions += [
                assert_util.assert_positive(
                    lams, message="lams has components less than or equal to 0."
                ),
                assert_util.assert_less(
                    lams,
                    one,
                    message="lams has components greater than or equal to 1."
                )
            ]

    return assertions


@kullback_leibler.RegisterKL(ContinuousBernoulli, ContinuousBernoulli)
def _kl_bernoulli_bernoulli(a, b, name=None):
    """Calculate the batched KL divergence KL(a || b) with a and b continuous
    Bernoulli.

    Args:
      a: instance of a continuous Bernoulli distribution object.
      b: instance of a continuous Bernoulli distribution object.
      name: Python `str` name to use for created operations.
        Default value: `None`
        (i.e., `'kl_continuous_bernoulli_continuous_bernoulli'`).

    Returns:
      Batchwise KL(a || b)
    """
    with tf.name_scope(name or "kl_continuous_bernoulli_continuous_bernoulli"):
        (
            a_log_lams0,
            a_log_lams1
        ) = a._outcome_log_lams()  # pylint:disable=protected-access
        (
            b_log_lams0,
            b_log_lams1
        ) = b._outcome_log_lams()  # pylint:disable=protected-access
        a_mean = a._mean()  # pylint:disable=protected-access
        a_log_norm = a._cont_bern_log_norm()  # pylint:disable=protected-access
        b_log_norm = b._cont_bern_log_norm()  # pylint:disable=protected-access

        return (
            a_mean * (a_log_lams1 + b_log_lams0 - a_log_lams0 - b_log_lams1)
            + a_log_norm
            - b_log_norm
            + a_log_lams0
            - b_log_lams0
        )
