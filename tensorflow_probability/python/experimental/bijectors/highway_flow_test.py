import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions

seed = 1  # test_util.test_seed(sampler_type='stateless')


def _dx(x, activation):
    if activation == 'sigmoid':
        return tf.math.sigmoid(x) * (1 - tf.math.sigmoid(x))
    elif activation == 'softplus':
        return tf.math.sigmoid(x)
    elif activation == 'tanh':
        return 1. - tf.math.tanh(x) ** 2


def _activation_log_det_jacobian(x, residual_fraction, activation):
    if activation == 'none':
        return tf.zeros(x.shape[0])
    else:
        return tf.reduce_sum(tf.math.log(
            residual_fraction + (1 - residual_fraction) * _dx(x, activation)),
            -1)


@test_util.test_all_tf_execution_regimes
class HighwayFlowTests(test_util.TestCase):

    def testBijector(self):
        width = 2
        for dim in range(2):
            if dim == 0:
                x = tf.ones((5,
                             width)) * samplers.uniform((5, width), minval=-1.,
                                                        maxval=1.,
                                                        seed=seed)
            elif dim == 1:
                x = tf.ones((5, width,
                             width)) * samplers.uniform((5, width, width),
                                                        minval=-1.,
                                                        maxval=1., seed=seed)

            bijector = tfp.experimental.bijectors.build_highway_flow_layer(
                width, activation_fn=tf.nn.softplus)
            self.evaluate([v.initializer for v in bijector.trainable_variables])
            self.assertStartsWith(bijector.name, 'highway_flow')
            self.assertAllClose(x, bijector.inverse(
                tf.identity(bijector.forward(x))))
            self.assertAllClose(
                bijector.forward_log_det_jacobian(x, event_ndims=dim + 1),
                -bijector.inverse_log_det_jacobian(
                    tf.identity(bijector.forward(x)), event_ndims=dim + 1))

    def testJacobianWithActivation(self):
        activations = ['softplus']
        batch_size = 3
        width = 4
        dtype = tf.float32
        residual_fraction = tf.constant(0.5)
        for activation in activations:

            if activation == 'sigmoid':
                activation_fn = tf.nn.sigmoid
            elif activation == 'softplus':
                activation_fn = tf.nn.softplus
            elif activation == 'tanh':
                activation_fn = tf.nn.tanh
            elif activation == 'none':
                activation_fn = None

            bijector = tfp.experimental.bijectors.HighwayFlow(
                residual_fraction=residual_fraction,
                activation_fn=activation_fn,
                bias=tf.zeros(width),
                upper_diagonal_weights_matrix=tf.eye(width),
                lower_diagonal_weights_matrix=tf.eye(width)
            )

            self.evaluate([v.initializer for v in bijector.trainable_variables])
            x = tf.ones((batch_size,
                         width)) * samplers.uniform((batch_size, width), -10.,
                                                    10., seed=seed)
            if activation == 'none':
                y = x
            else:
                y = residual_fraction * x + (
                        1 - residual_fraction) * activation_fn(x)
            expected_forward_log_det_jacobian = \
                _activation_log_det_jacobian(x,
                                             residual_fraction,
                                             activation)
            expected_inverse_log_det_jacobian = \
                -expected_forward_log_det_jacobian
            self.assertAllClose(y, bijector.forward(x))
            self.assertAllClose(x, bijector.inverse(y))
            self.assertAllClose(
                expected_inverse_log_det_jacobian,
                bijector.inverse_log_det_jacobian(y, event_ndims=1),
            )
            self.assertAllClose(
                expected_forward_log_det_jacobian,
                bijector.forward_log_det_jacobian(x, event_ndims=1),
            )

    def testResidualFractionGradientsWithCenteredDifference(self):
        width = 4
        batch_size = 3
        residual_fraction = tf.constant(0.5)
        bijector = tfp.experimental.bijectors.HighwayFlow(
            residual_fraction=residual_fraction,
            activation_fn=tf.nn.softplus,
            bias=tf.zeros(width),
            upper_diagonal_weights_matrix=tf.eye(width),
            lower_diagonal_weights_matrix=tf.eye(width)
        )
        target = tfd.MultivariateNormalDiag(loc=tf.zeros(width),
                                            scale_diag=tf.ones(width))
        x = tf.ones((batch_size, width))
        with tf.GradientTape() as g:
            g.watch(bijector.residual_fraction)
            y = tf.reduce_mean(target.log_prob(bijector.forward(x)))
        tf_grad = g.gradient(y, bijector.residual_fraction)

        h = 1e-3

        bijector._residual_fraction = residual_fraction + h
        y1 = tf.reduce_mean(target.log_prob(bijector.forward(x)))
        bijector._residual_fraction = residual_fraction - h
        y2 = tf.reduce_mean(target.log_prob(bijector.forward(x)))

        manual_grad = (y1 - y2) / (2 * h)

        self.assertAllClose(tf_grad, manual_grad, rtol=1e-4)


if __name__ == '__main__':
    tf.test.main()
