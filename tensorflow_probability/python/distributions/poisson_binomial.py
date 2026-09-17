# Copyright 2022 The TensorFlow Probability Authors.
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
"""The Poisson Binomial distribution class."""
import tensorflow.compat.v2 as tf
import numpy as np

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from scipy import fft
from math import pi


class PoissonBinomial(distribution.Distribution):

    def __init__(self,
               probs=None,
               validate_args=False,
               allow_nan_stats=True,
             name='PoissonBinomial'):
        """
    Initialize a Poisson Binomial distribution.

    The Poisson Binomial distribution is used to calculate the probability
    of the number of successes in independent Bernoulli trials that are not necessarily
    identically distributed. Each of the trials has its own probability of success,
    so the parameters of this distribution are p1, p2, ..., pn for n Bernoulli trials.

    This naive version uses the fast fourier transform method to obtain the PMF
    of the Poisson Binomial distribution [1]. The CDF is obtained through the cumulative 
    sum of this pdf at a given value of successes.

    ### References

    [1]: Yili Hong. On computing the distribution function for the Poisson binomial distribution.
        _Computational Statistics & Data Analysis_, 59:41-51, 2013.
        https://doi.org/10.1016/j.csda.2012.10.006
 
    """
        self._dtype = dtype_util.common_dtype([probs], tf.float32)
        self._probs = tensor_util.convert_nonref_to_tensor(
          probs, dtype=self._dtype)
        self._total_count = tf.shape(self._probs)
        self._pmf_list = self.get_pmf()
        self._cdf_list = tf.math.cumsum(self._pmf_list)

    

    """Cumulative density function"""
    def _cdf(self, count):
        return self._cdf_list[count]

    """Probability mass function"""
    def _prob(self, count):
        return self._pmf_list[count]

    """PMF helper"""
    def get_pmf(self):
        
        def xi(period, l):
            if l == 0:
                return tf.dtypes.complex(1, 0)
            else:
                wl = period*l
                real = 1+self._probs*(tf.math.cos(wl)-1)
                imag = self._probs*tf.math.sin(wl)
                mod = tf.math.atan2(imag**2+real**2)
                arg = tf.math.atan2(imag,real)
                d = tf.math.exp((tf.math.log(mod)).sum())
                arg_sum = arg.sum()
                a = d*tf.math.cos(arg_sum)
                b = d*tf.math.sin(arg_sum)
                return tf.dtypes.complex(a,b)
    
        period = 2*pi/(1+n)
        n = self._total_count

        x = [xi(period,i) for i in range((n+1)//2+1)]
        for i in range((n+1)//2+1,n+1):
            c = x[n+1-i]
            x.append(tf.math.conj(c))

        return tf.math.real(tf.signal.fft(x))/(n+1)
    

    

        
