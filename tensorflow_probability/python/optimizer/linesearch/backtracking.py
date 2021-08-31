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
"""Implements the Backtracking line search algorithm.
Line searches are a central component for many optimization algorithms (e.g.
BFGS, conjugate gradient, ISTA, FISTA etc). Sophisticated line search methods 
aim to find the appropriate step length.
This module implements the Backtracking Line Search Algorithm.
"""

function = lambda x: x**2 +3*x
differentiation = lambda x: 2*x + 3
value = 11




def backtracking ( function,
                   differentiation,
                   value,
                   beta = 0.707,
                   alpha = 1): 

  while function(value-(alpha*differentiation(value)))>function(value) -(alpha/2)*((differentiation(value))**2):
    alpha *= beta
  return alpha

backtracking(function,differentiation,value)






  
  
