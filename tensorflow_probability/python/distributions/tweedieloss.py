import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import distribution



_tweedie_sample_note = """definition is [here](https://en.wikipedia.org/wiki/Tweedie_distribution), 
                          Reference Document [here](https://arxiv.org/pdf/1912.12356.pdf)"""


class Tweedie(distribution.AutoCompositeTensorDistribution):

  def __init__(self, p=1.5,
               name='Tweedie'):
    """Input Paramters for the Tweedie Loss are the p value, lies bettern (1,2) for Poisson-gamma EDM   
     |---------------|---------------------|
     |  Value of p   |     Distribution    |
     |---------------|---------------------|
     |       0       |     Normal          |
     |       1       |     Poisson         |
     |     (1,2)     |     Poisson-Gamma   |
     |       2       |     Gamma           |
     |---------------|---------------------|
     
    """
    self.p =p
    self.name = name

  def log_likelihood(self,y,y_pred):
    """The Log likelihood"""
    self.loglikelihood = - y * (tf.pow(y_pred, 1-self.p)/(1-self.p)) + (tf.pow(y_pred,2-self.p)/(2-self.p))
    return tf.reduce_mean(self.loglikelihood)

  def deviance(self,y,y_pred):
    if self.p ==0:
        return tf.pow(y-y_pred,2)
    elif self.p ==1:
        return 2*(y*log(y_pred/y) + y_pred - y)
    elif self.p ==2:
        return 2*(log(y_pred/y)+ y/y_pred -1)
    else:
        return 2*(((max(y,0)**(2-self.p))/((1-self.p)*(2-self.p))) - (y*(tf.pow(y_pred,(1-self.p)))/(1-self.p)) + (tf.pow(y_pred,2-self.p)/(2-self.p)))