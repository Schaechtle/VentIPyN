#python2.7
#Written by Greg Ver Steeg
#See readme.pdf for documentation
#Or go to http://www.isi.edu/~gregv/npeet.html

import scipy.spatial as ss
from scipy.special import digamma,gamma
from math import log,pi
import numpy.random as nr
import numpy as np
import random

#####DISCRETE ESTIMATORS
def entropyd(sx,base=2):
  """ Discrete entropy estimator
      Given a list of samples which can be any hashable object
  """
  return entropyfromprobs(hist(sx),base=base)
def hist(sx):
  #Histogram from list of samples
  d = dict()
  for s in sx:
    d[s] = d.get(s,0) + 1
  return map(lambda z:float(z)/len(sx),d.values())

def entropyfromprobs(probs,base=2):
#Turn a normalized list of probabilities of discrete outcomes into entropy (base 2)
  return -sum(map(elog,probs))/log(base)

