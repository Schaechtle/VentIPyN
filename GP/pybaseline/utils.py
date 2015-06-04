from math import lgamma, exp, log
import numpy as np

def attach_to(target):
    def a(thing):
        setattr(target, thing.__name__, thing)
    return a

def logassess_normal(x, mean, stdev):
    return -log(stdev) - 0.5*log(2*np.pi) - (x-mean)**2 / (2 * stdev**2)

def logassess_gamma(x, shape, scale):
    return -lgamma(shape) - shape*log(scale) + (shape-1)*log(x) - x/scale
