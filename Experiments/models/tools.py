__author__ = 'ulli'

import  venture.lite.types as t
import numpy as np


def array(xs):
  return t.VentureArrayUnboxed(np.array(xs),  t.NumberType())