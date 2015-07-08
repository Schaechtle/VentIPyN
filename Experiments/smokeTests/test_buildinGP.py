
import numpy as np

def const(c):
  def f(x1, x2):
    return c
  return f

def squared_exponential(a, l):
  def f(x1, x2):
    x = (x1-x2)/l
    return a * np.exp(- np.dot(x, x))
  return f


from venture import shortcuts as s
ripl = s.make_lite_church_prime_ripl()

from venture.lite.function import VentureFunction
from venture.lite.sp import SPType
import venture.lite.value as v
import venture.lite.types as t


fType = t.AnyType("VentureFunction")


xType = t.NumberType()
oType = t.NumberType()
kernelType = SPType([xType, xType], oType)



constantType = SPType([t.AnyType()], oType)
def makeConstFunc(c):
  return VentureFunction(lambda _: c, sp_type=constantType)

ripl.assume('make_const_func', VentureFunction(makeConstFunc, [xType], constantType))

def makeSquaredExponential(a, l):
  return VentureFunction(squared_exponential(a, l), sp_type=kernelType)

ripl.assume('make_squared_exponential', VentureFunction(makeSquaredExponential, [t.NumberType(), xType], fType))

program = """
  [assume mean (apply_function make_const_func 0)]

  [assume a (tag (quote hypers ) 0 (inv_gamma 2 5))]

  [assume l (tag (quote hypers ) 1 (inv_gamma 5 50))]


  [assume cov (apply_function make_squared_exponential a l)]

  [assume gp (make_gp mean cov)]


"""

ripl.execute_program(program)



def array(xs):
  return v.VentureArrayUnboxed(np.array(xs), xType)



ripl.observe("(gp (array 1 2 3))",array([1.1, 2.2, 3.3]))

ripl.infer("(mh (quote hypers) one 1)")