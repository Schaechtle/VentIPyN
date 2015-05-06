import random
import numpy.random as npr
import math
import scipy.special as ss
import numpy as np
import numpy.linalg as npla
import numbers
import itertools
import scipy.linalg.lapack as lapack
from venture.lite.sp import SP, SPType
from venture.lite.psp import NullRequestPSP, ESRRefOutputPSP, DeterministicPSP, TypedPSP

#
# Some of this code is modified from the PyGPs package #    Copyright (c) by
#    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting
#

def jitchol(A,maxtries=5):
    ''' Copyright (c) 2012, GPy authors (James Hensman, Nicolo Fusi, Ricardo Andrade,
        Nicolas Durrande, Alan Saul, Max Zwiessele, Neil D. Lawrence).
    All rights reserved
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
      * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
      * Neither the name of the <organization> nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    :param A: the matrixed to be decomposited
    :param int maxtries: number of iterations of adding jitters
    '''
    A = np.asfortranarray(A)
    L, info = lapack.dpotrf(A, lower=1)
    if info == 0:
        return L
    else:
        diagA = np.diag(A)
        if np.any(diagA <= 0.):
            raise np.linalg.LinAlgError, "kernel matrix not positive definite: non-positive diagonal elements"
        jitter = diagA.mean() * 1e-9
        while maxtries > 0 and np.isfinite(jitter):
            #print 'Warning: adding jitter of {:.10e} to diagnol of kernel matrix for numerical stability'.format(jitter)
            try:
                return np.linalg.cholesky(A + np.eye(A.shape[0]).T * jitter, lower=True)
            except:
                jitter *= 10
            finally:
                maxtries -= 1
        raise np.linalg.LinAlgError, "kernel matrix not positive definite, even with jitter."



def solve_chol(L, B):
    '''
    Solve linear equations from the Cholesky factorization.
    Solve A*X = B for X, where A is square, symmetric, positive definite. The
    input to the function is L the Cholesky decomposition of A and the matrix B.
    Example: X = solve_chol(chol(A),B)

    :param L: low trigular matrix (cholesky decomposition of A)
    :param B: matrix have the same first dimension of L
    :return: X = A \ B
    '''
    try:
        assert(L.shape[0] == L.shape[1] and L.shape[0] == B.shape[0])
    except AssertionError:
        raise Exception('Wrong sizes of matrix arguments in solve_chol.py');
    X = np.linalg.solve(L,np.linalg.solve(L.T,B))
    return X





def deterministic_typed_psp(f, args_types, return_type, descr=None, sim_grad=None, **kwargs):
  return TypedPSP(deterministic_psp(f, descr, sim_grad), SPType(args_types, return_type, **kwargs))

def ordering(d):
    orderings = [item for item in itertools.permutations(range(d))]
    dir_parameter=""
    ord_arr=""
    for order in orderings:
        dir_parameter+=" 0.5 "
        arr_str="(array "
        for item in order:
            arr_str+= str(item) + " "
        arr_str+=")"
        ord_arr+=arr_str
    dir_parameter_Str="(dirichlet (array " + dir_parameter + " ))"
    ordering_array_Str="(categorical orderPrior (array " + ord_arr + " ))"
    return dir_parameter_Str,ordering_array_Str

def ordering2(d):
    orderings = [item for item in itertools.permutations(range(d))]
    ord_arr=""
    for order in orderings:
        arr_str="(array "
        for item in order:
            arr_str+= str(item) + " "
        arr_str+=")"
        ord_arr+=arr_str
    ordering_array_Str="(categorical orderPrior (array " + ord_arr + " ))"
    return ordering_array_Str,len(orderings)

# This one is from http://stackoverflow.com/questions/1167617/in-python-how-do-i-indicate-im-overriding-a-method
def override(interface_class):
  def overrider(method):
    assert method.__name__ in dir(interface_class)
    return method
  return overrider

def extendedLog(x): return math.log(x) if x > 0 else float("-inf")

def normalizeList(seq): 
  denom = sum(seq)
  if denom > 0: return [ float(x)/denom for x in seq]
  else: 
    # Treat all impossible options as equally impossible.
    n = float(len(seq))
    return [1.0/n for x in seq]

def simulateCategorical(ps,os=None):
  if os is None: os = range(len(ps))
  ps = normalizeList(ps)
  return os[npr.multinomial(1,ps).argmax()]

def logDensityCategorical(val,ps,os=None):
  if os is None: os = range(len(ps))
  ps = normalizeList(ps)
  # TODO This should work for Venture Values while the comparison is
  # done by identity and in the absence of observations; do I want to
  # override the Python magic methods for VentureValues?
  p = None
  for i in range(len(os)): 
    if os[i] == val: 
      p = ps[i]
      break
  assert p is not None
  if p == 0:
    return float('-inf')
  return math.log(p)

def simulateDirichlet(alpha): return npr.dirichlet(alpha)

def logDensityDirichlet(theta, alpha):
  theta = np.array(theta)
  alpha = np.array(alpha)

  return ss.gammaln(sum(alpha)) - sum(ss.gammaln(alpha)) + np.dot((alpha - 1).T, np.log(theta).T)

# CONSIDER why not use itertools.prod?
def cartesianProduct(original):
  if len(original) == 0: return [[]]
  elif len(original) == 1: return [[x] for x in original[0]]
  else:
    firstGroup = original[0]
    recursiveProduct = cartesianProduct(original[1:])
    return [ [v] + vs for v in firstGroup for vs in recursiveProduct]

def logaddexp(items):
  "Apparently this was added to scipy in a later version than the one installed on my machine.  Sigh."
  the_max = max(items)
  if the_max > float("-inf"):
    return the_max + math.log(sum(math.exp(item - the_max) for item in items))
  else:
    return the_max # Don't want NaNs from trying to correct from the maximum

def logWeightsToNormalizedDirect(logs):
  "Converts an unnormalized categorical distribution given in logspace to a normalized one given in direct space"
  the_max = max(logs)
  if the_max > float("-inf"):
    # Even if the logs include some -inf values, math.exp will produce
    # zeros there and it will be fine.
    return normalizeList([math.exp(log - the_max) for log in logs])
  else:
    # If all the logs are -inf, force 0 instead of NaN.
    return [0 for _ in logs]

def sampleLogCategorical(logs):
  "Samples from an unnormalized categorical distribution given in logspace."
  the_max = max(logs)
  if the_max > float("-inf"):
    return simulateCategorical([math.exp(log - the_max) for log in logs])
  else:
    # normalizeList, as written above, will actually do the right
    # thing with this, namely treat all impossible options as equally
    # impossible.
    return simulateCategorical([0 for _ in logs])

def numpy_force_number(answer):
  if isinstance(answer, numbers.Number):
    return answer
  else:
    return answer[0,0]

# TODO Change it to use the scipy function when Venture moves to requiring scipy 0.14+
def logDensityMVNormal(x, mu, sigma):
  answer =  -.5*np.dot(np.dot(x-mu, npla.inv(sigma)), np.transpose(x-mu)) \
            -.5*len(sigma)*np.log(2 * np.pi)-.5*np.log(abs(npla.det(sigma)))
  return numpy_force_number(answer)

def careful_exp(x):
  try:
    return math.exp(x)
  except OverflowError:
    if x > 0: return float("inf")
    else: return float("-inf")

class FixedRandomness(object):
  """A Python context manager for executing (stochastic) code repeatably
against fixed randomness.

  Caveat: If the underlying code attempts to monkey with the state of
  the random number generator (other than by calling it) that
  monkeying will be suppressed, and not propagated to its caller. """

  def __init__(self):
    self.pyr_state = random.getstate()
    self.numpyr_state = npr.get_state()
    random.jumpahead(random.randint(1,2**31-1))
    npr.seed(random.randint(1,2**31-1))

  def __enter__(self):
    self.cur_pyr_state = random.getstate()
    self.cur_numpyr_state = npr.get_state()
    random.setstate(self.pyr_state)
    npr.set_state(self.numpyr_state)

  def __exit__(self, _type, _value, _backtrace):
    random.setstate(self.cur_pyr_state)
    npr.set_state(self.cur_numpyr_state)
    return False # Do not suppress any thrown exception
