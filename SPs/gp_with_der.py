import numpy as np
import numpy.linalg as la
import numpy.random as npr
#from utils import jitchol, solve_chol
from venture.lite.psp import DeterministicMakerAAAPSP, NullRequestPSP, RandomPSP, TypedPSP
from venture.lite.sp import SP, VentureSPRecord, SPType,SPAux
import venture.lite.types as t
import venture.lite.value as v
import numpy.linalg as npla
import copy
import collections
from venture.lite.exception import VentureValueError
import scipy.linalg.lapack as lapack
# XXX Replace by scipy.stats.multivariate_normal.logpdf when we
# upgrade to scipy 0.14.

def multivariate_normal_logpdf(x, mu, sigma):
  try:
    # dev = x - mu
    ans = 0
    ans += (-.5*(x-mu).transpose() * la.inv(sigma) * (x-mu))[0, 0]
    ans += -.5*len(sigma)*np.log(2 * np.pi)
    ans += -.5*np.log(la.det(sigma))
    return ans
  except la.LinAlgError:
    raise VentureValueError("Bad GP covariance matrix.")
    
def col_vec(xs):
  return np.matrix([xs]).T

class GP(object):
  """An immutable GP object."""
  def __init__(self, mean, covariance, samples={},test_call=False):
    self.mean = mean
    self.covariance = covariance
    self.samples = (collections.OrderedDict() if samples is None else samples)
    if not test_call:
        self.derivatives=covariance.stuff['derivatives']
  def toJSON(self):
    return self.samples
  
  def mean_array(self, xs):
    return col_vec(map(self.mean, xs))
  
  def cov_matrix(self, x1s, x2s=None):
    if x2s is None:
      return self.covariance(np.asmatrix(x1s).T)
    return self.covariance(np.asmatrix(x1s).T, np.asmatrix(x2s).T) #ToDo: that's ugly and inefficient
  
  def getNormal(self, xs):
    """Returns the mean and covariance matrices at a set of input points."""

    if len(self.samples) == 0:
        mu = self.mean_array(xs)
        sigma = self.cov_matrix(xs, xs)
        #print(sigma)
    else:

        x2s = self.samples.keys()
        o2s = self.samples.values()
        
        mu1 = self.mean_array(xs)
        mu2 = self.mean_array(x2s)
        a2 = col_vec(o2s)
    
        sigma11 = self.cov_matrix(xs, xs)
        sigma12 = self.cov_matrix(xs, x2s)
        sigma21 = self.cov_matrix(x2s, xs)
        sigma22 = self.cov_matrix(x2s,x2s)
        inv22 = la.pinv(sigma22)

        mu = mu1 +np.dot(sigma12,(np.dot(inv22, (a2 - mu2))))

        sigma = sigma11 - np.dot(sigma12,np.dot(inv22,sigma21))

    return mu, sigma

  def sample(self, *xs):
    """Sample at a (set of) point(s)."""
    mu, sigma = self.getNormal(xs)
    os = npr.multivariate_normal(mu.A1, sigma)
    return os

  def logDensity(self, xs, os):
    #print("logDensity")
    n = len(xs)
    K = self.cov_matrix(xs,xs)            # evaluate covariance matrix
    m = self.mean_array(xs)
    y = np.asmatrix(os).T                        # evaluate mean vector
    #print("np.exp(likfunc.hyp[0])",np.exp(likfunc.hyp[0]))
    sn2   = 0.1                       # noise variance of likGauss
    #L     = np.linalg.cholesky(K/sn2+np.eye(n)).T         # Cholesky factor of covariance with noise
    try:
        L     = jitchol(K/sn2+np.eye(n)).T                     # Cholesky factor of covariance with noise
    except:
        #print("numerical issues with K, trying the naive way")
        mu, sigma = self.getNormal(xs)
        return multivariate_normal_logpdf(col_vec(os), mu, sigma)
    alpha = solve_chol(L,y-m)/sn2
    nlZ = np.dot((y-m).T,alpha)/2. + np.log(np.diag(L)).sum() + n*np.log(2*np.pi)/2. # -log marg lik
    lD = -float(nlZ)
    print(lD)
    return lD



  def logDensityOfCounts(self):
    """Log density of the current samples."""
    if len(self.samples) == 0:
      return 0

    xs = self.samples.keys()
    os = self.samples.values()
    n = len(xs)
    K = self.cov_matrix(xs,xs)            # evaluate covariance matrix
    m = self.mean_array(xs)
    y = np.asmatrix(os).T                        # evaluate mean vector
    #print("np.exp(likfunc.hyp[0])",np.exp(likfunc.hyp[0]))
    sn2   = 0.1                       # noise variance of likGauss
    #L     = np.linalg.cholesky(K/sn2+np.eye(n)).T         # Cholesky factor of covariance with noise
    try:
        L     = jitchol(K/sn2+np.eye(n)).T                     # Cholesky factor of covariance with noise
    except:
        #print("numerical issues with K, trying the naive way")
        mu, sigma = self.getNormal(xs)
        return multivariate_normal_logpdf(col_vec(os), mu, sigma)
    alpha = solve_chol(L,y-m)/sn2
    nlZ = np.dot((y-m).T,alpha)/2. + np.log(np.diag(L)).sum() + n*np.log(2*np.pi)/2. # -log marg lik
    lD = -float(nlZ)
    return lD

  
  def gradient(self):
    xs = self.samples.keys()
    n = len(xs)
    os = self.samples.values()
    y = np.asmatrix(os).T
    K = self.cov_matrix(xs, xs)    # evaluate covariance matrix
    m = self.mean_array(xs)                             # ToDo: evaluate mean vector
    sn2   = 0.1                       # noise variance of likGauss
    L     = jitchol(K/sn2+np.eye(n)).T                     # Cholesky factor of covariance with noise
    alpha = solve_chol(L,y-m)/sn2
    dnlZ = []
    Q = np.dot(alpha,alpha.T) - solve_chol(L,np.eye(n))/sn2 # precompute for convenience
    
    for i in range(len(self.derivatives)):
        dnlZ.append((Q*self.derivatives[i](xs,xs)).sum()/2.)
    return np.zeros(xs.shape).tolist(),[dnlZ]


class GPOutputPSP(RandomPSP):
  def __init__(self, mean, covariance):
    self.mean = mean
    self.covariance = covariance

  def makeGP(self, samples):
    return GP(self.mean, self.covariance, samples)

  def simulate(self,args):
    samples = args.spaux().samples
    xs = args.operandValues()[0]
    return self.makeGP(samples).sample(*xs)

  def logDensity(self,os,args):
    samples = args.spaux().samples
    xs = args.operandValues()[0]
    return self.makeGP(samples).logDensity(xs, os)

  def logDensityOfCounts(self,aux):
    return self.makeGP(aux.samples).logDensityOfCounts()

  def incorporate(self,os,args):
    samples = args.spaux().samples
    xs = args.operandValues()[0]

    for x, o in zip(xs, os):
      samples[x] = o

  def unincorporate(self,_os,args):
    samples = args.spaux().samples
    xs = args.operandValues()[0]
    for x in xs:
      del samples[x]

  def gradientOfLogDensityOfCounts(self,args):
    samples = args.spaux().samples
    xs = args.operandValues()[0]
    return self.makeGP(samples).gradient(*xs)

gpType = SPType([t.ArrayUnboxedType(t.NumberType())], t.ArrayUnboxedType(t.NumberType()))


class GPSPAux(SPAux):
  def __init__(self, samples):
    self.samples = samples
  def copy(self):
    return GPSPAux(copy.copy(self.samples))

  def asVentureValue(self):
    """
    Returns both the (x,y) pair with highest y, and a list of all (x,y) pairs:
    [(best_x, best_y), ((x,y) ...)]
    """
    pairs = self.samples.items()
    ys = map(lambda p: p[1], pairs)
    if len(ys) > 0:
        best_pair = pairs[np.argmax(ys)]
    else:
        best_pair = []
    venturized_items = [v.VentureArray(map(v.VentureNumber, pair)) for pair in pairs]
    return v.VentureArray([
        v.VentureArray(map(v.VentureNumber, best_pair)),
        v.VentureArray(venturized_items)])


class GPSP(SP):
  def __init__(self, mean, covariance):
    self.mean = mean
    self.covariance = covariance
    output = TypedPSP(GPOutputPSP(mean, covariance), gpType)
    super(GPSP, self).__init__(NullRequestPSP(),output)

  def constructSPAux(self): return GPSPAux(collections.OrderedDict())
  def show(self,spaux): return GP(self.mean, self.covariance, spaux)

class MakeGPOutputPSP(DeterministicMakerAAAPSP):
  def simulate(self,args):
    mean = args.operandValues()[0]
    covariance = args.operandValues()[1]
    return VentureSPRecord(GPSP(mean, covariance))

  def gradientOfLogDensityOfCounts(self,aux,args):
    mean_array = args.operandValues()[0]
    cov_matrix = args.operandValues()[1]
    #ipdb.set_trace()
    partial_derivatives=cov_matrix.stuff['derivatives']
    xs = np.asmatrix(aux.samples.keys()).T
    n = len(xs)
    os = aux.samples.values()
    y = np.asmatrix(os).T
    K = cov_matrix(xs, xs)    # evaluate covariance matrix
    m = mean_array(xs)                             # ToDo: evaluate mean vector
    #sn2   = np.exp(likfunc.hyp[0])                       # noise variance of likGauss
    sn2   = 0.1                       # noise variance of likGauss
    L     = jitchol(K/sn2+np.eye(n)).T                     # Cholesky factor of covariance with noise
    alpha = solve_chol(L,y-m)/sn2
    dlZ = []
    Q = np.dot(alpha,alpha.T) - solve_chol(L,np.eye(n))/sn2 # precompute for convenience
    isigma = npla.inv(K) # ToDo: this is incorrect, stub for debugging!
    gradX = -np.dot(isigma, xs)
    for i in range(len(partial_derivatives)):
        dlZ.append((Q*partial_derivatives[i](xs,xs)).sum()/2.)
    x_array = xs.tolist()
    #import ipdb;ipdb.set_trace()
    #    return (0,dlZ)

    return [v.VentureNumber(0),t.VentureArrayUnboxed(np.array(dlZ),t.NumberType())]


  def cov_matrix(self, x1s, x2s=None):
    if x2s is None:
        return self.covariance(np.asmatrix(x1s).T)
    return self.covariance(np.asmatrix(x1s).T, np.asmatrix(x2s).T) #ToDo: that's ugly and inefficient
  def childrenCanAAA(self): return True

  def description(self, _name=None):
    return """Constructs a Gaussian Process with the given mean and covariance functions. Wrap the gp in a mem if input points might be sampled multiple times. Global Logscore is broken with GPs, as it is with all SPs that have auxen."""

makeGPType = SPType([t.AnyType("mean function"), t.AnyType("covariance function")], gpType)
makeGPSP = SP(NullRequestPSP(), TypedPSP(MakeGPOutputPSP(), makeGPType))



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
