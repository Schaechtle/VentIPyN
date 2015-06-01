import numpy as np
import numpy.linalg as la
import numpy.random as npr
from utils import jitchol, solve_chol
from venture.lite.psp import DeterministicMakerAAAPSP, NullRequestPSP, RandomPSP, TypedPSP
from venture.lite.sp import SP, VentureSPRecord, SPType,SPAux
import venture.lite.types as t
import numpy.linalg as npla
import copy
# XXX Replace by scipy.stats.multivariate_normal.logpdf when we
# upgrade to scipy 0.14.
def multivariate_normal_logpdf(x, mu, sigma):
  dev = x - mu
  ans = 0
  ans += (-.5*(x-mu).transpose() * la.inv(sigma) * (x-mu))[0, 0]
  ans += -.5*len(sigma)*np.log(2 * np.pi)
  ans += -.5*np.log(la.det(sigma))
  return ans

def col_vec(xs):
  return np.matrix([xs]).T

class GP(object):
  """An immutable GP object."""
  def __init__(self, mean, covariance, samples={}):
    self.mean = mean
    self.covariance = covariance
    self.samples = samples
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
        '''
        n, D = xs.shape
        K = self.cov_matrix(xs, xs)      # evaluate covariance matrix
        m = self.mean_array(xs)                             # evaluate mean vector
        #print("np.exp(likfunc.hyp[0])",np.exp(likfunc.hyp[0]))
        sn2   = 0.1                   # noise variance of likGauss
        #L     = np.linalg.cholesky(K/sn2+np.eye(n)).T         # Cholesky factor of covariance with noise
        L     = jitchol(K/sn2+np.eye(n)).T                     # Cholesky factor of covariance with noise
        alpha = solve_chol(L,y-m)/sn2
        '''
        
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
    if lD>0:
        print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        return -float('Inf')
    return lD

  def logDensityOfCounts(self):
    """Log density of the current samples."""
    #print("in logDensity of Counts")
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
    if lD>0:
        print(":oooooooooooooooooooooooooooooooooooooooooooooo")
        return -float('Inf')
    return lD
  def gradient(self):
    xs = self.samples.keys()
    n = len(xs)
    os = self.samples.values()
    y = np.asmatrix(os).T
    K = self.cov_matrix(xs, xs)    # evaluate covariance matrix
    m = self.mean_array(xs)                             # ToDo: evaluate mean vector
    #sn2   = np.exp(likfunc.hyp[0])                       # noise variance of likGauss
    sn2   = 0.1                       # noise variance of likGauss
    L     = jitchol(K/sn2+np.eye(n)).T                     # Cholesky factor of covariance with noise
    alpha = solve_chol(L,y-m)/sn2
    dnlZ = []
    Q = np.dot(alpha,alpha.T) - solve_chol(L,np.eye(n))/sn2 # precompute for convenience
    
    for i in range(len(self.derivatives)):
        dnlZ.append((Q*self.derivatives[i](xs,xs)).sum()/2.)
    return dnlZ


class GPOutputPSP(RandomPSP):
  def __init__(self, mean, covariance):
    self.mean = mean
    self.covariance = covariance

  def makeGP(self, samples):
    return GP(self.mean, self.covariance, samples)

  def simulate(self,args):
    samples = args.spaux.samples
    xs = args.operandValues[0]
    return self.makeGP(samples).sample(*xs)

  def logDensity(self,os,args):
    samples = args.spaux.samples
    xs = args.operandValues[0]
    return self.makeGP(samples).logDensity(xs, os)

  def logDensityOfCounts(self,args):
    samples = args.samples
    return self.makeGP(samples).logDensityOfCounts()

  def incorporate(self,os,args):
    samples = args.spaux.samples
    xs = args.operandValues[0]

    for x, o in zip(xs, os):
      samples[x] = o

  def unincorporate(self,_os,args):
    samples = args.spaux.samples
    xs = args.operandValues[0]
    for x in xs:
      del samples[x]

  def gradientOfLogDensityOfCounts(self,args):
    samples = args.spaux.samples
    xs = args.operandValues[0]
    return self.makeGP(samples).gradient(*xs)

gpType = SPType([t.ArrayUnboxedType(t.NumberType())], t.ArrayUnboxedType(t.NumberType()))


class GPSPAux(SPAux):
  def __init__(self, samples):
    self.samples = samples
  def copy(self):
    return GPSPAux(copy.copy(self.samples))


class GPSP(SP):
  def __init__(self, mean, covariance):
    self.mean = mean
    self.covariance = covariance
    output = TypedPSP(GPOutputPSP(mean, covariance), gpType)
    super(GPSP, self).__init__(NullRequestPSP(),output)

  def constructSPAux(self): return GPSPAux({})
  def show(self,spaux): return GP(self.mean, self.covariance, spaux)

class MakeGPOutputPSP(DeterministicMakerAAAPSP):
  def simulate(self,args):
    mean = args.operandValues[0]
    covariance = args.operandValues[1]
    return VentureSPRecord(GPSP(mean, covariance))
  def gradientOfLogDensityOfCounts(self,aux,args):
    mean_array = args.operandValues[0]
    cov_matrix = args.operandValues[1]
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
    dlZ = [0]
    Q = np.dot(alpha,alpha.T) - solve_chol(L,np.eye(n))/sn2 # precompute for convenience
    isigma = npla.inv(K) # ToDo: this is incorrect, stub for debugging!
    gradX = -np.dot(isigma, xs)
    for i in range(len(partial_derivatives)):
        dlZ.append((Q*partial_derivatives[i](xs,xs)).sum()/2.)
    import ipdb;ipdb.set_trace()
    #    return (0,dlZ)
    return dlZ


  def cov_matrix(self, x1s, x2s=None):
    if x2s is None:
        return self.covariance(np.asmatrix(x1s).T)
    return self.covariance(np.asmatrix(x1s).T, np.asmatrix(x2s).T) #ToDo: that's ugly and inefficient
  def childrenCanAAA(self): return True

  def description(self, _name=None):
    return """Constructs a Gaussian Process with the given mean and covariance functions. Wrap the gp in a mem if input points might be sampled multiple times. Global Logscore is broken with GPs, as it is with all SPs that have auxen."""

makeGPType = SPType([t.AnyType("mean function"), t.AnyType("covariance function")], gpType)
makeGPSP = SP(NullRequestPSP(), TypedPSP(MakeGPOutputPSP(), makeGPType))