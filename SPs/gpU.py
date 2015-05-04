import numpy as np
import numpy.linalg as la
import numpy.random as npr
import sys
sys.path.insert(0, '/home/ulli/Venturecxx-master/backend/lite')
from psp import DeterministicPSP, NullRequestPSP, RandomPSP, TypedPSP
from sp import SP, VentureSPRecord, SPType
import venture.lite.value as v
from CovarianceFunction import RBF
from gp_tools import jitchol,solve_chol
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

class GPnew(object):
  """An immutable GP object."""
  def __init__(self, mean, covariance, samples={}):
    self.mean = mean
    self.covariance = covariance
    self.samples = samples
  
  def toJSON(self):
    return self.samples
  
  def mean_array(self, xs):
      if isinstance(xs,float):
          return map(self.mean, [xs])
      else:
          return col_vec(map(self.mean, xs))
  
  
  def getNormal(self, xs):
    """Returns the mean and covariance matrices at a set of input points."""

    if len(self.samples) == 0: # normal posterior p(y|x)
      mu = self.mean_array(xs)
      sigma = self.covariance.getCovMatrix(xs,xs,'cross')
    else:  # predictive distribution
      x2s = self.samples.keys()
      o2s = self.samples.values()
      mu1 = self.mean_array(xs)
      mu2 = self.mean_array(x2s)
      a2 = o2s
      
      sigma11 = self.covariance.getCovMatrix(xs,xs,'cross')
      sigma12 = self.covariance.getCovMatrix(xs,x2s,'cross')
      sigma21 = self.covariance.getCovMatrix(x2s,xs,'cross')
      sigma22 = self.covariance.getCovMatrix(x2s,x2s,'cross')
      inv22 = la.pinv(sigma22)
      print(mu1)
      print(mu2)
      print(sigma11)
      print(sigma12)
      print(inv22)
      print(sigma21)
      
      mu = mu1 + sigma12 * (inv22 * (a2 - mu2))
      sigma = sigma11 - np.dot(np.dot(sigma12,inv22),sigma21)
    
    return mu, sigma
  def gradient(self,xs):
    n, D = xs.shape
    K = self.covariance.getCovMatrix(x=xs, mode='train')            # evaluate covariance matrix
    m = self.mean_array(xs)                               # ToDo: evaluate mean vector
    #sn2   = np.exp(likfunc.hyp[0])                       # noise variance of likGauss
    sn2   = 0                       # noise variance of likGauss
    L     = jitchol(K/sn2+np.eye(n)).T                     # Cholesky factor of covariance with noise
    alpha = solve_chol(L,y-m)/sn2
    
    dnlZ = []
    Q = solve_chol(L,np.eye(n))/sn2 - np.dot(alpha,alpha.T) # precompute for convenience
    dnlZ.append((Q*covfunc.getDerMatrix(x=xs, mode='train', der=0)).sum()/2.)
    dnlZ.append((Q*covfunc.getDerMatrix(x=xs, mode='train', der=1)).sum()/2.)
    return dnlZ


  def sample(self, xs):
    """Sample at a (set of) point(s)."""
    mu, sigma = self.getNormal(xs)
    print("mu")
    print(mu)
    print("sigma")
    print(sigma)
    os = npr.normal(mu, sigma)
    print("os")
    if isinstance(os,float):
        return [[os]]
    return os.tolist()[0][0]

  def logDensity(self, xs, os):
    """Log density of a set of samples."""
    mu, sigma = self.getNormal(xs)
    return multivariate_normal_logpdf(col_vec(os), mu, sigma)

  def logDensityOfCounts(self):
    """Log density of the current samples."""
    if len(self.samples) == 0:
      return 0
    
    xs = self.samples.keys()
    os = self.samples.values()
    
    mu = self.mean_array(xs)
    sigma = self.covariance.getCovMatrix(self,x=xs,z=None,mode='train')
    
    return multivariate_normal_logpdf(col_vec(os), mu, sigma)
  


class GPnewOutputPSP(RandomPSP):
  def __init__(self, mean, covariance):
    self.mean = mean
    self.covariance = covariance
  
  def makeGP(self, samples):
    return GPnew(self.mean, self.covariance, samples)
  
  def simulate(self,args):
    samples = args.spaux
    xs = args.operandValues[0]
    return self.makeGP(samples).sample(xs)

  def logDensity(self,os,args):
    samples = args.spaux
    xs = args.operandValues[0]
    return self.makeGP(samples).logDensity(xs, os)

  def logDensityOfCounts(self,samples):
    return self.makeGP(samples).logDensityOfCounts()
  
  def gradientOfLogDensity(self,args):
    samples = args.spaux
    xs = args.operandValues[0]
    return self.makeGP(samples).gradient(*xs)
  
  def incorporate(self,os,args):
    samples = args.spaux
    xs = args.operandValues[0]
    if not isinstance(xs,list):
        samples[xs] = os
    else:
        for x, o in zip(xs, os):
            samples[x] = o

  def unincorporate(self,_os,args):
    samples = args.spaux
    xs = args.operandValues[0]
    if not isinstance(xs,list):
        del samples[xs]
    else:
        for x in xs:
            del samples[x]


gpnewType = SPType([v.NumberType()],v.NumberType())

class GPnewSP(SP):
  def __init__(self, mean, covariance):
    self.mean = mean
    self.covariance = covariance
    output = TypedPSP(GPnewOutputPSP(mean, covariance), gpnewType)
    super(GPnewSP, self).__init__(NullRequestPSP(),output)

  def constructSPAux(self): return {}
  def show(self,spaux): return GPnew(self.mean, self.covariance, spaux)

class MakeGPnewOutputPSP(DeterministicPSP):
  def simulate(self,args):
    mean = args.operandValues[0]
    covariance = RBF(args.operandValues[1],args.operandValues[2])

    return VentureSPRecord(GPnewSP(mean, covariance))

  def childrenCanAAA(self): return True

  def description(self, _name=None):
    return """Constructs a Gaussian Process with the given mean and covariance functions. Wrap the gp in a mem if input points might be sampled multiple times. Global Logscore is broken with GPs, as it is with all SPs that have auxen."""

makeGPnewType = SPType([v.AnyType("mean function"), v.NumberType(),v.NumberType()], gpnewType)
makeGPnewSP = SP(NullRequestPSP(), TypedPSP(MakeGPnewOutputPSP(), makeGPnewType))

