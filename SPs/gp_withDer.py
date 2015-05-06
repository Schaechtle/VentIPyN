
#
# Some of this code is modified from the PyGPs package #    Copyright (c) by
#    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting
#


import numpy as np
import numpy.linalg as la
import numpy.random as npr
from utils import jitchol,solve_chol


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
  
    def cov_matrix(self, x1s, x2s=None): #ToDo: that's ugly and inefficient
        if x2s is None:
            return self.covariance(np.asmatrix(x1s).T,None)
        else:
            return self.covariance(np.asmatrix(x1s).T, np.asmatrix(x2s).T)
  
    def getNormal(self, xs):
        """Returns the mean and covariance matrices at a set of input points."""

        if len(self.samples) == 0:
            mu = self.mean_array(xs)
            sigma = self.cov_matrix(xs, xs)
        else:
            x2s = self.samples.keys()
            o2s = self.samples.values()
            n = len(x2s)
            y = np.asmatrix(o2s).T
            K = self.cov_matrix(x2s,x2s)            # evaluate covariance matrix
            #print("np.exp(likfunc.hyp[0])",np.exp(likfunc.hyp[0]))
            sn2   = 0.01                      # noise variance of likGauss
            #L     = np.linalg.cholesky(K/sn2+np.eye(n)).T         # Cholesky factor of covariance with noise
            L     = jitchol(K/sn2+np.eye(n)).T                     # Cholesky factor of covariance with noise
            alpha = solve_chol(L,y)/sn2
            ks = self.cov_matrix(x2s,xs)
            mu = ks.T * alpha
            Kss= self.cov_matrix(xs,xs)
            v = solve_chol(L,ks)
            sigma = Kss - np.dot(v.T,v)
        return mu, sigma

    def sample(self, *xs):
        """Sample at a (set of) point(s)."""
        mu, sigma = self.getNormal(xs)
        os = npr.multivariate_normal(mu.A1, sigma)
        return os
    '''
    def logDensity(self, xs, os):
        n = len(xs)
        K = self.cov_matrix(xs, xs)      # evaluate covariance matrix
        m = self.mean_array(xs)                             # evaluate mean vector
        L     = jitchol(K+np.eye(n)).T                     # Cholesky factor of covariance with noise
        alpha = solve_chol(L,np.asmatrix(os).T)
        nlZ = np.dot(np.asmatrix(os),alpha)/2. + np.log(np.diag(L)).sum() + n*np.log(2*np.pi)/2. # -log marg lik
        return -float(nlZ)
    '''
    def logDensity(self, xs, os):
        if len(self.samples) == 0:
            mu = self.mean_array(xs)
            sigma = self.cov_matrix(xs, xs)
            return multivariate_normal_logpdf(col_vec(os), mu, sigma)
        else:
            n = len(xs)
            y = np.asmatrix(os).T
            K = self.cov_matrix(xs,xs)            # evaluate covariance matrix
            #print("np.exp(likfunc.hyp[0])",np.exp(likfunc.hyp[0]))
            sn2   = 0.01                      # noise variance of likGauss
            #L     = np.linalg.cholesky(K/sn2+np.eye(n)).T         # Cholesky factor of covariance with noise
            L     = jitchol(K/sn2+np.eye(n)).T                     # Cholesky factor of covariance with noise
            alpha = solve_chol(L,y)/sn2
            nlZ =  np.dot((y).T,alpha)/2. + np.log(np.diag(L)).sum() + n*np.log(2*np.pi)/2. # -log marg lik
            return  -float(nlZ)
    def logDensityOfCounts(self):
        """Log density of the current samples."""
        if len(self.samples) == 0:
          return 0

        xs = self.samples.keys()
        os = self.samples.values()

        mu = self.mean_array(xs)
        sigma = self.cov_matrix(xs, xs)

        return multivariate_normal_logpdf(col_vec(os), mu, sigma)

    def gradient(self):
        if len(self.samples) == 0:
            return 0
        xs = self.samples.keys()
        y = self.samples.values()
        n, D = xs.shape
        K = self.cov_matrix(xs, xs)    # evaluate covariance matrix
        m = self.mean_array(xs)                             # ToDo: evaluate mean vector
        #sn2   = np.exp(likfunc.hyp[0])                       # noise variance of likGauss
        sn2   = 0.01                       # noise variance of likGauss
        L     = jitchol(K/sn2+np.eye(n)).T                     # Cholesky factor of covariance with noise
        alpha = solve_chol(L,y-m)/sn2

        dnlZ = []
        Q = np.dot(alpha,alpha.T) - solve_chol(L,np.eye(n))/sn2 # precompute for convenience

        for i in range(len(self.derivatives)):
            dnlZ.append((Q*self.derivatives[i](xs,xs)).sum()/2.)
        return dnlZ

from venture.lite.psp import DeterministicPSP, NullRequestPSP, RandomPSP, TypedPSP
from venture.lite.sp import SP, VentureSPRecord, SPType
import venture.lite.value as v

class GPOutputPSP(RandomPSP):
  def __init__(self, mean, covariance):  
    self.mean = mean
    self.covariance = covariance
  def makeGP(self, samples):
    return GP(self.mean, self.covariance, samples)
  
  def simulate(self,args):
    samples = args.spaux
    xs = args.operandValues[0]
    return self.makeGP(samples).sample(*xs)

  def logDensity(self,os,args):
    samples = args.spaux
    xs = args.operandValues[0]
    return self.makeGP(samples).logDensity(xs, os)

  def logDensityOfCounts(self,samples):
    return self.makeGP(samples).logDensityOfCounts()
 
  def gradientOfLogDensity(self,args):
    samples = args.spaux
    xs = args.operandValues[0]
    return self.makeGP(samples).gradient()
 
  def incorporate(self,os,args):
    samples = args.spaux
    xs = args.operandValues[0]
    
    for x, o in zip(xs, os):
      samples[x] = o

  def unincorporate(self,_os,args):

    samples = args.spaux
    xs = args.operandValues[0]
    #print(samples)
    for x in xs:
        #print(x)
        del samples[x]

gpType = SPType([v.ArrayUnboxedType(v.NumberType())], v.ArrayUnboxedType(v.NumberType()))

class GPSP(SP):
  def __init__(self, mean, covariance):
    self.mean = mean
    self.covariance = covariance
    output = TypedPSP(GPOutputPSP(mean, covariance), gpType)
    super(GPSP, self).__init__(NullRequestPSP(),output)

  def constructSPAux(self): return {}
  def show(self,spaux): return GP(self.mean, self.covariance, spaux)

class MakeGPOutputPSP(DeterministicPSP):
  def simulate(self,args):
    mean = args.operandValues[0]
    covariance = args.operandValues[1]

    return VentureSPRecord(GPSP(mean, covariance))

  def childrenCanAAA(self): return True

  def description(self, _name=None):
    return """Constructs a Gaussian Process with the given mean and covariance functions. Wrap the gp in a mem if input points might be sampled multiple times. Global Logscore is broken with GPs, as it is with all SPs that have auxen."""

makeGPType = SPType([v.AnyType("mean function"), v.AnyType("covariance function")], gpType)
makeGPSP = SP(NullRequestPSP(), TypedPSP(MakeGPOutputPSP(), makeGPType))


