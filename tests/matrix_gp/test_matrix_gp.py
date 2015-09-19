# Copyright (c) 2014 MIT Probabilistic Computing Project.
#
# This file is part of Venture.
#
# Venture is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Venture is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Venture.  If not, see <http://www.gnu.org/licenses/>.

'''
Tests for GP Inference
==========================
We do both, smoke tests and statistical tests
--------------------------
We start with testing the parameters of the multivariate normal, as determined by the GP and some observations.
'''
import numpy as np
from scipy import stats
import venture.lite.types as t
from venture import shortcuts
from venture.test.config import in_backend

import sys
sys.path.append('../SPs/')
from venture.lite.function import VentureFunction
import imp
import collections
gp_w_der = imp.load_source('gp_with_der', '/home/ulli/VentIPyN/SPs/gp_with_der.py')
covs = imp.load_source('rationalQuadratic', '/home/ulli/VentIPyN/Experiments/covFunctions_noLog.py')

import seaborn as sns

@in_backend("lite")
def testNormalParameters():
  r'''
  Training data:
  :math:`\mathbf{x} =[1.3, {-2}, 0]^T` 

  :math:`\mathbf{y} =[5, 2.3, 8.2]^T`

  Unseen  data: 

  :math:`\mathbf{x}^* =[1.4, {-3.2} ]^T` 

  We provide a squared-exponential covariance matrix with :math:`\sigma = 2.1` and :math:`\ell=1.8`.  
  '''
  x = np.array([[1.3,-2,0]]).T
  z =   np.array([[5,2.3,8]]).T
  y = np.array([[1.4,-3.2]])
  expect_mu= np.array([ [4.6307],[-0.9046]])
  expect_sig= np.array([[  0.0027,   -0.0231],[ -0.0231,    1.1090]])
  sigma = 2.1
  l = 1.8
  observations = collections.OrderedDict([(x[i][0], z[i][0])for i in range(x.shape[0])])

  gp_class = gp_w_der.GP(lambda x:0,covs.squared_exponential(sigma,l),observations,True)
  actual_mu,actual_sig = gp_class.getNormal(y)
  np.testing.assert_almost_equal(actual_mu, expect_mu, decimal=4)
  np.testing.assert_almost_equal(actual_sig, expect_sig, decimal=4)


@in_backend("lite")
def testOneSample(): 
  r'''
  Training data:

  :math:`\mathbf{x} =[1.3, {-2}, 0]^T` 

  :math:`\mathbf{y} =[5, 2.3, 8.2]^T`

  Unseen  data: 

  :math:`x^* =1.4` 

  We provide a squared-exponential covariance matrix with :math:`\sigma = 2.1` and :math:`\ell=1.8`.  

  We use a one-sample t-test to test if the samples are not different from a Gaussian with the mean that was analytically computed by hand. We also add some sanity checks in the end. 
  Future applications of this smoke test may reveal it to be too conservative.
  ''' 
  x = np.array([[1.3,-2,0]]).T
  z =   np.array([[5,2.3,8]]).T
  y = 1.4
  expect_mu= np.array([[4.6307]])
  sigma = 2.1
  l = 1.8
  observations = collections.OrderedDict([(x[i][0], z[i][0])for i in range(x.shape[0])])
  gp_class = gp_w_der.GP(lambda x:0,covs.squared_exponential(sigma,l),observations,True)

  n = 200 # number of samples drawn
  samples =[]
  for i in range(n):
    samples.append(gp_class.sample(y))
  (test_result, p_value)=stats.ttest_1samp(samples,expect_mu)
  assert(p_value>=0.05)
  (test_result, p_value)=stats.ttest_1samp(samples,0)
  assert(p_value<0.05)
  (test_result, p_value)=stats.ttest_1samp(np.random.uniform(0,1,n),expect_mu)
  assert(p_value<0.05)
  (test_result, p_value)=stats.ttest_1samp(np.random.normal(10,1,n),n) # sanity check
  assert(p_value<0.05)


@in_backend("none")
def testTwoSamples():
  r'''
  Training data:

  :math:`\mathbf{x} =[1.3, {-2}, 0]^T` 

  :math:`\mathbf{y} =[5, 2.3, 8]^T`

  Unseen  data with low covariance: 

  :math:`\mathbf{x}^* =[1.4,-20]` 

  Unseen  data with high covariance: 

  :math:`\mathbf{x}^* =[1.4,1.5]` 

  We provide a squared-exponential covariance matrix with :math:`\sigma = 2.1` and :math:`\ell=1.8`.  
  We compute Pearson's r and test for significance to check whether covariance is high with two datapoints where we expect it and where we do not expect.
  ''' 
  x = np.array([[1.3,-2,0]]).T
  z =   np.array([[5,2.3,8]]).T
  y_low_cov =[1.4,-20]
  y_high_cov = [1.4,1.5]

  sigma = 2.1
  l = 1.8
  observations = collections.OrderedDict([(x[i][0], z[i][0])for i in range(x.shape[0])])

  gp_class = gp_w_der.GP(lambda x:0,covs.squared_exponential(sigma,l),observations,True)

  n = 200 # number of samples drawn
  low_cov_x = []
  low_cov_y = []
  high_cov_x = []
  high_cov_y = []
  high_cov_samples = []
  for i in range(n):
    x,y=gp_class.sample(y_low_cov)
    low_cov_x.append(x)
    low_cov_y.append(y)
    high_cov_samples.append(gp_class.sample(y_high_cov))

    x,y=gp_class.sample(y_high_cov)
    high_cov_x.append(x)
    high_cov_y.append(y)
    high_cov_samples.append(gp_class.sample(y_high_cov))

  (test_result, p_value)=stats.pearsonr(high_cov_x,high_cov_y)
  assert(p_value<0.05)

  (test_result, p_value)=stats.pearsonr(low_cov_x,low_cov_y)
  assert(p_value>=0.05)




###############################################
#############Test GP Utils ################
###############################################
def array(xs):
  return t.VentureArrayUnboxed(np.array(xs),  t.NumberType())

def makeObservations(x,y,ripl):
    xString = genSamples(x)
    ripl.observe(xString, array(y))

def genSamples(x):
    sampleString='(gp (array '
    for i in range(len(x)):
        sampleString+= str(x[i]) + ' '
    sampleString+='))'
    #print(sampleString)
    return sampleString


def f(x):
    return 0.3 + 0.4*x + 0.5*np.sin(2.7*x) + (1.1/(1+x**2))

def f_periodic(x):
    return 5*np.sin(x)

def f_LIN_PER_WN(x):
  return 2*x+ 5*np.sin(3*x)


@in_backend("lite")
def test_gp_inference_known_example():
  r'''
  Inference Quality Smoke Test
--------------------------

  We take a simple data generating process with a clear structure:
  
  :math:`f(x)= 2x + 5 \sin 3x +  \eta` with :math:`\eta \sim \mathcal{N}(0,1)`.
  
  We learn the hyper-parameters for K = LIN + PER + WN and extrapolate. We perform ad-hoc tests (as statistical tests for an extrapolation seem to be too conservative). We use 100 data points for training data.
  '''
  ripl = shortcuts.make_lite_church_prime_ripl()
  n = 100
  x = np.random.uniform(0,20,n)
  y = f_LIN_PER_WN(x) + np.random.normal(0,1,n)
  ripl.bind_foreign_sp("make_gp_part_der",gp_w_der.makeGPSP)
  ripl.assume('make_const_func', VentureFunction(covs.makeConstFunc, [t.NumberType()], covs.constantType))
  ripl.assume('zero', "(apply_function make_const_func 0)")

  ripl.assume("func_plus", covs.makeLiftedAdd(lambda x1, x2: x1 + x2))

  ripl.assume('make_per',VentureFunction(covs.makePeriodic,[t.NumberType(),t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
  ripl.assume('make_linear',VentureFunction(covs.makeLinear,[t.NumberType()], t.AnyType("VentureFunction")))
  ripl.assume('make_noise',VentureFunction(covs.makeNoise,[t.NumberType()], t.AnyType("VentureFunction")))


  ripl.assume('sf','(tag (quote hyper) 0 (uniform_continuous 0 10))')
  ripl.assume('p','(tag (quote hyper) 1  (uniform_continuous 0 10))')
  ripl.assume('l','(tag (quote hyper) 2 (uniform_continuous 0 10))')
  ripl.assume('s','(tag (quote hyper) 3 (uniform_continuous 0 10))')


  ripl.assume('sigma','(tag (quote hyper) 4 (uniform_continuous 0 2 ))')

  ripl.assume('per', "(apply_function make_per sf p l )")
  ripl.assume('wn','(apply_function make_noise sigma  )')
  ripl.assume('lin','(apply_function make_linear s  )')


  ripl.assume('gp',"""(tag (quote model) 0
                        (make_gp_part_der zero  (apply_function func_plus lin (apply_function func_plus per wn  )
                                ))

                             )""")


  makeObservations(x,y,ripl)

  ripl.infer("(mh (quote hyper) one 100)")

  xpost = 25 # were extrapolating quite far out.
  ypost = []
  for i in range(500):
      y = ripl.sample("(gp (array " + str(xpost) + " ))")
      ypost.append(y)
  # ad-hoc tests
  assert(np.std(ypost)<2.5)
  assert(np.std(ypost)>0.5)
  assert(abs(np.mean(ypost)-f_LIN_PER_WN(xpost))<=3)


