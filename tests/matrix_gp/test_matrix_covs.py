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
r'''
Smoke Tests
************************
Smoke tests for GP covariande functions
==========================
The test functions do not take any input. We fix both, data vectors and hyper-parameters. The expected results were computed by hand and with MATLAB.

We are using :math:`\mathbf{x} = [1.3, -2, 0]^T`
and  :math:`\mathbf{y} = [1.3, -3.2]^T` to test both :math:`\mathbf{K}(\mathbf{x},\mathbf{x})` and :math:`\mathbf{K}(\mathbf{x},\mathbf{y})`. The hyper-parameters are defined as follows:

:math:`\sigma = 2.1`

:math:`\ell= 1.8`

:math:`p = 0.9`

:math:`\alpha= 0.8`

:math:`\sigma = 2.1`
'''
import numpy as np
import numpy.linalg as npla
from scipy import stats
from nose.tools import assert_almost_equal

from venture.test.config import in_backend

import imp

covs = imp.load_source('rationalQuadratic', '/home/ulli/VentIPyN/Experiments/covFunctions_noLog.py')



def apply_cov(f):
  x = np.array([[1.3,-2,0]]).T
  y = np.array([[1.3,-3.2]]).T

  return f(x,x),f(x,y)

@in_backend("lite")
def test_noise():
  '''
  Tests the noise covariance 
  WN :math:`=\sigma^2 \delta_{x,x^\prime}`
  '''
  f  = covs.noise(2.1)
  cov_train,cov_test = apply_cov(f)
  expect_cov_train = 4.41*np.eye(3)
  expect_cov_test = np.zeros((3,2))
  expect_cov_test[0][0]= 4.41
  np.testing.assert_almost_equal(cov_train, expect_cov_train)
  np.testing.assert_almost_equal(cov_test, expect_cov_test)

@in_backend("lite")
def test_rationalQuadratic():
  r'''
  Tests the rational quadratic covariance 
  RQ :math:`=\sigma^2 \bigg(1 + \frac{(x - x^\prime)^2}{2 \alpha \ell^2} \bigg)^{-\alpha}`
  '''
  l = 1.8
  sigma = 2.1
  alpha = 0.8
  f  = covs.rationalQuadratic(l,sigma,alpha)
  cov_train,cov_test = apply_cov(f)
  expect_cov_train = np.array([[4.4100, 1.7835,  3.5189],[1.7835,4.4100, 2.7909],[ 3.5189, 2.7909,4.4100]])
  expect_cov_test =  np.array([[4.4100,  1.2355],[1.7835,  3.6247],[ 3.5189,  1.8434]])
  np.testing.assert_almost_equal(cov_train, expect_cov_train, decimal=4)
  np.testing.assert_almost_equal(cov_test, expect_cov_test, decimal=4)

@in_backend("lite")
def test_squaredExponential():
  r'''
  Tests the squared-exponential covariance
  SE :math:`= \sigma^2 \exp{-\frac{(x-x^\prime)^2}{2\ell^2}}`
  '''
  l = 1.8
  sigma = 2.1
  f  = covs.squared_exponential(sigma,l)
  cov_train,cov_test = apply_cov(f)
  expect_cov_train = np.array([[4.4100,  0.8215,   3.3976],[ 0.8215,4.4100,  2.3788],[  3.3976,  2.3788,4.4100]])
  expect_cov_test =  np.array([[4.4100,  0.1938],[ 0.8215,   3.5313],[ 3.3976,   0.9081]])
  np.testing.assert_almost_equal(cov_train, expect_cov_train, decimal=4)
  np.testing.assert_almost_equal(cov_test, expect_cov_test, decimal=4)

@in_backend("lite")
def test_const():
  sigma = 2.1
  f  = covs.const(sigma)
  cov_train,cov_test = apply_cov(f)
  expect_cov_train = 4.41*np.ones((3,3))
  expect_cov_test = 4.41*np.ones((3,2))
  np.testing.assert_almost_equal(cov_train, expect_cov_train)
  np.testing.assert_almost_equal(cov_test, expect_cov_test)

@in_backend("lite")
def test_periodic():
  r'''
  Tests the periodic covariance
  PER :math:`= \sigma^2 \exp \bigg( \frac{2 \sin^2 ( \pi (x - x^\prime)/p}{\ell^2} \bigg)`
  '''
  l = 1.8
  sigma = 2.1
  p = 0.9
  f  = covs.periodic(l,p,sigma)
  cov_train,cov_test = apply_cov(f)
  expect_cov_train =np.array([[ 4.4100      ,  2.7757,  2.4235],
       [ 2.7757,  4.4100      ,  3.41722163],
       [ 2.4235,  3.4172,  4.4100      ]]) #ToDo Double check, output was not calculated by hand, but compared with GPML
       # toolbox, since I have found 3 different for versions of the periodic kernel, all similar but none equal to matlab
      #implementation. I suspect a change in a scaling factor

  expect_cov_test = np.array([[   4.4100,    4.4100],
    [2.7757,    2.7757],
    [2.4235,    2.4235 ]])
  np.testing.assert_almost_equal(cov_train, expect_cov_train, decimal=4)
  np.testing.assert_almost_equal(cov_test, expect_cov_test, decimal=4)

@in_backend("lite")
def test_linear():
  r'''
  Tests the linear covariance
  LIN  :math:`=\sigma^2 (x x^\prime)`
  '''
  sigma = 2.1
  f  = covs.linear(sigma)
  cov_train,cov_test = apply_cov(f)
  expect_cov_train = np.zeros((3,3))
  expect_cov_train[0][0]= 7.4529
  expect_cov_train[0][1]=  11.4660
  expect_cov_train[1][0]=  11.4660
  expect_cov_train[1][1]=  17.64
  expect_cov_test = np.zeros((3,2))
  expect_cov_test[0][0]= 7.4529
  expect_cov_test[0][1]= 18.3456
  expect_cov_test[1][0]=  11.4660
  expect_cov_test[1][1]=  28.224
  np.testing.assert_almost_equal(cov_train, expect_cov_train, decimal=4)
  np.testing.assert_almost_equal(cov_test, expect_cov_test, decimal=4)


