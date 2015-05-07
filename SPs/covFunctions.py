
# In[1]:

from venture import shortcuts


import numpy as np

import scipy.spatial.distance as spdist

from venture.lite.function import VentureFunction
from venture.lite.sp import SPType
import venture.lite.value as v




constantType = SPType([v.AnyType()], v.NumberType())
covfunctionType = SPType([v.NumberType(), v.NumberType()], v.NumberType())



def makeConstFunc(c):
  return VentureFunction(lambda _: c, sp_type=constantType)
def array(xs):
  return v.VentureArrayUnboxed(np.array(xs),  v.NumberType())
#### Squared Exponential Covariance Function

# In[6]:

def squared_exponential(sf, l):
  sf = np.exp(sf)
  l = np.exp(l)
  def f(x1,  x2=None):
    if x2 is None: # self-covariances for test cases
        nn,D = x1.shape
        A = np.zeros((nn,1))
    else:    
        A = spdist.cdist(x1/l,x2/l,'sqeuclidean')
    return sf * np.exp(-0.5*A)
  return f
# gradient
def squared_exponential_der_l(sf, l):
  sf = np.exp(sf)
  l = np.exp(l)
  def f(x1, x2):
    A = spdist.cdist(x1/l,x2/l,'sqeuclidean')
    return sf * np.exp(-0.5*A) * A
  return f
def squared_exponential_der_sf(sf, l):
  sf = np.exp(sf)
  l = np.exp(l)
  def f(x1, x2):
    A = spdist.cdist(x1/l,x2/l,'sqeuclidean')
    return  2. * sf * np.exp(-0.5*A)
  return f



squaredExponentialType = SPType([v.NumberType(), v.NumberType()], v.NumberType())
def makeSquaredExponential(sf, l): 
  return VentureFunction(squared_exponential(sf, l), sp_type=squaredExponentialType,derivatives={0:squared_exponential_der_l(sf,l),1:squared_exponential_der_sf(sf,l)},name="SE")


#### Periodic Covariance Function

# In[7]:

def periodic(l,p,sf):
  sf = np.exp(2.*sf)
  p = np.exp(p)
  l = np.exp(l)
  def f(x1, x2=None):
    if x2 is None: # self-covariances for test cases
        nn,D = x1.shape
        A = np.zeros((nn,1))
    else:
        A = np.sqrt(spdist.cdist(x1,x2,'sqeuclidean'))
    A = np.pi*A/p
    A = np.sin(A)/l
    A = A * A
    A = sf *np.exp(-2.*A)
    return A
  return f


# gradient

# In[8]:

def periodic_der_l(l,p,sf):
  sf = np.exp(2.*sf)
  p = np.exp(p)
  l = np.exp(l)
  def f(x1, x2):    
    A = np.sqrt(spdist.cdist(x1,x2,'sqeuclidean'))
    A = np.pi*A/p
    A = np.sin(A)/l
    A = A * A
    A = 4. *sf *np.exp(-2.*A) * A
    return A
  return f
def periodic_der_p(l,p,sf):
  sf = np.exp(2.*sf)
  p = np.exp(p)
  l = np.exp(l)
  def f(x1, x2):
    A = np.sqrt(spdist.cdist(x1,x2,'sqeuclidean'))
    A = np.pi*A/p
    R = np.sin(A)/l
    A = 4 * sf/l * np.exp(-2.*R*R)*R*np.cos(A)*A
    return A
  return f
def periodic_der_sf(l,p,sf):
  sf = np.exp(2.*sf)
  p = np.exp(p)
  l = np.exp(l)
  def f(x1, x2):
    A = np.sqrt(spdist.cdist(x1,x2,'sqeuclidean'))
    A = np.pi*A/p
    A = np.sin(A)/l
    A = A * A
    A = 2. * sf * np.exp(-2.*A)
    return A
  return f

def makePeriodic(l,p,sf):  
  return VentureFunction(periodic(l,p,sf), sp_type=covfunctionType,derivatives={0:periodic_der_l(l,p,sf),1:periodic_der_p(l,p,sf),2:periodic_der_sf(l,p,sf)},name="PER")


#### Linear Covariance Function

# In[9]:

def linear(sf):
  sf = np.exp(sf)
  def f(x1,  x2=None):
    if x2 is None: # self-covariances for test cases
        nn,D = x1.shape
        A = np.dot(x1,x1.T)
    else:
        A = np.dot(x1,x2.T) + 1e-10    # required for numerical accuracy
    return sf * A
  return f
# gradient
def linear_der_sf(sf):
  sf = np.exp(sf)
  def f(x1, x2):
    A = np.dot(x1,x2.T)
    return 2 * sf * A
  return f


def makeLinear(sf): 
  return VentureFunction(linear(sf), sp_type=covfunctionType,derivatives={0:linear_der_sf(sf)},name="LIN")


#### White Noise Covariance Function

# In[10]:

def noise(s):
  sf = np.exp(2.*s)
  def f(x1, x2=None):
    if x2 is None: # self-covariances for test cases
        nn,D = x1.shape
        A = np.zeros((nn,1))
    else:
        tol = 1.e-9                       # Tolerance for declaring two vectors "equal"
        M = spdist.cdist(x1, x2, 'sqeuclidean')
        A = np.zeros_like(M,dtype=np.float)
        A[M < tol] = 1.
    A = s*A
    return A
  return f
# gradient
def noise_der(s):
  sf = np.exp(s)
  def f(x1, x2):
    tol = 1.e-9                       # Tolerance for declaring two vectors "equal"
    M = spdist.cdist(x1, x2, 'sqeuclidean')
    A = np.zeros_like(M,dtype=np.float)
    A[M < tol] = 1.
    A = 2.*s*A
    return A
  return f

def makeNoise(s): 
  return VentureFunction(noise(s), sp_type=covfunctionType,derivatives={0:noise_der(s)},name="WN")


#### Binary Operators

# In[11]:

def lift_binary(op):
  def lifted(f1, f2):
    return lambda *xs: op(f1(*xs), f2(*xs))
  return lifted


liftedBinaryType = SPType([v.AnyType(), v.AnyType()], v.AnyType())

def makeLiftedAdd(op):
  lifted_op = lift_binary(op)
  def wrapped(f1, f2):
    sp_type = f1.sp_type
    assert(f2.sp_type == sp_type)
    der={}
    for i in range(len(f1.stuff['derivatives'])):
        der[i]=f1.stuff['derivatives'][i]
    for j in range(len(f2.stuff['derivatives'])):
        der[i+1+j]=f2.stuff['derivatives'][j]
    return VentureFunction(lifted_op(f1, f2), sp_type=sp_type,derivatives=der,name=f1.stuff['name']+"+"+f2.stuff['name'])
  return VentureFunction(wrapped, sp_type=liftedBinaryType)

def makeLiftedMult(op):
  lifted_op = lift_binary(op)
  def wrapped(f1, f2):
    sp_type = f1.sp_type
    assert(f2.sp_type == sp_type)
    der={}
    for i in range(len(f1.stuff['derivatives'])):
        der[i]= lambda *xs: np.dot(f1.stuff['derivatives'][i](*xs),f2.f(*xs))
    for j in range(len(f2.stuff['derivatives'])):
        der[i+1+j]= lambda *xs: np.dot(f2.stuff['derivatives'][j](*xs),f1.f(*xs))
    return VentureFunction(lifted_op(f1,f2), sp_type=sp_type,derivatives=der,name=f1.stuff['name']+"x"+f2.stuff['name'])
  return VentureFunction(wrapped, sp_type=liftedBinaryType)

