
# In[1]:

from venture import shortcuts


import numpy as np

import scipy.spatial.distance as spdist

from venture.lite.function import VentureFunction
from venture.lite.sp import SPType
import venture.lite.types as t
import venture.lite.value as v




constantType = SPType([t.AnyType()], t.NumberType())
covfunctionType = SPType([t.NumberType(), t.NumberType()], t.NumberType())



def makeConstFunc(c):
  return VentureFunction(lambda _: c, sp_type=constantType,derivatives={0:lambda _: 0})
def array(xs):
  return t.VentureArrayUnboxed(np.array(xs),  t.NumberType())

def const(sf2):         # s2
    def f(x1,  x2=None):
        if x2 is None: # self-covariances for test cases           # self covariances for the test cases
            nn,D = x1.shape
            A = sf2**2 * np.ones((D,D))
        else:            # compute covariance matix for dataset x
            n,D = x1.shape
            A = sf2**2 * np.ones((x1.shape[0],x2.shape[0]))
        return A
    return f


def const_der(sf2):
    def f(x1,  x2=None):
        n,D = x1.shape
        A = sf2 * np.ones((n,n))
        return 2. * A
    return f




def makeConst(sf):
  return VentureFunction(const(sf), sp_type=covfunctionType,name="C",derivatives={0:const_der(sf)},parameter=[sf])


def squared_exponential(sf, l):
 def f(x1,  x2=None):
    if x2 is None: # self-covariances for test cases
        nn,D = x1.shape
        A = np.zeros((nn,1))
    else:    
        A = spdist.cdist(x1/l,x2/l,'sqeuclidean')
    return sf**2 * np.exp(-0.5*A)
 return f

# gradient
def squared_exponential_der_l(sf, l):
  def f(x1, x2):
    A = spdist.cdist(x1/l,x2/l,'sqeuclidean')
    return sf**2 * np.exp(-0.5*A) * A
  return f
def squared_exponential_der_sf(sf, l):
  def f(x1, x2):
    A = spdist.cdist(x1/l,x2/l,'sqeuclidean')
    return  2. * sf * np.exp(-0.5*A)
  return f



squaredExponentialType = SPType([t.NumberType(), t.NumberType()], t.NumberType())
def makeSquaredExponential(sf, l): 
  return VentureFunction(squared_exponential(sf, l), sp_type=squaredExponentialType,derivatives={0:squared_exponential_der_l(sf,l),1:squared_exponential_der_sf(sf,l)},name="SE",parameter=[sf,l])


#### Periodic Covariance Function

# In[7]:

def periodic(l,p,sf):
  def f(x1, x2=None):
    if x2 is None: # self-covariances for test cases
        nn,D = x1.shape
        A = np.zeros((nn,1))
    else:
        A = np.sqrt(spdist.cdist(x1,x2,'sqeuclidean'))
    #import ipdb;ipdb.set_trace()
    A = np.pi*A/p
    A = np.sin(A)/(l)
    A = A*A
    A = sf**2 *np.exp(-2.*A)
    return A
  return f


# gradient

# In[8]:

def periodic_der_l(l,p,sf):
  sf = 2.*sf
  def f(x1, x2):
    A = np.sqrt(spdist.cdist(x1,x2,'sqeuclidean'))
    A = np.pi*A/p
    A = np.sin(A)/l
    A = A * A
    A = 4. *sf *np.exp(-2.*A) * A
    return A
  return f
def periodic_der_p(l,p,sf):
  sf = 2.*sf
  def f(x1, x2):
    A = np.sqrt(spdist.cdist(x1,x2,'sqeuclidean'))
    A = np.pi*A/p
    R = np.sin(A)/l
    A = 4 * sf/l * np.exp(-2.*R*R)*R*np.cos(A)*A
    return A
  return f
def periodic_der_sf(l,p,sf):
  sf = 2.*sf
  def f(x1, x2):
    A = np.sqrt(spdist.cdist(x1,x2,'sqeuclidean'))
    A = np.pi*A/p
    A = np.sin(A)/l
    A = A * A
    A = 2. * sf * np.exp(-2.*A)
    return A
  return f

def makePeriodic(l,p,sf):  
  return VentureFunction(periodic(l,p,sf), sp_type=covfunctionType,derivatives={0:periodic_der_l(l,p,sf),1:periodic_der_p(l,p,sf),2:periodic_der_sf(l,p,sf)},name="PER",parameter=[l,p,sf])


#### Linear Covariance Function

# In[9]:

def linear(sf): # ToDo
  def f(x1,  x2=None):
    if x2 is None: # self-covariances for test cases
        nn,D = x1.shape
        A = np.dot(x1,x1.T)
    else:
        A =  np.dot(x1,x2.T) + 1e-10    # required for numerical accuracy
    return sf**2* A
  return f
# gradient
def linear_der_sf(sf):
  def f(x1, x2):
    A = np.dot(x1,x2.T)
    return 2 * sf * A
  return f


def makeLinear(sf): 
  return VentureFunction(linear(sf), sp_type=covfunctionType,derivatives={0:linear_der_sf(sf)},name="LIN",parameter=[sf])


#### White Noise Covariance Function

# In[10]:

def noise(s):
  def f(x1, x2=None):
    if x2 is None: # self-covariances for test cases
        nn,D = x1.shape
        A = np.zeros((nn,1))
    else:
        tol = 1.e-9                       # Tolerance for declaring two vectors "equal"
        M = spdist.cdist(x1, x2, 'sqeuclidean')
        A = np.zeros_like(M,dtype=np.float)
        A[M < tol] = 1.
    A = s**2 *A
    return A
  return f
# gradient
def noise_der(s):
  def f(x1, x2):
    tol = 1.e-9                       # Tolerance for declaring two vectors "equal"
    M = spdist.cdist(x1, x2, 'sqeuclidean')
    A = np.zeros_like(M,dtype=np.float)
    A[M < tol] = 1.
    A = 2.*s*A
    return A
  return f

def makeNoise(s): 
  return VentureFunction(noise(s), sp_type=covfunctionType,derivatives={0:noise_der(s)},name="WN",parameter=[s])


def rationalQuadratic(ell,sf2,alpha):
    def f(x1, x2=None):
        if x2 is None:           # self covariances for the test cases
            nn,D = x1.shape
            D2 = np.zeros((nn,1))
        else:             # compute covariance between data sets x and z
            D2 = spdist.cdist(x1/ell, x2/ell, 'sqeuclidean')
        A = sf2**2 * ( ( 1.0 + 0.5*D2/alpha )**(-alpha) )
        return A
    return f

def rationalQuadratic_der_l(ell,sf2,alpha):
    def f(x1, x2=None):
        D2 = spdist.cdist(x1/ell, x2/ell, 'sqeuclidean')
        A = sf2 * ( 1.0 + 0.5*D2/alpha )**(-alpha-1) * D2
        return A
    return f

def rationalQuadratic_der_sf2(ell,sf2,alpha):
    def f(x1, x2=None):
        D2 = spdist.cdist(x1/ell, x2/ell, 'sqeuclidean')
        A = 2.* sf2 * ( ( 1.0 + 0.5*D2/alpha )**(-alpha) )
        return A
    return f

def rationalQuadratic_der_alpha(ell,sf2,alpha):
    def f(x1, x2=None):
        D2 = spdist.cdist(x1/ell, x2/ell, 'sqeuclidean')
        K = ( 1.0 + 0.5*D2/alpha )
        A = sf2 * K**(-alpha) * (0.5*D2/K - alpha*np.log(K) )
        return A
    return f

def makeRQ(l,sf,alpha):
  return VentureFunction(rationalQuadratic(l,sf,alpha), sp_type=covfunctionType,derivatives={0:rationalQuadratic_der_l(l,sf,alpha),1:rationalQuadratic_der_sf2(l,sf,alpha),2:rationalQuadratic_der_alpha(l,sf,alpha)},name="RQ",parameter=[l,sf,alpha])

def changePoint(steepness,center):
    def f(x1,x2=None):
        sumx1 = x1-center
        #import  ipdb;ipdb.set_trace()
        if x2 is None:
            n,D = x1.shape
            sx = 1./(1+np.exp(-x1))
            sz=sx
            sumz=sumx1
            A = np.dot(sx,sz.T)
            return A
        x1= sumx1*steepness
        sumx2 = x2-center
        x2= sumx2*steepness
        sx = 1./(1+np.exp(-x1))
        sz = 1./(1+np.exp(-x2)) # ToDo: why do I need to index here???

        A = np.dot(sx,sz.T)	
        return A
    return f
'''

def sig_func(x,c,s):
    return 0.5 * (1+ np.tanh((x-c)/s))

def sig_func2(x,c,s):
    return 0.5 * (1+ np.tanh((x-c)/s))
def changePoint(c,s):
    def f(x1,x2):
        #sx = 1./(1+np.exp(-x1)) -center
        #sz = 1./(1+np.exp(-x2)) -center# ToDo: why do I need to index here???
        a =  sig_func(x1,c,s) * sig_func(x2,c,s).T
        return 1 -  a
    return f

def InvChangePoint(c,s):
    def f(x1,x2):
        #sx = 1./(1+np.exp(-x1)) -center
        #sz = 1./(1+np.exp(-x2)) -center# ToDo: why do I need to index here???
        a =  sig_func2(x1,c,s) * sig_func2(x2,c,s).T
        return a
    return f
def makeINVCP(log_steepness,log_center):
   return VentureFunction(InvChangePoint(log_steepness,log_center), sp_type=covfunctionType,name="CP",derivatives={},parameter=[log_steepness,log_center])

'''


def makeCP(log_steepness,log_center):
   return VentureFunction(changePoint(log_steepness,log_center), sp_type=covfunctionType,name="CP",derivatives={},parameter=[log_steepness,log_center])


#### Binary Operators


def lift_binary(op):
  def lifted(f1, f2):
    return lambda *xs: op(f1(*xs), f2(*xs))
  return lifted




liftedBinaryType = SPType([t.AnyType(), t.AnyType()], t.AnyType())

def makeLiftedAdd(op):
  lifted_op = lift_binary(op)
  def wrapped(f1, f2):
    sp_type = f1.sp_type
    assert(f2.sp_type == sp_type)
    der={}
    i = 0 #TODO delete this, only here for stubby covfunctions where der is not yet implemented
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
    i = 0 #TODO delete this, only here for stubby covfunctions where der is not yet implemented
    for i in range(len(f1.stuff['derivatives'])):
        der[i]= lambda *xs: np.dot(f1.stuff['derivatives'][i](*xs),f2.f(*xs))
    for j in range(len(f2.stuff['derivatives'])):
        der[i+1+j]= lambda *xs: np.dot(f2.stuff['derivatives'][j](*xs),f1.f(*xs))
    return VentureFunction(lifted_op(f1,f2), sp_type=sp_type,derivatives=der,name=f1.stuff['name']+"x"+f2.stuff['name'])
  return VentureFunction(wrapped, sp_type=liftedBinaryType)

def lift_change_point(op):
  def lifted(f1, f2):
    return lambda x1,x2: op(f1(x1,x2), f2(x1,x2))
  return lifted

'''
def makeCP(op):
  lifted_op = lift_binary(op)
  def wrapped(f1, f2):
    sp_type = f1.sp_type
    assert(f2.sp_type == sp_type)
    der={}
    i = 0 #TODO delete this, only here for stubby covfunctions where der is not yet implemented
    return VentureFunction(lift_change_point(f1,f2), sp_type=sp_type,derivatives=der,name=f1.stuff['name']+"x"+f2.stuff['name'])
  return VentureFunction(wrapped, sp_type=liftedBinaryType)
'''
