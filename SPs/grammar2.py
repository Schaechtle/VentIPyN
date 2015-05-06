from discrete import  DiscretePSP
import random
import numpy as np
from samba.dcerpc.atsvc import First
from utils import simulateCategorical
from sp import SP, SPType
from venture.lite.function import VentureFunction
from value import AnyType
from psp import RandomPSP
def lift_binary(op):
  def lifted(f1, f2):
    return lambda *xs: op(f1(*xs), f2(*xs))
  return lifted


liftedBinaryType = SPType([AnyType(), AnyType()], AnyType())


lifted_plus = lift_binary(lambda x1, x2: x1+x2)
def addKernel(f1, f2):
  sp_type = f1.sp_type
  assert(f2.sp_type == sp_type)
  der={}
  for i in range(len(f1.stuff['derivatives'])):
      der[i]=f1.stuff['derivatives'][i]
  for j in range(len(f2.stuff['derivatives'])):
      der[i+1+j]=f2.stuff['derivatives'][j]
  return VentureFunction(lifted_plus(f1, f2), sp_type=sp_type,derivatives=der,name=f1.stuff['name']+"+"+f2.stuff['name'])



lifted_mult = lift_binary(lambda x1, x2: np.multiply(x1,x2))
def prodKernel(f1, f2):
  sp_type = f1.sp_type
  assert(f2.sp_type == sp_type)
  der={}
  for i in range(len(f1.stuff['derivatives'])):
      der[i]= lambda *xs: np.dot(f1.stuff['derivatives'][i](*xs),f2.f(*xs))
  for j in range(len(f2.stuff['derivatives'])):
      der[i+1+j]= lambda *xs: np.dot(f2.stuff['derivatives'][j](*xs),f1.f(*xs))
  return VentureFunction(lifted_mult(f1,f2), sp_type=sp_type,derivatives=der,name=f1.stuff['name']+"x"+f2.stuff['name'])





class Grammar(RandomPSP):
  def canAbsorb(self, _trace, _appNode, _parentNode): return False
  def simulate(self,args):
    list_of_cov_lists=[]
    number_base_kernels=0  
    for item in args.operandValues[0]:
          list_of_cov_lists.append(item)
          number_base_kernels+=len(item)
    covPrior = args.operandValues[1]
    global_structure_prior = args.operandValues[2]
    number_base_kernels = np.random.randint(1,number_base_kernels+1)
    for i in  range(number_base_kernels):
        cov_index = simulateCategorical(covPrior)
        if i ==0:
            K=list_of_cov_lists[cov_index].pop()
        else:
            # get global-prior determining mult, add
            if random.random()<global_structure_prior:
                K =addKernel(K, list_of_cov_lists[cov_index].pop())
            else:
                K =prodKernel(K, list_of_cov_lists[cov_index].pop())
        if not list_of_cov_lists[cov_index]:
                list_of_cov_lists.pop(cov_index)
                covPrior= np.delete(covPrior,cov_index)
    return K
  
  
  
  