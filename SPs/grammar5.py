from venture.lite.discrete import  DiscretePSP
import random
import numpy as np
from samba.dcerpc.atsvc import First
from venture.lite.utils import simulateCategorical
from venture.lite.sp import SP, SPType
from venture.lite.function import VentureFunction
from venture.lite.value import AnyType
from venture.lite.psp import RandomPSP
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
    covFunctions= args.operandValues[0]
    max_number = 0
    list_of_cov_lists=[]

    for item in covFunctions:
          list_of_cov_lists.append(item)
          max_number+= len(item)
    first = True
    while (True):
        cov_index = np.random.randint(0,len(list_of_cov_lists))
        if first:
            K=list_of_cov_lists[cov_index].pop()
            first=False
        else:

            if random.random()<0.5:
                K =addKernel(K, list_of_cov_lists[cov_index].pop())
            else:
                K =prodKernel(K, list_of_cov_lists[cov_index].pop())
        if not list_of_cov_lists[cov_index]:
                list_of_cov_lists.pop(cov_index)
        if (random.random()<0.5) or (not (list_of_cov_lists)): return K
    return K
  
  
  
  
