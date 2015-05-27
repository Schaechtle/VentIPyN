
import random
import numpy as np

from venture.lite.sp import SP, SPType
from venture.lite.function import VentureFunction
from venture.lite.types import  AnyType
from venture.lite.psp import RandomPSP,LikelihoodFreePSP,DeterministicPSP,PSP
import scipy
import scipy.misc as sc
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
  return VentureFunction(lifted_plus(f1, f2), sp_type=sp_type,derivatives=der,name="("+f1.stuff['name']+"+"+f2.stuff['name']+")",parameter=f1.stuff['parameter']+f2.stuff['parameter'])



lifted_mult = lift_binary(lambda x1, x2: np.multiply(x1,x2))
def prodKernel(f1, f2):
  sp_type = f1.sp_type
  assert(f2.sp_type == sp_type)
  der={}
  for i in range(len(f1.stuff['derivatives'])):
      der[i]= lambda *xs: np.dot(f1.stuff['derivatives'][i](*xs),f2.f(*xs))
  for j in range(len(f2.stuff['derivatives'])):
      der[i+1+j]= lambda *xs: np.dot(f2.stuff['derivatives'][j](*xs),f1.f(*xs))
  return VentureFunction(lifted_mult(f1,f2), sp_type=sp_type,derivatives=der,name="("+f1.stuff['name']+"x"+f2.stuff['name']+")",parameter=f1.stuff['parameter']+f2.stuff['parameter'])





class Grammar(RandomPSP):
  #def canAbsorb(self, _trace, _appNode, _parentNode): return False
  #def childrenCanAAA(self): return True
  def isRandom(self): return True
  def simulate(self,args):
    covFunctions= args.operandValues[0]
    number_covfunctions= args.operandValues[1].getNumber()+1
    assert (isinstance(number_covfunctions,int) or isinstance(number_covfunctions,float))
    max_number = 0
    list_of_cov_lists=[]

    for item in covFunctions:
          list_of_cov_lists.append(item)
          max_number+= len(item)
    first = True
    for i in range(int(number_covfunctions)):
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
    return K


  def logDensity(self,val,args):
    covFunctions= args.operandValues[0]
    number_operators = args.operandValues[1].getNumber() # number components minus one
    max_number =0
    for item in covFunctions:
         max_number+= len(item)

    if val:
        K_str = val.stuff['name']
        global_flips,local_flips = self.parse_global_vs_linear(K_str)
        if global_flips+local_flips!=number_operators:
            return -float("inf")
        else:
            if number_operators==0:
                return np.log(1./max_number)
            else:
                number_components=number_operators+1
                p_subsets=sc.factorial(number_components)/sc.factorial(max_number) #subsets n! / (n - r)!
                return scipy.stats.binom.logpmf(global_flips,number_operators,0.5) + np.log(p_subsets)
    else:
        raise ValueError('Log density evaluated without value')
    #return number_components * np.log(0.5) + np.log(1./possible_subsets)
  def parse_global_vs_linear(self,K_str):
      global_flips =  K_str.count('+')
      local_flips = K_str.count('x')
      return global_flips,local_flips






  
  
  
  
