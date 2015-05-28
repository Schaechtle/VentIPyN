
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
    p = args.operandValues[1].getSimplex()
    number_covfunctions=np.random.multinomial(1, p).argmax()+1
    max_number = 0
    list_of_cov_lists=[]
    for item in covFunctions:
          list_of_cov_lists.append(item)
          max_number+= len(item)
    first = True
    assert max_number == len(p)
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
    p = args.operandValues[1].getSimplex()
    if val:
        K_str = val.stuff['name']
        global_flips,local_flips = self.parse_global_vs_linear(K_str)
        for i in range(len(p)):
            if (global_flips+local_flips)==i:
                p_n = p[i]
                if i==0:
                    p_choose_kernel = 1./len(p)
                    return np.log(p_choose_kernel) + np.log(p_n)
                break
        p_subsets=sc.factorial(i+1)/sc.factorial(len(p)) #subsets n! / (n - r)!
        log_dens =np.log(p_subsets) + np.log(p_n)
        if log_dens <=0:
            return log_dens
        else:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return float("-inf")
    else:
        raise ValueError('Log density evaluated without value')

  def parse_global_vs_linear(self,K_str):
      global_flips =  K_str.count('+')
      local_flips = K_str.count('x')
      return global_flips,local_flips






  
  
  
  
