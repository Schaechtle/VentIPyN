from psp import DeterministicPSP, NullRequestPSP, RandomPSP, TypedPSP
from sp import SP, VentureSPRecord, SPType
import math
import scipy.special
import scipy.stats
from utils import simulateCategorical
from value import NumberType, AnyType # The type names are metaprogrammed pylint: disable=no-name-in-module
from copy import deepcopy
import random
import numpy as np
from venture.lite.function import VentureFunction
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



class GPGrammarSPAux(object):
  def __init__(self):
    self.currentStructure = None
    self.se_index = 0
    self.wn_index = 0
    self.lin_index = 0
    self.per_index = 0
    self.currentLength = 0

  def copy(self):
    gpGrammar = GPGrammarSPAux()
    gpGrammar.currentStructure = deepcopy(self.currentStructure)
    gpGrammar.se_index = deepcopy(self.se_index)
    gpGrammar.wn_index = deepcopy(self.wn_index)
    gpGrammar.lin_index = deepcopy(self.lin_index)
    gpGrammar.per_index = deepcopy(self.per_index)
    return gpGrammar

class GPGrammarSP(SP):
  def constructSPAux(self): return GPGrammarSPAux()
  def show(self,spaux):
    return {
      'type' : 'gpGrammar',
      'counts': spaux.currentStructure,
    }

class MakeGPGrammarOutputPSP(DeterministicPSP):
  def simulate(self,args):
    list_max_se = args.operandValues[0]
    list_max_lin = args.operandValues[1]
    list_max_per = args.operandValues[2]
    list_max_wn = args.operandValues[3]
    if len(args.operandValues)>4:
        p_add= args.operandValues[4]
    
    output = TypedPSP(GPGrammarOutputPSP(list_max_se,list_max_lin,list_max_per,list_max_wn,p_add), SPType([], AnyType()))
    return VentureSPRecord(GPGrammarSP(NullRequestPSP(),output))

  def childrenCanAAA(self): return False

  def description(self,name):
    return " Contructs a Grammar for the given lists.  Returns a sampler for the GP structure" % name

class GPGrammarOutputPSP(RandomPSP):
  def __init__(self,list_max_se,list_max_lin,list_max_per,list_max_wn,p_add=0.5):
    self.allCov = list_max_se
    self.WN = list_max_wn
    self.allCov.extend(list_max_wn)
    self.allCov.extend(list_max_lin)
    self.allCov.extend(list_max_per)
    self.p_add=p_add

  def simulate(self,args):
    aux = args.spaux
    #return simulateCategorical(counts,indices) 
    if aux.currentLength<=0:
        return self.WN[0]
  

    number_base_kernels = np.random.randint(1,len(self.allCov)+1)
    random.shuffle(self.allCov)

    for i in  range(number_base_kernels):
        if i==0:
            K =  self.allCov[i]
        else:
            if random.random()< self.p_add:
                K = addKernel(K,self.allCov[i])
            else:
                K = addKernel(K,self.allCov[i])          
    return K


  def logDensity(self,index,args):
    aux = args.spaux
    if index in aux.currentStructure:
      return math.log(aux.currentStructure[index] - self.d) - math.log(self.alpha + aux.numCustomers)
    else:
      return math.log(self.alpha + (aux.numTables * self.d)) - math.log(self.alpha + aux.numCustomers)

  # def gradientOfLogDensity(self, value, args):
  #   aux = args.spaux
  #   if index in aux.currentStructure:
  
  def incorporate(self,value,args):
    aux = args.spaux
    aux.currentStructure = value
    aux.currentLength+=1

  def logDensityOfCounts(self,aux): #ToDo - replace this by categoricals
    term1 = scipy.special.gammaln(self.alpha) - scipy.special.gammaln(self.alpha + aux.numCustomers)
    term2 = aux.numTables + math.log(self.alpha + (aux.numTables * self.d))
    term3 = sum([scipy.special.gammaln(aux.currentStructure[index] - self.d) for index in aux.currentStructure])
    return term1 + term2 + term3

  def enumerateValues(self,args):
    aux = args.spaux
    old_indices = [i for i in aux.currentStructure]
    indices = old_indices + [aux.nextIndex]
    return indices

