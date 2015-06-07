
import random
import numpy as np


from venture.lite.psp import RandomPSP
import scipy
import scipy.misc as sc






class Subset(RandomPSP):

  def isRandom(self): return True
  def simulate(self,args):
    list_of_items= args.operandValues[0]
    subset_size = args.operandValues[1]
    assert subset_size <= len(list_of_items)
    return random.sample(list_of_items,subset_size)


  def logDensity(self,val,args):
    list_of_items= args.operandValues[0]
    subset_size = args.operandValues[1]
    assert subset_size <= len(list_of_items)
    #import ipdb;ipdb.set_trace()
    assert len(val)==subset_size, 'needs to be block sampled with the size of the subset!'
    return np.log(sc.factorial(subset_size)/sc.factorial(len(list_of_items)))







  
  
  
  
