

import random
import numpy as np


from venture.lite.psp import RandomPSP
import scipy
import scipy.misc as sc



class Subset(RandomPSP):

  def isRandom(self): return True
  def simulate(self,args):
    list_of_items= args.operandValues[0]
    p = args.operandValues[1]
    subset_size = np.random.multinomial(1, p).argmax()+1
    assert subset_size <= len(list_of_items)
    return random.sample(list_of_items,subset_size)


  def logDensity(self,val,args):
    list_of_items= args.operandValues[0]
    p = args.operandValues[1]
    #import ipdb;ipdb.set_trace()
    return np.log(sc.factorial(len(val))/sc.factorial(len(list_of_items))) +np.log(p[len(val)-1])





  
  
  
  
