

import random
import numpy as np


from venture.lite.psp import RandomPSP
import scipy
import scipy.misc as sc


class Subset(RandomPSP):

  def isRandom(self): return True
  def simulate(self,args):
    vals = args.operandValues()
    list_of_items= vals[0]
    p = vals[1]
    subset_size = np.random.multinomial(1, p).argmax()+1
    assert subset_size <= len(list_of_items)
    return random.sample(list_of_items,subset_size)


  def logDensity(self,val,args):
    vals = args.operandValues()
    list_of_items= vals[0]
    p = vals[1]
    #import ipdb;ipdb.set_trace()
    return -np.log(desc_product(len(list_of_items), len(vals))) + np.log(p[len(val)-1])


def desc_product(n, k):
  return reduce(lambda x,y: x*y, range(n, n-k, -1), 1)

def log_desc_product(n, k):
  """ This is probably preferable to log(desc_product(n, k)) if n and k are large. """
  return sum(np.log(n-i) for i in range(k))



  
  
  
  
