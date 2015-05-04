from venture.lite.discrete import  DiscretePSP
import random
import numpy as np
from samba.dcerpc.atsvc import First

class EdgePrior(DiscretePSP):

  def simulate(self,args):
    if  len(args.operandValues)<3:
        prior = 0.5
    elif len(args.operandValues)<4:
        prior = args.operandValues[2]
    elif len(args.operandValues)<5:
        if self.breaks_ordering(args.operandValues[0] ,args.operandValues[1],args.operandValues[3]): # check if it obeys causal ordering if given
            prior =0. 
        else:
            prior = args.operandValues[2]       
    else:
        if self.breaks_ordering(args.operandValues[4]  ,args.operandValues[5] ,args.operandValues[3],True): # check if it obeys causal ordering if given
            prior =0. 
        else:
            prior = args.operandValues[2]      

    return random.random() < prior

  def logDensity(self,val,args):
    if  len(args.operandValues)<3:
        prior = 0.5
    elif len(args.operandValues)<4:
        prior = args.operandValues[2]
    elif len(args.operandValues)<5:
        if self.breaks_ordering(args.operandValues[0] ,args.operandValues[1],args.operandValues[3]): # check if it obeys causal ordering if given
            prior =0. 
        else:
            prior = args.operandValues[2]       
    else:
        if self.breaks_ordering(args.operandValues[4]  ,args.operandValues[5] ,args.operandValues[3],True): # check if it obeys causal ordering if given
            prior =0. 
        else:
            prior = args.operandValues[2]     
    return np.log(prior)
     

  def gradientOfLogDensity(self, val, args):
    if  len(args.operandValues)<3:
        p = 0.5
    elif len(args.operandValues)<4:
        p = args.operandValues[2]
    elif len(args.operandValues)<5:
        if self.breaks_ordering(args.operandValues[0] ,args.operandValues[1],args.operandValues[3]): # check if it obeys causal ordering if given
            p =0. 
        else:
            p = args.operandValues[2]       
    else:
        if self.breaks_ordering(args.operandValues[4]  ,args.operandValues[5] ,args.operandValues[3],True): # check if it obeys causal ordering if given
            p =0. 
        else:
            p = args.operandValues[2]   
    deriv = 1/p if val else -1 / (1 - p)
    return (0, [deriv])

  def enumerateValues(self,args):
    if  len(args.operandValues)<3:
        p = 0.5
    elif len(args.operandValues)<4:
        p = args.operandValues[2]
    elif len(args.operandValues)<5:
        if self.breaks_ordering(args.operandValues[0] ,args.operandValues[1],args.operandValues[3]): # check if it obeys causal ordering if given
            p =0. 
        else:
            p = args.operandValues[2]       
    else:
        if self.breaks_ordering(args.operandValues[4]  ,args.operandValues[5] ,args.operandValues[3],True): # check if it obeys causal ordering if given
            p =0. 
        else:
            p = args.operandValues[2]  
    if p == 1: return [True]
    elif p == 0: return [False]
    else: return [True,False]

  def description(self,name):
    return "  (%s p) returns true given the score of all graphs" % name

  def breaks_ordering(self,i,j,ordering,crp_order=False):
    if crp_order:
        for index in range(len(ordering)-1):
            if ordering[index]==(j-1):
                return True
            elif ordering[index]==(i-1):
                if ordering[index+1]==(j-1):
                    return False
        return True
          
    else:
        first_element_seen = False
        for item in ordering:
            if not first_element_seen:
                if item==j:
                    return True
                elif item==i:
                    first_element_seen=True
            else:
                if item==j:
                  return False
              
                  
