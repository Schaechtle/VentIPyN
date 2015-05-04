from discrete import  DiscretePSP
import random
import numpy as np
from samba.dcerpc.atsvc import First
from value import NumberType
class Edge(DiscretePSP):
  def canEnumerate(self):
    return True
  def simulate(self,args):
    dag = args.operandValues[0].datum
    if  len(args.operandValues)<4:
        prior = 0.5
    elif len(args.operandValues)<5:
        prior = args.operandValues[3]
    elif len(args.operandValues)<6:
        if self.breaks_ordering(args.operandValues[1] ,args.operandValues[2],args.operandValues[4]): # check if it obeys causal ordering if given
            prior =0. 
        else:
            prior = args.operandValues[3]
    elif len(args.operandValues)<7:
        if self.breaks_block(args.operandValues[4] ,args.operandValues[5]): # check if it obeys causal ordering if given
            prior =0. 
        else:
            prior = args.operandValues[3]
    
    else:
        if self.breaks_ordering(args.operandValues[5]  ,args.operandValues[6] ,args.operandValues[4],True): # check if it obeys causal ordering if given
            prior =0. 
        else:
            prior = args.operandValues[3]          
    if len(dag.data.shape)>1:
        p = dag.probabilityEdge(args.operandValues[1] ,args.operandValues[2],prior) 
    else:
        p = prior
    return random.random() < p

  def logDensity(self,val,args):
    if  len(args.operandValues)<4:
        prior = 0.5
    elif len(args.operandValues)<5:
        prior = args.operandValues[3]
    elif len(args.operandValues)<6:
        if self.breaks_ordering(args.operandValues[1] ,args.operandValues[2],args.operandValues[4]): # check if it obeys causal ordering if given
            prior =0. 
        else:
            prior = args.operandValues[3]
    elif len(args.operandValues)<7:
        if self.breaks_block(args.operandValues[4] ,args.operandValues[5]): # check if it obeys causal ordering if given
            prior =0. 
        else:
            prior = args.operandValues[3]
    
    else:
        if self.breaks_ordering(args.operandValues[5]  ,args.operandValues[6] ,args.operandValues[4],True): # check if it obeys causal ordering if given
            prior =0. 
        else:
            prior = args.operandValues[3]              
    dag = args.operandValues[0].datum
    if len(dag.data.shape)>1:
        return np.log(dag.probabilityEdge(args.operandValues[1] ,args.operandValues[2],prior))
    else:
        return np.log(prior)
     


  def gradientOfLogDensity(self, val, args):
    if  len(args.operandValues)<4:
        prior = 0.5
    elif len(args.operandValues)<5:
        prior = args.operandValues[3]
    elif len(args.operandValues)<6:
        if self.breaks_ordering(args.operandValues[1] ,args.operandValues[2],args.operandValues[4]): # check if it obeys causal ordering if given
            prior =0. 
        else:
            prior = args.operandValues[3]
    elif len(args.operandValues)<7:
        if self.breaks_block(args.operandValues[4] ,args.operandValues[5]): # check if it obeys causal ordering if given
            prior =0. 
        else:
            prior = args.operandValues[3]
    
    else:
        if self.breaks_ordering(args.operandValues[5]  ,args.operandValues[6] ,args.operandValues[4],True): # check if it obeys causal ordering if given
            prior =0. 
        else:
            prior = args.operandValues[3]              
    dag = args.operandValues[0].datum
    if len(dag.data.shape)>1:
        p= dag.probabilityEdge(args.operandValues[1] ,args.operandValues[2],prior)
    else:
        p = prior
    deriv = 1/p if val else -1 / (1 - p)
    return (0, [deriv])

  def enumerateValues(self,args):
    if  len(args.operandValues)<4:
        prior = 0.5
    elif len(args.operandValues)<5:
        prior = args.operandValues[3]
    elif len(args.operandValues)<6:
        if self.breaks_ordering(args.operandValues[1] ,args.operandValues[2],args.operandValues[4]): # check if it obeys causal ordering if given
            prior =0. 
        else:
            prior = args.operandValues[3]
    elif len(args.operandValues)<7:
        if self.breaks_block(args.operandValues[4] ,args.operandValues[5]): # check if it obeys causal ordering if given
            prior =0. 
        else:
            prior = args.operandValues[3]
    
    else:
        if self.breaks_ordering(args.operandValues[5]  ,args.operandValues[6] ,args.operandValues[4],True): # check if it obeys causal ordering if given
            prior =0. 
        else:
            prior = args.operandValues[3]              
    dag = args.operandValues[0].datum
    if len(dag.data.shape)>1:
        p= dag.probabilityEdge(args.operandValues[1] ,args.operandValues[2],prior)
    else:
        p = prior
    if p == 1: return [True]
    elif p == 0: return [False]
    else: return [True,False]

  def description(self,name):
    return "  (%s p) returns true given the score of all graphs" % name

  def breaks_block(self,i,j):
      if i==j:
          return True
      else:
          return False
  
  def breaks_ordering(self,i,j,ordering,crp_order=False):
      if crp_order:
          cluster_start=1
      else:
          cluster_start=0
      first_element_seen = False
      for item in ordering.getArray(NumberType()):
          item+=cluster_start
          if not first_element_seen:
              if item==j:
                  return True
              elif item==i:
                  first_element_seen=True
          else:
              if item==j:
                  return False
