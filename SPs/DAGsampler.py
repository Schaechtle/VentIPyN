from discrete import  DiscretePSP
import random
import numpy as np

class Edge(DiscretePSP):

  def simulate(self,args):
    dag = args.operandValues[0].datum 
    if len(dag.data.shape)>1:
        p = dag.probabilityEdge(args.operandValues[1] ,args.operandValues[2]) 
    else:
        p = 0.5
    return random.random() < p

  def logDensity(self,val,args):
    dag = args.operandValues[0].datum 
    return np.log(dag.scoreAllGraphsIJ(args.operandValues[1] ,args.operandValues[2]))

  def gradientOfLogDensity(self, val, args):
    dag = args.operandValues[0].datum 
    p = dag.probabilityEdge(args.operandValues[1] ,args.operandValues[2]) 
    deriv = 1/p if val else -1 / (1 - p)
    return (0, [deriv])

  def enumerateValues(self,args):
    dag = args.operandValues[0].datum 
    p = dag.probabilityEdge(args.operandValues[1] ,args.operandValues[2]) 
    if p == 1: return [True]
    elif p == 0: return [False]
    else: return [True,False]

  def description(self,name):
    return "  (%s p) returns true given the score of all graphs" % name

