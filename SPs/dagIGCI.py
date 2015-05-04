'''
from psp import DeterministicPSP, NullRequestPSP, RandomPSP, TypedPSP
from sp import SP, VentureSPRecord, SPType
import math
import random
from value import BoolType,ArrayUnboxedType,NumberType # The type names are metaprogrammed pylint: disable=no-name-in-module
from copy import deepcopy
import numpy as np
from cycleChecker import dagCheck
from DAGutil import getParents,count,getParentConfig,dirichletPrior   
import scipy.special 

class DAG():
    def __init__(self,d,data=np.array((0))): #ToDo add more stuff from other dag.py 
        self.d = d
   
    
    def probabilityEdge(self,i,j,prior=0.5):
        if prior==0.:
            return 0.
        edge_true = self.scoreAllGraphsIJ(i,j)
        edge_false = self.allScore - self.scoreAllGraphsIJ(i,j)
        return (edge_true*prior)/((edge_true*prior)+(edge_false*(1-prior)))
        
 
class DAGSPAux(object):
  def __init__(self,n):
      self.dag=DAG(n)

  def copy(self):
    dag = DAGSPAux()
    dag.dag = deepcopy(self.dag)
    return dag

class DAGSP(SP):
  def __init__(self,d):
      self.d = n
      output = TypedPSP(DAGOutputPSP(d), SPType([],ArrayUnboxedType(NumberType())))
      super(DAGSP, self).__init__(NullRequestPSP(),output) 
  def constructSPAux(self): return DAGSPAux(self.d)
  def show(self,spaux):
    return {
      'type' : 'dag'
    }


class MakeDAGOutputPSP(DeterministicPSP):
  def simulate(self,args):
    n = args.operandValues[0]
    output = TypedPSP(DAGOutputPSP(n), SPType([],ArrayUnboxedType(NumberType())))
    return VentureSPRecord(DAGSP(n))

  def childrenCanAAA(self): return True

  def description(self,name=None):
    return "(%s alpha) -> <SP () <number>>\n  Chinese Restaurant Process with hyperparameter alpha.  Returns a sampler for the table number." % name

class DAGOutputPSP(RandomPSP):
  def __init__(self,n):
    self.dag= DAG(n)

  def simulate(self,args):
    aux = args.spaux
    
    return aux.dag.sample()

#ToDo: I am pretty sure this is incorrect, since it's just last-in-first-out    
  def incorporate(self,os,args):
    aux = args.spaux
    if len(aux.dag.data[0])==0:
        aux.dag.data[0]=os
    else:
        aux.dag.data.append(os)
    
  def unincorporate(self,_os,args):
    aux = args.spaux
    if len(aux.dag.data)==1:
        aux.dag.data[0]=[]
    else:
        del aux.dag.data[-1]


from scipy import special



def ICGI(x,y,refmeasure="Gaussian",estimator="entropy"):
    f=0
    
    if refmeasure=="Gaussian":
        x = (x - np.mean(x))/np.std(x)
        y = (y - np.mean(y))/np.std(y)
    elif refmeasure=="uniform":
        x = (x - np.min(x))/(np.max(x)-np.min(x))
        y = (y - np.min(y))/(np.max(y)-np.min(y))
    else:
        print("ref measure is not known")

    if estimator=="entropy":
        x= np.sort(x,axis=None)
        y= np.sort(y,axis=None)
        hx = np.array([0.0])
        hy = np.array([0.0])
        for i in range(len(x)-1):
            delta_x = x[i+1]-x[i]
            delta_y = y[i+1]-y[i]
            if delta_x!=0:
                hx = hx + np.log(np.abs(delta_x))
            if delta_y!=0:
                hy = hy + np.log(np.abs(delta_y))
            
        hx = hx / (len(x) - 1) + special.polygamma(0, len(x)) - special.polygamma(0, 1)
        hy = hy / (len(y) - 1) + special.polygamma(0, len(y)) - special.polygamma(0, 1)
        f = hy-hx
    
    elif estimator=="integral":
        a = 0.
        b = 0.
        px= x.argsort(axis=None)
        py= y.argsort(axis=None)
        x_cxy = x[px]
        y_cxy = y[px]

        x_cyx = x[py]
        y_cyx = y[py]
 
    
        for i in range(len(x)-1):
   
            if (x_cxy[i+1]!=x_cxy[i])&(y_cxy[i+1]!=y_cxy[i]):
                a = a + np.log(np.abs((y_cxy[i+1]-y_cxy[i])/(x_cxy[i+1]-x_cxy[i])))
            if (x_cyx[i+1]!=x_cyx[i])&(y_cyx[i+1]!=y_cyx[i]):
                b = b + np.log(np.abs((x_cyx[i+1]-x_cyx[i])/(y_cyx[i+1]-y_cyx[i])))


        f = (a - b)/(len(x)-1)     
    else:
        print("estimator is not known")                          
    return -f

        
'''


