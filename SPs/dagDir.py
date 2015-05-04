from psp import DeterministicPSP, NullRequestPSP, RandomPSP, TypedPSP
from sp import SP, VentureSPRecord, SPType
import math
import random
from value import BoolType,ArrayUnboxedType,NumberType,MatrixType,IntegerType,AnyType # The type names are metaprogrammed pylint: disable=no-name-in-module
from copy import deepcopy
import numpy as np
from cycleChecker import dagCheck
from DAGutil import getParents,count,getParentConfig,dirichletPrior   
import scipy.special 

class DAG_sampler():
    def __init__(self,dag_scoreVFB,edgeList,samples={}): #ToDo change samples ???
        self.dag_score=dag_scoreVFB.datum
        self.d = dag_scoreVFB.datum.d
        self.edgeList=edgeList
        self.samples=samples        
        self.ranges = [2 for i in range(self.d)]
        
        
    def update_DAG_score(self,data): 
        for i in data.keys():
            if len(self.dag_score.data.shape)>1:
                if i<self.dag_score.data.shape[0]:
 
                    self.dag_score.data[i]=data[i]
            else:
                self.fill_missing_values(i)
                self.dag_score.data[i-1]=data[i]
    
    def latest(self):
        latest = -1
        for key in self.samples:
            if key > latest:
                latest = key
        return self.samples[latest]

    def getMatrix(self,xs):
        if len(self.samples)==0:
            return np.random.rand(3,2)
        else:   
             x2s = self.samples.keys()
             o2s = self.samples.values()
             print("here")
             print(o2s)
             return o2s
    
    def sample(self,*xs): #ToDo stub!!!
        out = self.getMatrix(xs)
        return out
  
            
        
                    
    def logDensity(self,index,args):
        print("ToDo implement log density")
        return 0

class DAGOutputPSP(RandomPSP):
  def __init__(self,dag_score,edgeList):
    self.dag_score= dag_score
    self.edgeList=edgeList

  def makeDAG(self,samples):
    return DAG_sampler(self.dag_score,self.edgeList, samples)


  def simulate(self,args):
    samples = args.spaux
    xs = args.operandValues[0]
    return self.makeDAG(samples).sample(xs)

  
  def logDensity(self,os,args):
    samples = args.spaux
    xs = args.operandValues[0]
    return self.makeDAG(samples).logDensity(xs, os)
  def incorporate(self,os,args):
    samples = args.spaux
    xs = args.operandValues[0]
    for x, o in zip(xs, os):
      samples[x] = o

  def unincorporate(self,_os,args):
    samples = args.spaux
    xs = args.operandValues[0]
    for x in xs:
      del samples[x]
 

DAG_Sampler_Type = SPType([ArrayUnboxedType(NumberType())], ArrayUnboxedType(NumberType()))

class DAGSP(SP):
  def __init__(self,dag,edgeList):
      self.dag = dag
      self.edgeList=edgeList
      output = TypedPSP(DAGOutputPSP(dag,edgeList), DAG_Sampler_Type)
      super(DAGSP, self).__init__(NullRequestPSP(),output) 
      
  def constructSPAux(self): return {}
  def show(self,spaux): return DAG_sampler(self.dag, self.edgeList, spaux)
'''   
class DAGSPAux(object):
  def __init__(self,dag,edgeList):
      self.dag = dag
      self.edgeList=edgeList
  def copy(self):
    dag = DAGSPAux()
    dag.dag = deepcopy(self.dag)
    dag.edgeList = deepcopy(self.edgeList)
    return dag
'''

class MakeDAGOutputPSP(DeterministicPSP):
  def simulate(self,args):
    dag = args.operandValues[0]
    edgeList = args.operandValues[1]    
    return VentureSPRecord(DAGSP(dag,edgeList))

  def childrenCanAAA(self): return True

  def description(self,name=None):
    return "(%s alpha) -> <SP () <number>>\n  DAG Returns a sampler for dag data." % name

''' ToDo
  def logDensityOfCounts(self,samples):
    return self.makeGP(samples).logDensityOfCounts()
   
'''  
makeDAGType = SPType([AnyType("DAG VentureForeignBlob"), AnyType("list of egdge")], DAG_Sampler_Type)
makeDAGSP = SP(NullRequestPSP(), TypedPSP(MakeDAGOutputPSP(), makeDAGType))