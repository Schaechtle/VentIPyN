from venture.lite.psp import DeterministicPSP, NullRequestPSP, RandomPSP, TypedPSP
from venture.lite.sp import SP, VentureSPRecord, SPType
import math
import random
from venture.lite.value import BoolType,ArrayUnboxedType,NumberType # The type names are metaprogrammed pylint: disable=no-name-in-module
from copy import deepcopy
import numpy as np
from cycleChecker import dagCheck
from DAGutil import getParents,count,getParentConfig,dirichletPrior   
import scipy.special 

class DAG():
    def __init__(self,d,data=np.array((0))): #ToDo add more stuff from other dag.py 
        self.d = d
        self.ranges = [2 for i in range(self.d)]
        self.allDAGbinary = self.allDAGs(self.d)
        self.lookUpScore={}
        self.lookUpScoreIJ={}
        self.logScoreDict={}
        self.data = data
        self.alpha =0.5 #
        if data.any():
            self.allScore = self.scoreAllGraphs()
           
        else:
            self.uniform_score = 1./len(self.allDAGbinary)
            for key in self.allDAGbinary:
                self.logScoreDict[key]=self.uniform_score
                
            self.allScore = 1
    def get_logScore(self,binRep):
        return self.logScoreDict[binRep]   
        
        
    def get_DAG_with_Edge(self,i,j):
        dags = []
        for dag in self.allDAGbinary:
            index = self.computeNegIndex(i, j, self.d) #ToDo, precompute key-value matrix

            if index<len(dag):
                if dag[-index]=="1":
                    dags.append(dag)
        return dags
    
    def scoreAllGraphs(self):
        s=0.
        for binRep in self.allDAGbinary:
            dag  = self.bin2mat(binRep,self.d)
            currentScore=self.scoreG(dag,self.ranges)
            self.logScoreDict[binRep]=currentScore
            s+=np.exp(currentScore)
        return s    
    
    def scoreAllGraphsIJ(self,i,j):
        s=0.
        key =(i,j)
        if key in self.lookUpScoreIJ:
            return self.lookUpScoreIJ[key]
        else:            
            for binRep in self.get_DAG_with_Edge(i,j):
                if self.data.any():
                    dag  = self.bin2mat(binRep,self.d)
                    currentScore=self.scoreG(dag,self.ranges)
                    s+=np.exp(currentScore)
                else:
                    s+=self.uniform_score
            self.lookUpScoreIJ[key]=s
        return s    
    
    def probabilityEdge(self,i,j,prior=0.5):
        if prior==0.:
            return 0.
        edge_true = self.scoreAllGraphsIJ(i,j)
        edge_false = self.allScore - self.scoreAllGraphsIJ(i,j)
        return (edge_true*prior)/((edge_true*prior)+(edge_false*(1-prior)))
        
    
    def scoreG(self,linkMatrix,allRanges):
        llG = 0
        for j in range(len(linkMatrix)):
            parents = getParents(linkMatrix, j)
            key =tuple(parents)+(j,)
            if key in self.lookUpScore:               
                myScore=self.lookUpScore[key]
            else:
                pc = getParentConfig([allRanges[index] for index in parents])
                n_ijk =count(pc,self.data,j,parents,allRanges)
                # ToDo: you want to get the dirichlet prior from outside, therefore, get it in as a parameter somehow!!
                myScore=self.score(n_ijk,dirichletPrior(n_ijk,self.alpha))    
                self.lookUpScore[key]=myScore
            llG +=myScore 
        return llG   
    def allDAGs(self,n):
        indec = self.diagIndeces(n)
        graphAllOnes =np.power(2,np.power(n,2))
        maxEdge =  (np.power(n,2)-n)/2.;
        diagIndex = 1
        dagsBin=[]
        for i in range(graphAllOnes):
            offdiag=True
            binRep=bin(i)
            if binRep.count("1")<=maxEdge:
                for index in indec:
                    if index<len(binRep):
                        if binRep[-index]=="1":
                            offdiag=False
                            break
                if offdiag:
                    if not dagCheck(binRep,n):                                    
                        dagsBin.append(binRep)
        return dagsBin
    ######################################################################
    def score(self,n_ijk,alpha_ijk):
        prod_k = (scipy.special.gammaln(n_ijk + alpha_ijk) - scipy.special.gammaln(alpha_ijk)).sum(0)
        alpha_ij = alpha_ijk.sum(0)
        n_ij = n_ijk.sum(0)
        prod_ij = scipy.special.gammaln(alpha_ij) - scipy.special.gammaln(alpha_ij + n_ij)
        return (prod_ij + prod_k).sum()
    def bin2mat(self,binRep,n):
        dag = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                ind=self.computeNegIndex(i, j, n)
                if ind<=len(binRep):
                    if binRep[-ind]=="1":
                        dag[i][j]=1
        return dag
    
    def mat2bin(self,dag):
        binRepresentation='0b'
        start=False
        for i in range(len(dag)):
            for j in range(len(dag)):
                if dag[i][j]:
                    binRepresentation+='1'
                    start=True
                else:
                    if start:
                        binRepresentation+='0'
        if start:                
            return binRepresentation
        return '0b0'       
    
    def sample(self):
        print(self.data)
        return self.bin2mat(random.choice(self.allDAGbinary), self.d)
    def diagIndeces(self,n):
        indec=[]
        for i in range(0,n):
            indec.append(i*n +(1+i))
        return indec
    def computeNegIndex(self,i,j,n):
        return n*n - ((i)*n + j + 1) + 1
    
    
    def is_new_data(self,newdata):
        if self.data.shape==newdata.shape:
            if np.all(self.data==newdata):
                return False
        return True
'''
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
'''

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

        


'''
  def logDensity(self,index,args):
      print("ToDo implement log density")
      return 0


  def logDensityOfCounts(self,aux):
    term1 = scipy.special.gammaln(self.alpha) - scipy.special.gammaln(self.alpha + aux.numCustomers)
    term2 = aux.numTables + math.log(self.alpha + (aux.numTables * self.d))
    term3 = sum([scipy.special.gammaln(aux.tableCounts[index] - self.d) for index in aux.tableCounts])
    return term1 + term2 + term3

  def enumerateValues(self,args):
    aux = args.spaux
    old_indices = [i for i in aux.tableCounts]
    indices = old_indices + [aux.nextIndex]
    return indices
'''
