from cycleChecker import computeNegIndex
import numpy as np
import itertools
import copy

#################################################################################################################
def dirichletPrior(counts,alpha):
    if counts.shape[0]!=1:
        prior=np.zeros(counts.shape)
        for j in range(counts.shape[1]):
            denom = counts[:,j].sum()+alpha*counts.shape[0]
            for i in range(counts.shape[0]):
                prior[i][j]=(counts[i][j]+alpha)/denom
    else:
        total = counts.sum()+alpha*counts.shape[1]
        prior = (counts+alpha)/total
    return prior


def bin2mat(binRep,n):
    dag = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            ind=computeNegIndex(i+1, j+1, n)
            if ind<=len(binRep):
                if binRep[-ind]=="1":
                    dag[i][j]=1
    return dag
                 
######################################################################
def getParents(dag,i):
    col = dag[:,i]
    return np.where(col==1)[0]   
######################################################################
def getTuple(parents,row):
    t = tuple()
    for i in range(len(parents)):
        t = t + (row[parents[i]],)
    return t


def count_withDict(parentConfig,data,childIndex,parentIndeces,ranges):
    countsC = np.zeros((ranges[childIndex],1))
    if not parentIndeces:
        for j in range(len(data)):
            countsC[data[j][childIndex]]+=1
        return countsC
    # above redone 15th April 2015
    hashmap={}
    for confTuple in parentConfig:
        print(confTuple)
        hashmap[confTuple]=copy.deepcopy(countsC)
    
    for j in range(len(data)):
        hashmap[getTuple(parentIndeces,data[j])][data[j][childIndex]]+=1
        
    return hashmap

def count(parentConfig,data,childIndex,parentIndeces,ranges):
    countsC = np.zeros((ranges[childIndex],1))
    if parentIndeces.size==0:
        for j in range(len(data)):
            countsC[data[j][childIndex]]+=1
        return countsC
    # above redone 15th April 2015
    hashmap={}
    matrix_index=0
    for confTuple in parentConfig:
        hashmap[confTuple]=matrix_index
        matrix_index+=1
    countMatrix=np.zeros((ranges[childIndex],matrix_index))
    for j in range(len(data)):
        configIndex = hashmap[getTuple(parentIndeces,data[j])]
        childValue=data[j][childIndex]
        countMatrix[childValue][configIndex]+=1
        
    return countMatrix

'''
>>> from scipy.stats import itemfreq
>>> x = [1,1,1,2,2,2,5,25,1,1]
itemfreq(x)
'''
######################################################################
def computeEdge(dag,z,k):
    # returns count for edges, usage example. How many edges from second class to first are present: nplus[1][0]
    nplus  = [[0 for i in range(k)] for j in range(k)]
    nminus=copy.deepcopy(nplus)
    for i in range(len(dag)):
        for j in range(len(dag[0])):
            if dag[i][j]==1:
                nplus[z[i]][z[j]]+=1
            else:
                nminus[z[i]][z[j]]+=1            
    return nplus,nminus
######################################################################  
def matchConfig(line,parents):
    return tuple([line[p] for p in parents])    
######################################################################
def getParentConfig(ranges): #get configuration of parents for counts
    if not ranges:
        return []
    parentvalues =[]
    for maxValue in ranges:
        allValues= []
        for i in range(0,maxValue):
            allValues.append(i)
        parentvalues.append(allValues)
    return itertools.product(*parentvalues) 
#################################################################################################################

def uniformPrior(counts): #for testing purposes!!! ToDo: remove in DAG class as soon as this prior is delivered form outside SP
    p = 1./(counts.shape[1]*2.)
    alphaIJK=np.empty(counts.shape)
    alphaIJK.fill(p)
    return alphaIJK

def dirichletPrior(counts,alpha):
    if counts.shape[0]!=1:
        prior=np.zeros(counts.shape)
        for j in range(counts.shape[1]):
            denom = counts[:,j].sum()+alpha*counts.shape[0]
            for i in range(counts.shape[0]):
                prior[i][j]=(counts[i][j]+alpha)/denom
    else:
        total = counts.sum()+alpha*counts.shape[1]
        prior = (counts+alpha)/total
    return prior

'''
dataList=[[1,0,1],
          [1,0,1],
          [1,1,0],
          [0,0,1],
          [1,0,0],
          [0,0,0],
          [1,1,0],
          [1,1,0]        
          ]

data = np.array(dataList)
c=count(getParentConfig([2,2]),data,2,[0,1],[2,2,2])
print(c)
print(uniformPrior(c))
 '''
 

