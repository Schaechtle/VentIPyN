
__author__ = 'ulli'


import seaborn as sns
import pylab as pl
#from plotting import load_experiments
import numpy as np
import pandas as pd
import scipy.io as scio
from models.covFunctions import *
import matplotlib.pyplot as plt
from venture import shortcuts
import sys
sys.path.append('../SPs/')
import venture.lite.types as t
from venture.lite.function import VentureFunction
import gp_der
import os
from models.tools import array

no = "a"
test_problem="airline"


sns.set(font_scale=2)

figlength = 30
figheigth = 15

n_samples = 200
dataset = "airline"
experiment = "test"
mh = '1000'
for i in range(1, len(sys.argv)):
    if str(sys.argv[i]) == "-p":  # structure posterior
        n_samples = int(sys.argv[i + 1])
    if str(sys.argv[i]) == "-d":  # structure posterior
        dataset= str(sys.argv[i + 1])
    if str(sys.argv[i]) == "-e":  # structure posterior
        experiment= 'experiment_' + str(sys.argv[i + 1]) + '_'
    if str(sys.argv[i]) == "-m":  # structure posterior
        mh = str(sys.argv[i + 1])

def array(xs):
  return t.VentureArrayUnboxed(np.array(xs),  t.NumberType())

def makeObservations(x,y,gp_str='(gp '):
    xString = genSamples(x,gp_str)
    ripl.observe(xString, array(y))

def genSamples(x,gp_str='(gp '):
    sampleString=gp_str+' (array '
    for i in range(len(x)):
        sampleString+= str(x[i]) + ' '
    sampleString+='))'
    #print(sampleString)
    return sampleString

mat_contents =scio.loadmat("real_world_data/"+test_problem+".mat")
X = mat_contents['X']
X= np.reshape(X,X.shape[0]).tolist()
y= mat_contents['y']
y= np.reshape(y,y.shape[0]).tolist()


ripl = shortcuts.make_lite_church_prime_ripl()
ripl.bind_foreign_sp("make_gp_part_der",gp_der.makeGPSP)
ripl.assume('make_const_func', VentureFunction(makeConstFunc, [t.NumberType()], constantType))
ripl.assume('zero', "(apply_function make_const_func 0)")

ripl.assume("func_times", makeLiftedMult(lambda x1, x2: np.multiply(x1,x2)))
ripl.assume("func_plus", makeLiftedAdd(lambda x1, x2: x1 + x2))



ripl.assume('make_periodic', VentureFunction(makePeriodic, [t.NumberType(), t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
ripl.assume('make_se',VentureFunction(makeSquaredExponential,[t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
ripl.assume('make_lin', VentureFunction(makeLinear, [t.NumberType()], t.AnyType("VentureFunction")))
ripl.assume('make_rq',VentureFunction(makeRQ, [t.NumberType(), t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))


ripl.assume('sf1','(tag (quote hypers) 0 (log (uniform_continuous 0 5)))')
ripl.assume('l1','(tag (quote hypers) 1 (log (uniform_continuous 0 5)))')



ripl.assume('ell','(tag (quote hypers) 2 (log (uniform_continuous 0 5)))')
ripl.assume('p','(tag (quote hypers) 3 (log (uniform_continuous 0.01 5)))')
ripl.assume('s','(tag (quote hypers) 4 (log (uniform_continuous 0 5)))')


ripl.assume('sf2','(tag (quote hypers) 5 (log (uniform_continuous 0 5)))')
ripl.assume('l2','(tag (quote hypers) 6 (log (uniform_continuous 0 5)))')


ripl.assume('sf_rq','(tag (quote hypers) 7 (log (uniform_continuous 0 5)))')
ripl.assume('l_rq','(tag (quote hypers) 8 (log (uniform_continuous 0 5)))')
ripl.assume('alpha','(tag (quote hypers)9 (log (uniform_continuous 0 5)))')

ripl.assume('a','(tag (quote hypers) 10 (log (uniform_continuous 0 5)))')




ripl.assume('lin','(apply_function make_lin a)')

ripl.assume('se1', "(apply_function make_se sf1 l1 )")
ripl.assume('se2', "(apply_function make_se sf2 l2 )")

ripl.assume('per', "(apply_function make_periodic ell p s)")

ripl.assume('rq', "(apply_function make_rq l_rq sf_rq alpha)")

ripl.assume('gp',"""(tag (quote model) 0
                        (make_gp_part_der zero
                            (apply_function func_plus
                                (apply_function func_times lin se1)
                                (apply_function func_times se2
                                         (apply_function func_times per rq)
                                )
                            )
                        )
                    )""")

makeObservations(X,y)
ripl.infer("(mh (quote hypers) one "+mh+" )")

ripl.assume('gp_SExLIN',"(make_gp_part_der zero (apply_function func_times se1 lin))")
ripl.assume('gp_SExPERxRQ',"""(make_gp_part_der zero (apply_function func_times se2
                                                                    (apply_function func_times per rq  )
                                                      )
                               )""")

if not os.path.exists("/home/ulli/Dropbox/gpmemplots/parsing_residuals/"+dataset):
        os.makedirs("/home/ulli/Dropbox/gpmemplots/parsing_residuals/"+dataset)
if not os.path.exists("/home/ulli/Dropbox/gpmemplots/parsing_residuals/"+dataset+"/"+experiment):
        os.makedirs("/home/ulli/Dropbox/gpmemplots/parsing_residuals/"+dataset+"/"+experiment)

'''
def predictions(gp_string):
    sample_string=genSamples(X,gp_string)
    y_predicted = []
    for i in range(n_samples):
        y_predicted.append(ripl.sample(sample_string))
    mean_pred = np.median(y_predicted,axis=0)
    np.save("/home/ulli/Dropbox/gpmemplots/parsing_residuals/"+dataset+"/"+experiment+"/y_"+gp_string[1:]+"predictions.npy", mean_pred)

predictions("(gp")
makeObservations(X,y,"(gp_SExLIN")
predictions("(gp_SExLIN")
makeObservations(X,y,"(gp_SExPERxRQ")
predictions("(gp_SExPERxRQ")

'''
gp_string = '0test'
sample_string=genSamples(X)
y_predicted = []
for i in range(n_samples):
    y_predicted.append(ripl.sample(sample_string))
mean_pred = np.median(y_predicted,axis=0)
np.save("/home/ulli/Dropbox/gpmemplots/parsing_residuals/"+dataset+"/"+experiment+"/y_"+gp_string[1:]+"predictions.npy", mean_pred)





#print(y_predict_SExLIN.shape)









