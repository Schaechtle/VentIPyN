
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
#from plotting import load_experiments
import numpy as np
import pandas as pd
import scipy.io as scio
from models.covFunctions import *

from venture import shortcuts
import sys
sys.path.append('../SPs/')
import venture.lite.types as t
from venture.lite.function import VentureFunction
import gp_der
import scipy.stats
from models.tools import array
figlength = 30
figheigth = 15
import random
import matplotlib.pyplot as plt
figlength = 10
figheigth = 10


sns.set(font_scale=2)

n_it = 1
n = 100
particles = '10'
no = "d"
outlier_sigma=1
for i in range(1,len(sys.argv)):
            if str(sys.argv[i])=="--no": # structure posterior
                no = str(sys.argv[i+1])
            if str(sys.argv[i])=="-s": # structure posterior
                outlier_sigma = float(sys.argv[i+1])
            if str(sys.argv[i])=="-i": # structure posterior
                n_it = int(sys.argv[i+1])
            if str(sys.argv[i])=="-n": # structure posterior
                n = int(sys.argv[i+1])
            if str(sys.argv[i])=="-p": # structure posterior
                particles = str(sys.argv[i+1])



def f(x):
    return 0.3 + 0.4*x + 0.5*np.sin(2.7*x) + (1.1/(1+x**2))



x2plot = np.linspace(-3,3,1000)
f2plot = f(x2plot)







'''
fig = plt.figure(figsize=(figlength,figheigth), dpi=200)
pl.scatter(x,y)
pl.plot(x2plot,f2plot,c='black')
'''

for i in range(1,len(sys.argv)):
            if str(sys.argv[i])=="-sp": # structure posterior
                structure_posterior= True


def array(xs):
  return t.VentureArrayUnboxed(np.array(xs),  t.NumberType())

def makeObservations(x,y,ripl):
    xString = genSamples(x)
    ripl.observe(xString, array(y))

def genSamples(x):
    sampleString='(gp (array '
    for i in range(len(x)):
        sampleString+= str(x[i]) + ' '
    sampleString+='))'
    #print(sampleString)
    return sampleString

def plot_hyper(n_it):

    x = np.random.normal(0,1,n)
    y=np.zeros(x.shape)



    for i in range(n):
        if random.random()>0.3:
            y[i] = f(x[i]) + np.random.normal(0,0.1,1)
        else:
            y[i] = f(x[i]) + np.random.normal(0,outlier_sigma,1)



    ripl = shortcuts.make_lite_church_prime_ripl()
    ripl.bind_foreign_sp("make_gp_part_der",gp_der.makeGPSP)
    ripl.assume('make_const_func', VentureFunction(makeConstFunc, [t.NumberType()], constantType))
    ripl.assume('zero', "(apply_function make_const_func 0)")



    ripl.infer('(resample '+ particles +' )')

    ripl.assume("func_plus", makeLiftedAdd(lambda x1, x2: x1 + x2))
    ripl.assume('make_se',VentureFunction(makeSquaredExponential,[t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
    ripl.assume('make_noise',VentureFunction(makeNoise,[t.NumberType()], t.AnyType("VentureFunction")))



    ripl.assume('sf','(tag (quote hyper) 0 (log (uniform_continuous 0 10 )))')
    ripl.assume('l','(tag (quote hyper) 1 (log (uniform_continuous 0 10 )))')

    ripl.assume('sigma','(tag (quote hyper) 2 (uniform_continuous 0 5 ))')
    ripl.assume('l_sigma','(log sigma)')

    ripl.assume('se', "(apply_function make_se sf l )")
    ripl.assume('wn','(apply_function make_noise sigma  )')

    ds = ripl.infer('(collect sf l sigma)')
    df = ds.asPandas()
    df['Hyper-Parameter Learning']= pd.Series(['before' for _ in range(len(df.index))], index=df.index)


    df_before =df


    ripl.assume('t_dist','(lambda (i) (student_t 4))')



    ripl.assume('gp',"""(tag (quote model) 0
                            (make_gp_part_der zero (apply_function func_plus se wn  )
                                    )

                                 )""")





    makeObservations(x,y,ripl)





    ripl.infer("(mh (quote hyper) one 200)")






    ds = ripl.infer('(collect sf l sigma)')
    df = ds.asPandas()
    df['Hyper-Parameter Learning']= pd.Series(['after' for _ in range(len(df.index))], index=df.index)


    plt.figure(n_it)
    sns.plt.yticks([])

    sns.distplot(df_before['sigma'])
    sns.plt.xlabel(" ")
    plt.savefig('/home/ulli/Dropbox/gpmemplots/neal_unif_sigma_before_'+no+str(n_it)+'.png', dpi=200,bbox_inches='tight')
    plt.figure(n_it+100)
    sns.distplot(df['sigma'])
    sns.plt.yticks([])
    sns.plt.xlabel(" ")
    plt.savefig('/home/ulli/Dropbox/gpmemplots/neal_unif_sigma_after_'+no+str(n_it)+'.png', dpi=200,bbox_inches='tight')
    #from matplotlib2tikz import save as tikz_save
    #tikz_save('/home/ulli/Dropbox/gpmemplots/neal_unif_sigma_before_'+no+str(n_it)+'.tikz', figureheight='8cm', figurewidth='8cm' )

for i in range(n_it):
    plot_hyper(i)