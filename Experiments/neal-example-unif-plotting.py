

import seaborn as sns
import pylab as pl
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

from models.tools import array
figlength = 30
figheigth = 15
import random
import matplotlib.pyplot as plt
figlength = 10
figheigth = 10



sns.set(font_scale=2)

n_it = 1

no = "b"
outlier_sigma=1
for i in range(1,len(sys.argv)):
            if str(sys.argv[i])=="-n": # structure posterior
                no = str(sys.argv[i+1])
            if str(sys.argv[i])=="-s": # structure posterior
                outlier_sigma = float(sys.argv[i+1])
            if str(sys.argv[i])=="-i": # structure posterior
                n_it = int(sys.argv[i+1])

n = 500



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


    # In[12]:

    ripl = shortcuts.make_lite_church_prime_ripl()
    ripl.bind_foreign_sp("make_gp_part_der",gp_der.makeGPSP)
    ripl.assume('make_const_func', VentureFunction(makeConstFunc, [t.NumberType()], constantType))
    ripl.assume('zero', "(apply_function make_const_func 0)")


    # Out[12]:

    #     <function models.covFunctions.<lambda>>

    # In[13]:

    ripl.infer('(resample 100)')

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



    '''
    fig = plt.figure(figsize=(figlength,figheigth), dpi=200)
    xpost= np.random.uniform(-3,3,100)
    for i in range(100):

        sampleString=genSamples(xpost)
        ypost = ripl.sample(sampleString)
        yp = [y_temp for (x_temp,y_temp) in sorted(zip(xpost,ypost))]
        pl.plot(sorted(xpost),yp,c="red",alpha=0.1,linewidth=2)

    pl.scatter(x,y,color='black',marker='x',s=50,edgecolor='black',linewidth='1.5',)
    plt.axis((-2,2,-1,3))
    pl.plot(x2plot,f2plot,c='black')
    fig.savefig('/home/ulli/Paper_VentureGP/figs/neal_se_1'+no+'.svg', dpi=fig.dpi)
    fig.savefig('/home/ulli/Paper_VentureGP/figs/neal_se_1'+no+'.png', dpi=fig.dpi)
    '''


    # Out[20]:

    #     '\nfig = plt.figure(figsize=(figlength,figheigth), dpi=200)\nxpost= np.random.uniform(-3,3,100)\nfor i in range(100):\n \n    sampleString=genSamples(xpost)\n    ypost = ripl.sample(sampleString)\n    yp = [y_temp for (x_temp,y_temp) in sorted(zip(xpost,ypost))]\n    pl.plot(sorted(xpost),yp,c="red",alpha=0.1,linewidth=2)\n\npl.scatter(x,y,color=\'black\',marker=\'x\',s=50,edgecolor=\'black\',linewidth=\'1.5\',)\nplt.axis((-2,2,-1,3))\npl.plot(x2plot,f2plot,c=\'black\')\nfig.savefig(\'/home/ulli/Paper_VentureGP/figs/neal_se_1\'+no+\'.svg\', dpi=fig.dpi)\nfig.savefig(\'/home/ulli/Paper_VentureGP/figs/neal_se_1\'+no+\'.png\', dpi=fig.dpi)\n'

    # In[21]:

    makeObservations(x,y,ripl)


    # In[22]:

    '''
    fig = plt.figure(figsize=(figlength,figheigth), dpi=200)
    xpost= np.random.uniform(-3,3,100)
    for i in range(100):

        sampleString=genSamples(xpost)
        ypost = ripl.sample(sampleString)
        yp = [y_temp for (x_temp,y_temp) in sorted(zip(xpost,ypost))]
        pl.plot(sorted(xpost),yp,c="red",alpha=0.1,linewidth=2)

    pl.scatter(x,y,color='black',marker='x',s=50,edgecolor='black',linewidth='1.5')
    plt.axis((-2,2,-1,3))
    pl.plot(x2plot,f2plot,c='black')
    fig.savefig('/home/ulli/Paper_VentureGP/figs/neal_se_2'+no+'.svg', dpi=fig.dpi)
    fig.savefig('/home/ulli/Paper_VentureGP/figs/neal_se_2'+no+'.png', dpi=fig.dpi)
    '''


    # Out[22]:

    #     '\nfig = plt.figure(figsize=(figlength,figheigth), dpi=200)\nxpost= np.random.uniform(-3,3,100)\nfor i in range(100):\n \n    sampleString=genSamples(xpost)\n    ypost = ripl.sample(sampleString)\n    yp = [y_temp for (x_temp,y_temp) in sorted(zip(xpost,ypost))]\n    pl.plot(sorted(xpost),yp,c="red",alpha=0.1,linewidth=2)\n    \npl.scatter(x,y,color=\'black\',marker=\'x\',s=50,edgecolor=\'black\',linewidth=\'1.5\')\nplt.axis((-2,2,-1,3))\npl.plot(x2plot,f2plot,c=\'black\')\nfig.savefig(\'/home/ulli/Paper_VentureGP/figs/neal_se_2\'+no+\'.svg\', dpi=fig.dpi)\nfig.savefig(\'/home/ulli/Paper_VentureGP/figs/neal_se_2\'+no+\'.png\', dpi=fig.dpi)\n'

    # In[23]:

    ripl.infer("(mh (quote hyper) one 200)")


    # Out[23]:

    #     []

    # In[24]:

    '''
    fig = plt.figure(figsize=(figlength,figheigth), dpi=200)
    xpost= np.random.uniform(-3,3,100)
    for i in range(100):

        sampleString=genSamples(xpost)
        ypost = ripl.sample(sampleString)
        yp = [y_temp for (x_temp,y_temp) in sorted(zip(xpost,ypost))]
        pl.plot(sorted(xpost),yp,c="red",alpha=0.1,linewidth=2)
    pl.scatter(x,y,color='black',marker='x',s=50,edgecolor='black',linewidth='1.5')
    plt.axis((-2,2,-1,3))
    pl.plot(x2plot,f2plot,c='black')
    fig.savefig('/home/ulli/Paper_VentureGP/figs/neal_se_3'+no+'.svg', dpi=fig.dpi)
    fig.savefig('/home/ulli/Paper_VentureGP/figs/neal_se_3'+no+'.png', dpi=fig.dpi)
    '''



    ds = ripl.infer('(collect sf l sigma)')
    df = ds.asPandas()
    df['Hyper-Parameter Learning']= pd.Series(['after' for _ in range(len(df.index))], index=df.index)


    df_all=pd.concat([df_before,df])



    g = sns.FacetGrid(df_all, col="Hyper-Parameter Learning", palette="Greens_d",col_order=['before','after'],size=4, aspect=2, xlim=(0, 5),margin_titles=True)
    g.map(sns.distplot, "sigma",norm_hist=True);
    g.savefig('/home/ulli/Dropbox/gpmemplots/neal_unif_sigma_'+no+str(n_it)+'.png', dpi=200)

for i in range(n_it):
    plot_hyper(i)