import seaborn as sns
import pylab as pl
#from plotting import load_experiments
import matplotlib.pyplot as plt
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
import random

figwidth = 10
figheight = 10

sns.set(font_scale=2)
no = "s"
n = 100

def f(x):
    return 0.3 + 0.4*x + 0.5*np.sin(2.7*x) + (1.1/(1+x**2))

x2plot = np.linspace(-3,3,1000)
f2plot = f(x2plot)

x = np.random.normal(0,1,n)
y=np.zeros(x.shape)

for i in range(n):
    if random.random()>0.10:
        y[i] = f(x[i]) + np.random.normal(0,0.1,1)
    else:
        y[i] = f(x[i]) + np.random.normal(0,1,1)
        
np.save('syndata/x_'+no+'.npy', x)
np.save('syndata/y_'+no+'.npy', y)

def array(xs):
  return t.VentureArrayUnboxed(np.array(xs),  t.NumberType())

def makeObservations(x,y):
    xString = genSamples(x)
    ripl.observe(xString, array(y))

def genSamples(xs):
    return '(gp (array %s))' % ' '.join(str(x) for x in xs)

ripl = shortcuts.make_lite_church_prime_ripl()
ripl.bind_foreign_sp("make_gp_part_der",gp_der.makeGPSP)
ripl.assume('make_const_func', VentureFunction(makeConstFunc, [t.NumberType()], constantType))
ripl.assume('zero', "(apply_function make_const_func 0)")

#ripl.assume("func_times", makeLiftedMult(lambda x1, x2: np.multiply(x1,x2)))
ripl.assume("func_plus", makeLiftedAdd(lambda x1, x2: x1 + x2))

ripl.assume('make_se',VentureFunction(makeSquaredExponential,[t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
ripl.assume('make_noise',VentureFunction(makeNoise,[t.NumberType()], t.AnyType("VentureFunction")))

ripl.assume('alpha_sf','(tag (quote hyperhyper) 0 (gamma 7 1))')
ripl.assume('beta_sf','(tag (quote hyperhyper) 2 (gamma 1 0.5))')
ripl.assume('alpha_l','(tag (quote hyperhyper) 1 (gamma 7 1))')
ripl.assume('beta_l','(tag (quote hyperhyper) 3 (gamma 1 0.5))')
ripl.assume('alpha_s','(tag (quote hyperhyper) 4 (gamma 7 1))')
ripl.assume('beta_s','(tag (quote hyperhyper) 5 (gamma 1 0.5))')

ripl.assume('sf','(tag (quote hyper) 0 (log (gamma alpha_sf beta_sf )))')
ripl.assume('l','(tag (quote hyper) 1 (log (gamma alpha_l beta_l )))')

ripl.assume('sigma','(tag (quote hyper) 2 (uniform_continuous 0 2 ))')
ripl.assume('l_sigma','(log sigma)')

ripl.assume('se', "(apply_function make_se sf l )")
ripl.assume('wn','(apply_function make_noise sigma  )')

ds = ripl.infer('(collect sf l sigma)')
df = ds.asPandas()
df['Hyper-Parameter Learning']= pd.Series(['before' for _ in range(len(df.index))], index=df.index)

df_before =df

ripl.assume('composite_covariance', '(apply_function func_plus se wn)')
ripl.assume('gp', '(make_gp_part_der zero composite_covariance)')

fig = plt.figure(figsize=(figwidth,figheight), dpi=200)
#xpost= np.random.uniform(-3,3,200)
for i in range(100):
    xpost= np.random.uniform(-3,3,200)
    sampleString=genSamples(xpost)
    ypost = ripl.sample(sampleString)
    yp = [y_temp for (x_temp,y_temp) in sorted(zip(xpost,ypost))]
    pl.plot(sorted(xpost),yp,c="red",alpha=0.008,linewidth=2)

plt.axis((-2,2,-1,3))
pl.plot(x2plot,f2plot,color='blue')
pl.scatter(x,y,color='black',marker='x',s=50,edgecolor='black',linewidth='1.5')   
    
fig.savefig('neal_example_figs/neal_se_1'+no+'.svg', dpi=fig.dpi)
fig.savefig('neal_example_figs/neal_se_1'+no+'.png', dpi=fig.dpi)


makeObservations(x,y)

fig = plt.figure(figsize=(figwidth,figheight), dpi=200)
#xpost= np.random.uniform(-3,3,200)
for i in range(100):
    xpost= np.random.uniform(-3,3,200)
    sampleString=genSamples(xpost)
    ypost = ripl.sample(sampleString)
    yp = [y_temp for (x_temp,y_temp) in sorted(zip(xpost,ypost))]
    pl.plot(sorted(xpost),yp,c="red",alpha=0.008,linewidth=2)

plt.axis((-2,2,-1,3))
pl.plot(x2plot,f2plot,color='blue')
pl.scatter(x,y,color='black',marker='x',s=50,edgecolor='black',linewidth='1.5')   
    
fig.savefig('neal_example_figs/neal_se_2'+no+'.svg', dpi=fig.dpi)
fig.savefig('neal_example_figs/neal_se_2'+no+'.png', dpi=fig.dpi)

ripl.infer("(repeat 100 (do (mh (quote hyperhyper) one 2) (mh (quote hyper) one 1)))")



fig = plt.figure(figsize=(figwidth,figheight), dpi=200)
#xpost= np.random.uniform(-3,3,200)
for i in range(500):
    xpost= np.random.uniform(-3,3,200)
    sampleString=genSamples(xpost)
    ypost = ripl.sample(sampleString)
    yp = [y_temp for (x_temp,y_temp) in sorted(zip(xpost,ypost))]
    pl.plot(sorted(xpost),yp,c="red",alpha=0.008,linewidth=2)

plt.axis((-2,2,-1,3))
pl.plot(x2plot,f2plot,color='blue')
pl.scatter(x,y,color='black',marker='x',s=50,edgecolor='black',linewidth='1.5')

fig.savefig('neal_example_figs/neal_se_3'+no+'.svg', dpi=fig.dpi)
fig.savefig('neal_example_figs/neal_se_3'+no+'.png', dpi=fig.dpi)
