
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




n_it = 2

particles = '3'
no = "cont_0_"
outlier_sigma=1
for i in range(1,len(sys.argv)):
            if str(sys.argv[i])=="--no": # structure posterior
                no = str(sys.argv[i+1])
            if str(sys.argv[i])=="-s": # structure posterior
                outlier_sigma = float(sys.argv[i+1])
            if str(sys.argv[i])=="-i": # structure posterior
                n_it = int(sys.argv[i+1])
            if str(sys.argv[i])=="-p": # structure posterior
                particles = str(sys.argv[i+1])











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

def plot_hyper(n_iteration):



    x = np.load('/home/ulli/Dropbox/gpmemplots/x_s.npy')
    y = np.load('/home/ulli/Dropbox/gpmemplots/y_s.npy')


    ripl = shortcuts.make_lite_church_prime_ripl()
    ripl.bind_foreign_sp("make_gp_part_der",gp_der.makeGPSP)
    ripl.assume('make_const_func', VentureFunction(makeConstFunc, [t.NumberType()], constantType))
    ripl.assume('zero', "(apply_function make_const_func 0)")



    ripl.infer('(resample '+ particles +' )')

    ripl.assume("func_plus", makeLiftedAdd(lambda x1, x2: x1 + x2))
    ripl.assume('make_se',VentureFunction(makeSquaredExponential,[t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
    ripl.assume('make_noise',VentureFunction(makeNoise,[t.NumberType()], t.AnyType("VentureFunction")))


    ripl.assume('alpha_sf','(tag (quote hyperhyper) 0 (gamma 7 1))')
    ripl.assume('beta_sf','(tag (quote hyperhyper) 2 (gamma 1 0.5))')
    ripl.assume('alpha_l','(tag (quote hyperhyper) 1 (gamma 7 1))')
    ripl.assume('beta_l','(tag (quote hyperhyper) 3 (gamma 1 0.5))')
    ripl.assume('alpha_s','(tag (quote hyperhyper) 4 (gamma 7 1))')
    ripl.assume('beta_s','(tag (quote hyperhyper) 5 (gamma 1 0.5))')

    ripl.assume('sf','(tag (quote hyper) 0 (gamma alpha_sf beta_sf ))')
    ripl.assume('l','(tag (quote hyper) 1 (gamma alpha_l beta_l ))')

    ripl.assume('sigma','(tag (quote hyper) 2 (uniform_continuous 0 2 ))')

    ripl.assume('se', "(apply_function make_se (log sf ) (log l ) )")
    ripl.assume('wn','(apply_function make_noise (log sigma) )')

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





    ripl.infer("(repeat 100 (do (mh (quote hyperhyper) one 2) (mh (quote hyper) one 1)))")






    ds = ripl.infer('(collect sf l sigma)')
    df = ds.asPandas()
    df['Hyper-Parameter Learning']= pd.Series(['after' for _ in range(len(df.index))], index=df.index)


    plt.figure(n_iteration)
    sns.plt.yticks([])

    #from matplotlib2tikz import save as tikz_save
    #tikz_save('/home/ulli/Dropbox/gpmemplots/neal_contourunif_sigma_before_'+no+str(n_iteration)+'.tikz', figureheight='8cm', figurewidth='8cm' )

    fig = plt.figure(figsize=(figlength,figheigth), dpi=200)

    for i in range(100):
        xpost= np.random.uniform(-3,3,200)
        sampleString=genSamples(xpost)
        ypost = ripl.sample(sampleString)
        yp = [y_temp for (x_temp,y_temp) in sorted(zip(xpost,ypost))]
        plt.plot(sorted(xpost),yp,c="red",alpha=0.008,linewidth=2)

    plt.axis((-2,2,-1,3))

    plt.scatter(x,y,color='black',marker='x',s=50,edgecolor='black',linewidth='1.5')
    fig.savefig('/home/ulli/Dropbox/gpmemplots/neal_contourcheck_'+no+'_'+str(n_iteration)+'.png', dpi=fig.dpi)
    plt.clf()
    plot_contours(df_before,'before',n_iteration)
    plot_contours(df,'after',n_iteration)



def plot_contours(df,name,n_iteration):

    joint_grid_plot("l","sigma",df,name,n_iteration)
    joint_grid_plot("l","sf",df,name,n_iteration)
    joint_grid_plot("sf","sigma",df,name,n_iteration)
    joint_grid_plot("l","sigma",df,name,n_iteration,False)
    joint_grid_plot("l","sf",df,name,n_iteration,False)
    joint_grid_plot("sf","sigma",df,name,n_iteration,False)



def joint_grid_plot(var1,var2,df,name,n_iteration,marginal=True):
    if marginal:
        g = sns.JointGrid(var1, var2, df, space=0)
        g.plot_marginals(sns.kdeplot, shade=True)
        name = name+"_marginal_"
        g.plot_joint(sns.kdeplot, shade=True, cmap="PuBu", n_levels=40);
        ax = g.ax_joint
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        sns.kdeplot(df[[var1,var2]].values, shade=True, cmap="PuBu", n_levels=40);
        sns.plt.xlabel("")
        sns.plt.ylabel("")
    sns.set(font_scale=2)
    plt.savefig('/home/ulli/Dropbox/gpmemplots/neal_contour_'+var1+'_vs_'+var2+'_'+name+'_'+no+'_'+str(n_iteration)+'.png', dpi=200,bbox_inches='tight')
    plt.clf()


for i in range(n_it):
    plot_hyper(i)
