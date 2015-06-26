
# In[1]:

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
import os
import matplotlib.pyplot as plt
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
figlength = 10
figheigth = 10


# In[2]:

def plot_contours(df,name):
    df = df.loc[df['l'] < 10]
    df = df.loc[df['sf'] < 10]
    df = df.loc[df['l'] > -10]
    df = df.loc[df['sf'] > -10]
    joint_grid_plot("l","sigma",df,name)
    joint_grid_plot("l","sf",df,name)
    joint_grid_plot("sf","sigma",df,name,)
    joint_grid_plot("l","sigma",df,name,False)
    joint_grid_plot("l","sf",df,name,False)
    joint_grid_plot("sf","sigma",df,name,False)




no = 's'



run_name = "test01/"



path= "syndata/"+run_name



def joint_grid_plot(var1, var2, df, name, marginal=True):
    if marginal:
        g = sns.JointGrid(var1, var2, df, space=0)
        g.plot_marginals(sns.kdeplot, shade=True)
        name = "_marginal_" + name
        g.plot_joint(sns.kdeplot, shade=True, cmap="PuBu", n_levels=40);
        ax = g.ax_joint
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        sns.kdeplot(df[[var1, var2]].values, shade=True, cmap="PuBu", n_levels=40);
        sns.plt.xlabel("")
        sns.plt.ylabel("")
    sns.set(font_scale=2)
    plt.savefig('/home/ulli/Dropbox/gpmemplots/'+path+'neal_contour_' + var1 + '_vs_' + var2 + '_' + no + '_'+name + '.png', dpi=200, bbox_inches='tight')
    plt.clf()



file_str = "before_parameters_"
df_list =[]
for i in os.listdir(path):
    if os.path.isfile(os.path.join(path,i)) and 'before_parameters_' in i:
        df = pd.read_pickle(path+i)
        df_list.append(df)
df_before=  pd.concat(df_list)




df_before['Hyper-Parameter Learning']= pd.Series(['before' for _ in range(len(df_before))], index=df_before.index)


file_str = "after_parameters_"
df_list =[]
for i in os.listdir(path):
    if os.path.isfile(os.path.join(path,i)) and 'after_parameters_' in i:
        df = pd.read_pickle(path+i)
        df_list.append(df)
df_after =  pd.concat(df_list)




df_after['Hyper-Parameter Learning']= pd.Series(['after' for _ in range(len(df_after))], index=df_after.index)




plot_contours(df_before[['l','sf','sigma']],'before')
plot_contours(df_after[['l','sf','sigma']],'after')





