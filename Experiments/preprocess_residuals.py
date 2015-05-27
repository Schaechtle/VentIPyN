import seaborn as sns
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt
import sys

import os.path


name = 'cut_off_50'

cut =50


def remove_massive_outliers(df,replace_by=50.):
    df.residuals[df.residuals >replace_by] = replace_by
    df.residuals[df.residuals < (-replace_by)] = -replace_by
    return df

for i in range(1,len(sys.argv)):
            if str(sys.argv[i])=="-n":
                name= str(sys.argv[i+1])

            if str(sys.argv[i])=="--cut-off":
                cut= float(sys.argv[i+1])
            if str(sys.argv[i])=="-d":
                date_str= str(sys.argv[i+1])

fname = "results/n_res_"+date_str
if os.path.isfile(fname):
    print("loading one df where all experiments are collected")
    df = pd.read_pickle(fname)
else:
    print("loading from scratch")
    df,b=get_last_n_residuals("experiment.ini",date_str,100)

df = pd.read_pickle("results/n_res_"+date_str)
df = remove_massive_outliers(df,cut)
df.to_pickle("results/cleaned/"+name+date_str)

