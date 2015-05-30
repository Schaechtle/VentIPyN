import sys
from plotting import *
import pandas as pd
structure_posterior = False
mean_residuals = False
median_residuals = False
last_n_residuals = False

for i in range(1,len(sys.argv)):
            if str(sys.argv[i])=="-sp": # structure posterior
                structure_posterior= True
            if str(sys.argv[i])=="--median": # structure posterior
                median_residuals= True
            if str(sys.argv[i])=="--mean": # structure posterior
                mean_residuals= True
            if str(sys.argv[i])=="--last": # structure posterior
                last_n_residuals= True
            if str(sys.argv[i])=="-d":
                date_experiment= str(sys.argv[i+1])
            if str(sys.argv[i])=="-f":
                experiment_ini_file= str(sys.argv[i+1])


if structure_posterior:
    df =get_posterior_structure(date_experiment,experiment_ini_file)
if mean_residuals:
    df =get_dataFrame(date_experiment,experiment_ini_file)
if median_residuals:
    df =get_dataFrame_median(date_experiment,experiment_ini_file)
if last_n_residuals:
    df =get_last_n_residuals(date_experiment,experiment_ini_file)