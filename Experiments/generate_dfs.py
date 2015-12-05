import sys
from plotting import *
import pandas as pd
import os

structure_posterior = False
mean_residuals = False
median_residuals = False
last_n_residuals = False
overwrite_existing = False
simple_df_load = False


for i in range(1,len(sys.argv)):
            if str(sys.argv[i])=="-sp": # structure posterior
                structure_posterior= True
            if str(sys.argv[i])=="--median": # median residuals
                median_residuals= True
            if str(sys.argv[i])=="--mean": # mean residuasl
                mean_residuals= True
            if str(sys.argv[i])=="--last": # last n residuals
                last_n_residuals= True
            if str(sys.argv[i])=="-d":
                date_experiment= str(sys.argv[i+1])
            if str(sys.argv[i])=="-f":
                experiment_ini_file= str(sys.argv[i+1])
            if str(sys.argv[i])=="--simple":
                str_2_start= str(sys.argv[i+1])
                simple_df_load=True
            if str(sys.argv[i])=="--delete": # last n residuals
                overwrite_existing = True

if overwrite_existing and last_n_residuals:
    print("Warning - cannot overwrite last n residuals - needs to be deleted by hand")


if structure_posterior:
    _ = get_posterior_structure(date_experiment, experiment_ini_file,overwrite_existing)
if mean_residuals:
    _ = get_dataFrame(date_experiment,experiment_ini_file,overwrite_existing)
if median_residuals:
    _ = get_dataFrame_median(date_experiment,experiment_ini_file,overwrite_existing)
#if last_n_residuals:
#   os.system("preprocess_residuals.py -d "+date_experiment+" -f "+experiment_ini_file)
if overwrite_existing and last_n_residuals:
    print("Warning - cannot overwrite last n residuals - needs to be deleted by hand")
if simple_df_load:
    simple_load(date_experiment,str_2_start)
