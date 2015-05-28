from plotting import get_dataFrame
import os
import sys
import pandas as pd

for i in range(1,len(sys.argv)):
            if str(sys.argv[i])=="-f":
                ini_file_path= str(sys.argv[i+1])
            if str(sys.argv[i])=="-d":
                date_experiment= str(sys.argv[i+1])

fname = "results/mean_residual_"+date_experiment
if os.path.isfile(fname):
    print("loading existing df with median")
    df = pd.read_pickle(fname)
else:
    print("loading from scratch")
    df  = get_dataFrame(date_experiment,ini_file_path)
    df.to_pickle("fname")



print(df)