import pandas as pd
from util import Config,ConfigSectionMap
import numpy as np

def average_frames(repeated_experiments):
    repeated = len(repeated_experiments)
    assert repeated == int(ConfigSectionMap("others")['repeat'])
    df = repeated_experiments[0]
    for i in range(1,repeated):
        df['logscore']+=repeated_experiments[i]['logscore']
        df['residuals']+=repeated_experiments[i]['residuals']
        df['base-line']+=repeated_experiments[i]['base-line']

        # this does not work for the log-score for some reason :/ different data type, I guess. Therefore, treated below
    averaged_log_scores = []
    mean_residuals=[]
    std_residuals=[]
    df['residuals']=df['residuals']/repeated
    df['base-line']=df['base-line']/repeated
    for j in range(len(df.index)):
        averaged_log_scores.append(np.mean(df['logscore'].iloc[j]))
        mean_residuals.append(np.mean(df['residuals'].iloc[j]))
        std_residuals.append(np.std(df['residuals'].iloc[j]))
    df['logscore']=averaged_log_scores
    df['residuals']= mean_residuals
    df['mean-residuals']= pd.Series(mean_residuals, index=df.index)
    df['std-residuals']=pd.Series(std_residuals, index=df.index)
    return df

def load_experiments(ini_file_path,date_exp):
    Config.read(ini_file_path)
    Config.sections()

    models = dict(Config.items('inference'))

    number_data_points = ConfigSectionMap("test-data")['data-points'].split(',')
    list_noise_variance = ConfigSectionMap("test-data")['observation_noise'].split(',')
    repeat =  ConfigSectionMap("others")['repeat']
    number_test_points =  ConfigSectionMap("others")['number-test-points']
    test_problems =  ConfigSectionMap("test-data")['test-problems'].split(',')
    total_steps_outer = ConfigSectionMap("MCMC")["total-steps-outer"]

    condition=[]
    directory="results/"+date_exp
    un_averaged_frames=[]
    for index in range(int(repeat)):
        frames =[]
        for key in models:
            all_inf_string = ConfigSectionMap("inference")[key]
            list_of_infer= all_inf_string.split(";")
            for infer in list_of_infer:
                for test_problem in test_problems:
                    for noise in list_noise_variance:
                        for n in number_data_points:

                            experiment_name = key+'_'+infer+'_'+total_steps_outer+'_'\
                                               + test_problem +'_'+noise+'_'+n+'_'+number_test_points+'_'+str(index)
                            output_file_name = directory+"/exp_"+ experiment_name
                            try:
                                df = pd.read_pickle(output_file_name)
                                df['model'] = pd.Series([key for _ in range(len(df.index))], index=df.index)
                                df['test_problem'] = pd.Series([test_problem for _ in range(len(df.index))], index=df.index)
                                df['noise'] = pd.Series([noise for _ in range(len(df.index))], index=df.index)
                                df['n'] = pd.Series([n for _ in range(len(df.index))], index=df.index)
                                df['repeat'] = pd.Series([index for _ in range(len(df.index))], index=df.index)
                                #df['mcmcm-step-index'] = pd.Series([i for i in range(len(total_steps_outer))], index=df.index)
                                frames.append(df)
                            except ValueError:
                                ("could not open "+output_file_name)
        un_averaged_frames.append(pd.concat(frames))
    return  average_frames(un_averaged_frames)









