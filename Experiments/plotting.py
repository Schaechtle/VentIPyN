import pandas as pd
from util import Config,ConfigSectionMap
import numpy as np
import os.path
from simplify_kernel import simplify

def average_frames(repeated_experiments):
    repeated = len(repeated_experiments)
    assert repeated == int(ConfigSectionMap("others")['repeat'])
    df = repeated_experiments[0]
    df['residuals']=np.abs(df['residuals'])

    for i in range(1,repeated):
        df['logscore']+=repeated_experiments[i]['logscore']
        df['residuals']+=np.abs(repeated_experiments[i]['residuals'])
        df['inter-residuals']+=np.abs(repeated_experiments[i]['inter-residuals'])
        df['base-line']+=np.abs(repeated_experiments[i]['base-line'])
    averaged_log_scores = []
    mean_residuals=[]

    mean_baseline=[]
    df['residuals']=df['residuals']/repeated

    df['base-line']=df['base-line']/repeated
    for j in range(len(df.index)):
        averaged_log_scores.append(np.mean(df['logscore'].iloc[j]))
        mean_residuals.append(np.mean(df['residuals'].iloc[j]))
        #mean_inter_residuals.append(np.mean(df['inter-residuals'].iloc[j]))
        mean_baseline.append(np.mean(df['base-line'].iloc[j]))
        #std_residuals.append(np.std(df['residuals'].iloc[j]))
    df['logscore']=averaged_log_scores
    df['residuals']= mean_residuals
    #df['inter-residuals']= mean_inter_residuals
    df['base-line']=mean_baseline
    df['mean-residuals']= pd.Series(mean_residuals, index=df.index)

    #df['std-residuals']=pd.Series(std_residuals, index=df.index)
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
    df_experiment = average_frames(un_averaged_frames)
    df_experiment.to_pickle("results/experiment_"+date_exp)
    return  df_experiment




def get_last_n_residuals(ini_file_path,date_exp,n_last_residuals):
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
    base_line = []
    residuals=[]
    model=[]
    problem=[]
    noise_level=[]
    n_training_data=[]
    for index in range(int(repeat)):
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
                                df_all = pd.read_pickle(output_file_name)
                                df=df_all.tail(n_last_residuals)

                                base_line.extend(df['base-line'].iloc[0])
                                #index=[]
                                for j in range(len(df.index)):
                                    residuals.extend(df['residuals'].iloc[j])

                                    model.extend([key for _ in range(int(number_test_points))])
                                    problem.extend([test_problem for _ in range(int(number_test_points))])
                                    noise_level.extend([noise for _ in range(int(number_test_points))])
                                    n_training_data.extend([n for _ in range(int(number_test_points))])
                                    #index.append([j for _ in range(int(number_test_points))])


                                    #residuals= np.concatenate(residuals,df['residuals'].iloc[j])


                                '''
                                df['model'] = pd.Series([key for _ in range(len(df.index))], index=df.index)
                                df['test_problem'] = pd.Series([test_problem for _ in range(len(df.index))], index=df.index)
                                df['noise'] = pd.Series([noise for _ in range(len(df.index))], index=df.index)
                                df['n'] = pd.Series([n for _ in range(len(df.index))], index=df.index)
                                df['repeat'] = pd.Series([index for _ in range(len(df.index))], index=df.index)
                                #df['mcmcm-step-index'] = pd.Series([i for i in range(len(total_steps_outer))], index=df.index)
                                '''
                            except ValueError:
                                ("could not open "+output_file_name)
    return pd.DataFrame({'residuals':residuals,'model':model,'test_problem':problem,'noise':noise_level,'n':n_training_data}),np.mean(base_line)

def get_dataFrame(date_experiment,ini_file_path):
    file_path="results/experiment_"+date_experiment
    if os.path.isfile(file_path):
        return pd.read_pickle(file_path)
    else:
        return load_experiments(ini_file_path,date_experiment)

def get_dataFrame_median(date_experiment,ini_file_path):
    file_path="results/median_residual_"+date_experiment
    if os.path.isfile(file_path):
        print("loading existing df")
        return pd.read_pickle(file_path)
    else:
        print("creating df from scratch")
        return load_median_experiments(ini_file_path,date_experiment)

def load_median_experiments(ini_file_path,date_exp):
    Config.read(ini_file_path)
    Config.sections()

    models = dict(Config.items('inference'))
    number_data_points = ConfigSectionMap("test-data")['data-points'].split(',')
    list_noise_variance = ConfigSectionMap("test-data")['observation_noise'].split(',')
    repeat =  ConfigSectionMap("others")['repeat']
    number_test_points =  ConfigSectionMap("others")['number-test-points']
    test_problems =  ConfigSectionMap("test-data")['test-problems'].split(',')
    total_steps_outer = ConfigSectionMap("MCMC")["total-steps-outer"]


    directory="results/"+date_exp
    residual_list = []

    df_list= []
    index_it=0
    for key in models:
        all_inf_string = ConfigSectionMap("inference")[key]
        list_of_infer= all_inf_string.split(";")
        for infer in list_of_infer:
            for test_problem in test_problems:
                for noise in list_noise_variance:
                    for n in number_data_points:
                        df = pd.DataFrame()
                        print(index_it)
                        index_it+=1
                        for index in range(int(repeat)):
                            experiment_name = key+'_'+infer+'_'+total_steps_outer+'_'\
                                               + test_problem +'_'+noise+'_'+n+'_'+number_test_points+'_'+str(index)
                            output_file_name = directory+"/exp_"+ experiment_name
                            try:
                                df = pd.read_pickle(output_file_name)

                                residual_list.append(np.abs(df['residuals'].values))

                            except ValueError:
                                ("could not open "+output_file_name)
                        res_matrix = np.matrix(residual_list)
                        residual_list = []
                        df['median-residual']=pd.Series([np.median(res_matrix[:,i].tolist()) for i in df.index ])
                        df['model'] = pd.Series([key for _ in range(len(df.index))], index=df.index)
                        df['test_problem'] = pd.Series([test_problem for _ in range(len(df.index))], index=df.index)
                        df['noise'] = pd.Series([noise for _ in range(len(df.index))], index=df.index)
                        df['n'] = pd.Series([n for _ in range(len(df.index))], index=df.index)
                        df['repeat'] = pd.Series([index for _ in range(len(df.index))], index=df.index)
                        df_list.append(df)
    df_experiment=pd.concat(df_list)
    df_experiment.to_pickle("results/median_residual_"+date_exp)
    return  df_experiment

def get_posterior_structure(date_experiment,ini_file_path):
    file_path="results/structure_posterior"+date_experiment
    if os.path.isfile(file_path):
        return pd.read_pickle(file_path)
    else:
        return load_posterior_structure(ini_file_path,date_experiment)
def load_posterior_structure(ini_file_path,date_exp):
    Config.read(ini_file_path)
    Config.sections()

    models = dict(Config.items('inference'))
    number_data_points = ConfigSectionMap("test-data")['data-points'].split(',')
    list_noise_variance = ConfigSectionMap("test-data")['observation_noise'].split(',')
    repeat =  ConfigSectionMap("others")['repeat']
    number_test_points =  ConfigSectionMap("others")['number-test-points']
    test_problems =  ConfigSectionMap("test-data")['test-problems'].split(',')
    total_steps_outer = ConfigSectionMap("MCMC")["total-steps-outer"]


    directory="results/"+date_exp
    residual_list = []

    df_list= []
    index_it=0
    for key in models:
        all_inf_string = ConfigSectionMap("inference")[key]
        list_of_infer= all_inf_string.split(";")
        for infer in list_of_infer:
            for test_problem in test_problems:
                for noise in list_noise_variance:
                    for n in number_data_points:
                        df = pd.DataFrame()

                        for index in range(int(repeat)):
                            print(index_it)
                            index_it+=1
                            experiment_name = key+'_'+infer+'_'+total_steps_outer+'_'\
                                               + test_problem +'_'+noise+'_'+n+'_'+number_test_points+'_'+str(index)
                            output_file_name = directory+"/exp_"+ experiment_name
                            try:
                                df = pd.read_pickle(output_file_name)
                                df['Covariance Structure']=df['Covariance Structure'].apply(simplify)
                                df['model'] = pd.Series([key for _ in range(len(df.index))], index=df.index)
                                df['test_problem'] = pd.Series([test_problem for _ in range(len(df.index))], index=df.index)
                                df['noise'] = pd.Series([noise for _ in range(len(df.index))], index=df.index)
                                df['n'] = pd.Series([n for _ in range(len(df.index))], index=df.index)
                                df['repeat'] = pd.Series([index for _ in range(len(df.index))], index=df.index)
                                df_list.append(df)

                            except ValueError:
                                ("could not open "+output_file_name)

    df_experiment=pd.concat(df_list)
    df_experiment.to_pickle("results/structure_posterior"+date_exp)
    return  df_experiment