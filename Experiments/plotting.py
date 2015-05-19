import pandas as pd
from util import Config,ConfigSectionMap

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
    frames =[]
    for key in models:
        all_inf_string = ConfigSectionMap("inference")[key]
        list_of_infer= all_inf_string.split(";")
        for infer in list_of_infer:
            for test_problem in test_problems:
                for noise in list_noise_variance:
                    for n in number_data_points:
                        for index in range(int(repeat)):
                            experiment_name = key+'_'+infer+'_'+total_steps_outer+'_'\
                                               +test_problem+'_'+noise+'_'+n+'_'+number_test_points+'_'+str(index)
                            output_file_name = directory+"/exp_"+ experiment_name
                            try:
                                df = pd.read_pickle(output_file_name)
                                df['model'] = pd.Series([key for _ in range(len(df.index))], index=df.index)
                                df['test_problem'] = pd.Series([test_problem for _ in range(len(df.index))], index=df.index)
                                df['noise'] = pd.Series([noise for _ in range(len(df.index))], index=df.index)
                                df['n'] = pd.Series([n for _ in range(len(df.index))], index=df.index)
                                df[('repeat')] = pd.Series([index for _ in range(len(df.index))], index=df.index)

                                frames.append(df)

                            except:
                                print("could not open "+output_file_name)
    return  pd.concat(frames)




experimental_df = load_experiments("experiment_stub.ini","2015-05-18")





