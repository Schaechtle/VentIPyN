from all_registered_models import registered_models
from all_registered_problems import registered_problems
import os.path
import numpy as np

def generate_data(f, noise, n,number_of_testpoints):
    x_training = np.random.uniform(0,10,n)
    y_training = f(x_training)#+np.random.randn(0,noise,n)
    x_test = np.random.uniform(0,10,number_of_testpoints)
    f_test = f(x_test)
    y_test = f_test+np.random.normal(0,noise,number_of_testpoints)
    f_error = np.abs(f_test - y_test)
    return x_training,y_training,x_test,y_test,f_test,f_error



def experiment(key, infer,total_steps_outer, test_problem, noise, n, index_str,number_test_points):
    output_file_name = "results/exp_"+key+'_'+infer+'_'+total_steps_outer+'_'+test_problem+'_'+noise+'_'+n+'_'+number_test_points+'_'+index_str
    if os.path.isfile(output_file_name):
        #print(output_file_name)
        pass
    else:
        model = registered_models[key]
        data_func = registered_problems[test_problem]
        x_training,y_training,x_test,y_test,f_test,f_error=generate_data(data_func,float(noise),int(n),int(number_test_points))
        #import pdb; pdb.set_trace()
        df = model.run(x_training,y_training,x_test,y_test,f_test,f_error,infer,total_steps_outer)
        df.to_pickle(output_file_name)


