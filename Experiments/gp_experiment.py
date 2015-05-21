from all_registered_models import registered_models
from all_registered_problems import registered_problems
import os.path
import numpy as np

def saveXY_data(x_training,y_training,x_test,y_test,path):
    np.save(path+'_X_train',x_training)
    np.save(path+'_Y_train',y_training)
    np.save(path+'_X_test',x_test)
    np.save(path+'_Y_test',y_test)
def generate_data(f, noise, n,number_of_testpoints):
    x_training = np.random.uniform(0,10,n)
    y_training = f(x_training)#+np.random.randn(0,noise,n)
    x_test = np.random.uniform(10,12,number_of_testpoints)
    f_test = f(x_test)
    y_test = f_test+np.random.normal(0,noise,number_of_testpoints)
    f_error = f_test - y_test
    return x_training,y_training,x_test,y_test,f_test,f_error



def experiment(key, infer, total_steps_outer, test_problem, noise, n, index_str, number_test_points, date_exp):
    directory="results/"+date_exp
    experiment_name = key+'_'+infer+'_'+total_steps_outer+'_'\
                       +test_problem+'_'+noise+'_'+n+'_'+number_test_points+'_'+index_str
    output_file_name = directory+"/exp_"+ experiment_name
    if os.path.isfile(output_file_name):
        pass
    else:
        if not os.path.exists(directory):
            os.mkdir(directory)
            os.mkdir(directory+'/XYdata')
        model = registered_models[key]
        data_func = registered_problems[test_problem]
        x_training,y_training,x_test,y_test,f_test,f_error=generate_data(data_func,float(noise),int(n),int(number_test_points))
        df = model.run(x_training,y_training,x_test,y_test,f_test,f_error,infer,total_steps_outer)
        saveXY_data(x_training,y_training,x_test,y_test,directory+'/XYdata/'+experiment_name)
        df.to_pickle(output_file_name)



