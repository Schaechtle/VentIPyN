import ConfigParser
import sys
from util import Config,ConfigSectionMap
from gp_experiment import experiment
import datetime
def f_exp(arg_list):
    experiment(*arg_list)

if __name__ == '__main__':

    ini_file_path="experiment.ini"
    date_exp=datetime.date.today()
    run_locally=False
    cores  = 60
    message = " no message "
    for i in range(1,len(sys.argv)):
            if str(sys.argv[i])=="-f":
                ini_file_path = str(sys.argv[i+1])
            if str(sys.argv[i])=="-d":
                date_exp = str(sys.argv[i+1])
            if str(sys.argv[i])=="--local":
                run_locally = True
            if str(sys.argv[i])=="--cores":
                cores  = str(sys.argv[i+1])
            if str(sys.argv[i])=="-m":
                message = str(sys.argv[i+1])

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
    for key in models:
        all_inf_string = ConfigSectionMap("inference")[key]
        list_of_infer= all_inf_string.split(";")
        for infer in list_of_infer:
            for test_problem in test_problems:
                for noise in list_noise_variance:
                    for n in number_data_points:
                        for index in range(int(repeat)):
                            condition.append([key,infer,total_steps_outer,test_problem,noise,n,str(index),number_test_points,str(date_exp)])

    if run_locally:
        for item in condition:
            f_exp(item)
    else:
        from multiprocessing import Pool
        pool = Pool(cores)
        pool.map(f_exp, condition)
    with open("meta-file.txt") as input:
        # Read non-empty lines from input file
        lines = [line for line in input if line.strip()]
    with open("meta-file.txt", "w") as output:
        for line in lines:
            output.write(line)
        output.write("-------------------------\n")
        output.write(str(datetime.date.today())+"\n")
        output.write(ini_file_path+"\n")
        output.write("writing to directory: "+str(date_exp)+"\n")
        output.write("Info: "+message+"\n")
        output.write("\n")

