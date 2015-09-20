import ConfigParser
import argparse
from util import Config,ConfigSectionMap
from gp_experiment import experiment
import datetime
def f_exp(arg_list):
    experiment(*arg_list)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--ini-file-path", type=str, default="experiment'ini",
            help="Path for configuration file")
    parser.add_argument("-d", metavar="DATE", type=str, default=datetime.date.today(),
            help="Date")
    parser.add_argument("--local", action="store_true",
            help="Do not use parallelism")
    parser.add_argument("--cores", type=int, default=60,
            help="Number of cores")
    parser.add_argument("-m", metavar="MESSAGE", type=str, default=" no message ", help="Message")

    ns = parser.parse_args()
    ini_file_path = ns.ini_file_path
    date_exp = ns.d
    run_locally = ns.local
    cores = ns.cores
    message = ns.m

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
                            #experiment(key, infer, total_steps_outer, test_problem, number_test_points, date_exp, index_str, n=None,noise=None)
                            condition.append([key,infer,total_steps_outer,test_problem,number_test_points,str(date_exp),str(index),n,noise])

    if run_locally:
        for item in condition:
            f_exp(item)
    else:
        from multiprocessing import Pool
        pool = Pool(cores)
        pool.map(f_exp, condition)
    with open("meta-file.txt", "a") as output:
        output.write("-------------------------\n")
        output.write(str(datetime.date.today())+"\n")
        output.write(ini_file_path+"\n")
        output.write("writing to directory: "+str(date_exp)+"\n")
        output.write("Info: "+message+"\n")
        output.write("\n")

