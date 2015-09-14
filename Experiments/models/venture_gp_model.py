__author__ = 'ulli'
from init_gp_ripl import init_gp_ripl
from tools import array
import numpy as np
import pandas as pd


class Venture_GP_model():
    def __init__(self):

        self.inf_strings=[]
        self.inf_cycles=[]
        self.record_interpretation = False
    def run(self,x_training,y_training,x_test,y_test,f_test,f_error,inf_string,outer_mcmc_steps):
        self.ripl= init_gp_ripl()
        self.get_inf_string(inf_string)
        self.make_gp(self.ripl)
        #self.get_prior()
        self.makeObservations(x_training,y_training)
        global_logs_core,residuals,base_line,mcmc_index,mcmc_cycle,interpretations,parameters = self.run_inference(x_test,f_test,f_error,outer_mcmc_steps)
        self.ripl.clear()
        if self.record_interpretation:
            return pd.DataFrame({'residuals':residuals,'logscore':global_logs_core,'base-line':base_line,'index':mcmc_index,'cycle':mcmc_cycle,'Parameters':parameters,'Covariance Structure':interpretations})
        else:
            return pd.DataFrame({'residuals':residuals,'logscore':global_logs_core,'base-line':base_line,'index':mcmc_index,'cycle':mcmc_cycle,'Parameters':parameters})

    def make_gp(self, ripl):
        raise ValueError('Covariance Structure not defined')
        pass

    def run_inference(self,x_test,f_test,f_error,outer_mcmc_steps):
        global_logs_core=[]
        residuals=[]
        residuals_inter=[]
        base_line =[]
        mcmc_index = []
        mcmc_cycle = []
        current_index = 0
        interpreation =[]
        parameters=[] # list of lists
        assert len(self.inf_cycles)==len(self.inf_strings)
        step_counter = 0
        for i in range(int(outer_mcmc_steps)):
            for j in range(len(self.inf_strings)):
                for k in range(int(self.inf_cycles[j])):
                    #import pdb; pdb.set_trace()
                    if self.inf_strings[j].startswith("ds"):
                        self.dynamicInferece(self.scope_instruction,self.n_steps)
                    else:
                        self.ripl.infer(self.inf_strings[j])
                    current_global_posterior= self.ripl.infer("global_posterior")

                    global_logs_core.append(current_global_posterior)
                    sampleString=self.genSamples(x_test)
                    y_posterior = self.ripl.sample(sampleString)
                    residuals.append(f_test - y_posterior)
                    parameters.append(self.collect_parameters(self.ripl))
                    #assert current_global_posterior[0] <= 0

                    step_counter+=1



                    base_line.append(f_error)
                    mcmc_index.append(current_index)
                    current_index+=1
                    mcmc_cycle.append(j)
                    if self.record_interpretation:
                        interpreation.append(self.ripl.sample("(covariance_string cov_structure)"))
        return global_logs_core,residuals,base_line,mcmc_index,mcmc_cycle,interpreation,parameters

    def get_inf_string(self,inf_string):
        self.inf_strings=[]
        self.inf_cycles=[]

        for inf_routine in inf_string.split(','):
            if inf_routine.startswith("ds"):
                self.scope_instruction,self.n_steps=inf_routine[2:].split('_')
                self.inf_strings.append(inf_routine)
                self.inf_cycles.append('1')

            else:
                inf_step,repeat_step=inf_routine.split(':')
                self.inf_strings.append(inf_step)
                self.inf_cycles.append(repeat_step)
        # e.g.:
        # venture-gp-LIN=(mh (quote parameter) 0 1):10,(hmc (quote hypers) 0 0.1 1 1):10
        # venture-gp-SE=(mh (quote parameter) 0 1):10

    def dynamicInferece(self,scope_instruction,n_steps):
        kernel_functions_used = self.ripl.sample(scope_instruction)
        for label in kernel_functions_used:
            self.ripl.infer("(mh (quote "+str(label)+" ) one "+n_steps+")")


    def genSamples(self,x):
        sampleString='(gp (array '
        for i in range(len(x)):
            sampleString+= str(x[i]) + ' '
        sampleString+='))'
        #print(sampleString)
        return sampleString

    def makeObservations(self,x,y):
        xString = self.genSamples(x)
        self.ripl.observe(xString, array(y.tolist()))
    def collect_parameters(self,_):
        return []







