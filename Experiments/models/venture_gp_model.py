__author__ = 'ulli'
from init_gp_ripl import init_gp_ripl
from tools import array
import numpy as np
import pandas as pd


class Venture_GP_model():
    def __init__(self):

        self.inf_strings=[]
        self.inf_cycles=[]

    def run(self,x_training,y_training,x_test,y_test,f_test,f_error,inf_string,outer_mcmc_steps):
        self.ripl= init_gp_ripl()
        self.get_inf_string(inf_string)
        self.make_gp(self.ripl)
        #self.get_prior()
        self.makeObservations(x_training,y_training)
        global_logs_core,residuals,base_line,mcmc_index,mcmc_cycle = self.run_inference(x_test,f_test,f_error,outer_mcmc_steps)
        self.ripl.clear()
        return pd.DataFrame({'residuals':residuals,'logscore':global_logs_core,'base-line':base_line,'index':mcmc_index,'cycle':mcmc_cycle})

    def make_gp(self, ripl):
        raise ValueError('Covariance Structure not defined')
        pass

    def run_inference(self,x_test,f_test,f_error,outer_mcmc_steps):
        global_logs_core=[]
        residuals=[]
        base_line =[]
        mcmc_index = []
        mcmc_cycle = []
        current_index = 0
        assert len(self.inf_cycles)==len(self.inf_strings)
        for i in range(int(outer_mcmc_steps)):
            for j in range(len(self.inf_strings)):
                for k in range(int(self.inf_cycles[j])):
                    #import pdb; pdb.set_trace()
                    self.ripl.infer(self.inf_strings[j])
                    global_logs_core.append(self.ripl.infer("global_posterior"))
                    sampleString=self.genSamples(x_test)
                    y_posterior = self.ripl.sample(sampleString)
                    residuals.append(np.abs(f_test - y_posterior))
                    base_line.append(f_error)
                    mcmc_index.append(current_index)
                    current_index+=1
                    mcmc_cycle.append(j)
        return global_logs_core,residuals,base_line,mcmc_index,mcmc_cycle

    def get_inf_string(self,inf_string):
        self.inf_strings=[]
        self.inf_cycles=[]
        for inf_routine in inf_string.split(','):
            inf_step,repeat_step=inf_routine.split(':')
            self.inf_strings.append(inf_step)
            self.inf_cycles.append(repeat_step)
        # e.g.:
        # venture-gp-LIN=(mh (quote parameter) 0 1):10,(hmc (quote hypers) 0 0.1 1 1):10
        # venture-gp-SE=(mh (quote parameter) 0 1):10

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







