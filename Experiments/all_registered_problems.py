__author__ = 'ulli'
import numpy as np

def function_linear(a,x):
    return a*x

registered_problems={}
registered_problems['function-linear']=lambda x: x*2
registered_problems['function-se']=lambda x: np.sin(x)+np.sqrt(x)
registered_problems['function-per']=lambda x: np.sin(3*x)
registered_problems['function-lin+per']=lambda x: 2* x + 2* np.sin(5*x)
registered_problems['function-linxper']=lambda x: 2* x * np.sin(5*x)
registered_problems['function-quadratic']=lambda x: x**2
