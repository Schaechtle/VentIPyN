

import pandas as pd
import sys

from sympy.parsing.sympy_parser import parse_expr
from sympy import  *
max_number_same=6
LIN,SE,WN,PER,C = symbols(("LIN","SE","WN","PER","C"))
def simplify(eq_str):
    eq_str=eq_str.replace("x","*")
    expr = parse_expr(eq_str).expand()
    for i in range(max_number_same,0,-1):
        expr = expr.subs({SE**i:SE,LIN*i:LIN,WN*SE:WN,WN*PER:WN,WN*LIN*LIN:LIN*WN})
    return str(expr)

'''
date_str = "2015-05-28"
df_name="exp_venture-cov-learning-unif_(do (mh (quote grammar) one 1) (repeat 1 (do (mh (quote parameter) one 10))) ):1_5000_function-linxper_0.7_500_10_7"

for i in range(1,len(sys.argv)):
            if str(sys.argv[i])=="-n":
                name= str(sys.argv[i+1])

            if str(sys.argv[i])=="--cut-off":
                cut= float(sys.argv[i+1])
            if str(sys.argv[i])=="-d":
                date_str= str(sys.argv[i+1])
            if str(sys.argv[i])=="-f":
                ini_file= str(sys.argv[i+1])


df = pd.read_pickle("results/2015-05-28/exp_venture-cov-learning-unif_(do (mh (quote grammar) one 1) (repeat 1 (do (mh (quote parameter) one 10))) ):1_5000_function-linxper_0.7_500_10_0")





max_number_same=3








import time


start = time.time()



df['Covariance Structure']=df['Covariance Structure'].apply(simplify)


end = time.time()

print(end-start)
print(df['Covariance Structure'])
'''

