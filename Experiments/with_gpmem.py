import seaborn as sns
import pylab as pl
#from plotting import load_experiments
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio
from models.covFunctions import *

from venture import shortcuts
import sys
sys.path.append('../SPs/')
import venture.lite.types as t
from venture.lite.function import VentureFunction
from venture.lite.builtin import deterministic_typed
import gp_der
import gpmem2 as gpmem
import pickle
from without_gpmem import PlotData

from models.tools import array


## PREPARATION

argmaxSP = deterministic_typed(np.argmax, [t.HomogeneousArrayType(t.NumberType())], t.NumberType())
absSP = deterministic_typed(abs, [t.NumberType()], t.NumberType())
def make_ripl():
    ripl = shortcuts.make_lite_church_prime_ripl()
    ripl.bind_foreign_sp('make_gp_part_der', gp_der.makeGPSP)
    ripl.bind_foreign_sp('gpmem', gpmem.gpmemSP)
    ripl.bind_foreign_sp('argmax_of_array', argmaxSP)
    ripl.bind_foreign_sp('abs', absSP)
    return ripl

ripl = make_ripl()

## SECRET FUNCTION

def f(x):
    f.count += 1 # A tracker for how many times I am called
    return (0.2 + np.exp(-0.1*abs(x-2))) * np.cos(0.4*x)
f.count = 0
f_sp = deterministic_typed(f, [t.NumberType()], t.NumberType())
ripl.bind_foreign_sp('f', f_sp)

## MODEL
ripl.assume('sf1','(tag (quote hyper) 0 (log (uniform_continuous 0 10)))')
ripl.assume('l1','(tag (quote hyper) 1 (log (uniform_continuous 0 10)))')
ripl.assume('make_se',VentureFunction(makeSquaredExponential,[t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
ripl.assume('se', '(apply_function make_se sf1 l1)')
ripl.assume('compute_and_emu', '(gpmem f se)')

## SEARCH STRATEGY

# TODO easy to use other search strategies such as gaussian drift
ripl.assume('get_uniform_candidate', '''
    (lambda (prev_xs) (uniform_continuous -20 20))
    ''')
ripl.assume('mc_argmax', '''
    (lambda (emulator prev_xs)
      ((lambda (candidate_xs)
         (lookup  candidate_xs
                  (argmax_of_array (mapv emulator candidate_xs))))
       (mapv (lambda (i) (get_uniform_candidate prev_xs))
            (linspace 0 15 14))))
    ''')

def get_plot_data(ripl):
    sf1 = ripl.sample('sf1')
    l1 = ripl.sample('l1')
    stats = ripl.infer('(extract_stats (second compute_and_emu))')
    assert len(stats[1]) > 0
    (xbest, ybest) = stats[0]
    all_pairs = stats[1]
    (Xseen, Yseen) = zip(*all_pairs)
    return PlotData(sf1, l1, Xseen, Yseen)

plot_datas = []
for i in range(15):
    xs = [ripl.sample('(uniform_continuous -20 20)') for _ in range(15)]
    ys = [ripl.sample('(lookup ((second compute_and_emu) (array %f)) 0)' % x) for x in xs]
    ripl.predict('((first compute_and_emu) %f)' % xs[np.argmax(ys)])
    # Once the GP copying stuff works, we will be able to replace the above with:
    # ripl.predict('((first compute_and_emu) (mc_argmax (lambda (x) (lookup ((second compute_and_emu) (array x)) 0)) (quote _)))')
    
    # For convenience in plotting, we want to have the info before and after
    # hyper inference
    row = []
    row.append(get_plot_data(ripl))
    ripl.infer('(mh (quote hyper) one 50)')
    row.append(get_plot_data(ripl))
    plot_datas.append(row)

print "Inferred sigma = %.2f, l = %.2f" % (ripl.sample('sf1'), ripl.sample('l1'))
stats = ripl.infer('(extract_stats (second compute_and_emu))')
(xbest, ybest) = stats[0]
all_pairs = stats[1]
(Xseen, Yseen) = zip(*all_pairs)
print "Best (x,y) pair: (%.2f, %.2f)" % (xbest, ybest)
print "Number of calls to f: %d" % (f.count,)

# Log the data for later plots
log_fname = 'log_with_gpmem/plot_data.pkl'
print "Logging to %s" % (log_fname)
with open(log_fname, 'wb') as f:
    pickle.dump(plot_datas, f)

