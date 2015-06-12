import seaborn as sns
import pylab as pl
#from plotting import load_experiments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as scio
from models.covFunctions import *

from venture import shortcuts
import sys
sys.path.append('../SPs/')
import venture.lite.types as t
from venture.lite.function import VentureFunction
import gp_der
import gpmem2 as gpmem

from models.tools import array
from random import random as rand
from venture.lite.builtin import deterministic_typed

## UNINTERESTING PREPARATION

# Plotting stuff
sns.set(font_scale=2)

def plot_current(fname_prefix):
    figwidth = 10
    figheight = 10
    fig = plt.figure(figsize=(figwidth,figheight), dpi=200)
    for i in range(100):
        xs = np.sort(np.random.uniform(-3,3,200))
        ys = ripl.sample(get_emu_expr(xs))
        pl.plot(xs, ys, c="red", alpha=0.008, linewidth=2)
    plt.axis((-2,2,-1,3))
    x2plot = np.linspace(-3,3,1000)
    f2plot = f_noiseless(x2plot)
    plt.plot(x2plot,f2plot,color='blue')
    plt.scatter(data_xs, data_ys, color='black', marker='x', s=50, edgecolor='black', linewidth='1.5')   

    fig.savefig(fname_prefix + '.svg', dpi=fig.dpi)
    fig.savefig(fname_prefix + '.png', dpi=fig.dpi)
    return fig

# Other miscellaneous utilities
def array(xs):
    return t.VentureArrayUnboxed(np.array(xs),  t.NumberType())

def make_ripl():
    ripl = shortcuts.make_lite_church_prime_ripl()
    ripl.bind_foreign_sp("make_gp_part_der",gp_der.makeGPSP)
    ripl.bind_foreign_sp('gpmem', gpmem.gpmemSP)
    ripl.assume('make_const_func', VentureFunction(makeConstFunc, [t.NumberType()], constantType))
    ripl.assume('zero', "(apply_function make_const_func 0)")
    ripl.assume("func_plus", makeLiftedAdd(lambda x1, x2: x1 + x2))
    ripl.assume('make_se',VentureFunction(makeSquaredExponential,[t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
    ripl.assume('make_noise',VentureFunction(makeNoise,[t.NumberType()], t.AnyType("VentureFunction")))
    return ripl


## GENERATING THE DATA

# The true regression function and noise model
def f_noiseless(x):
    return 0.3 + 0.4*x + 0.5*np.sin(2.7*x) + (1.1/(1+x**2))
@np.vectorize
def f_noisy(x):
    p_outlier = 0.1
    stdev = (1.0 if rand() < p_outlier else 0.1)
    return np.random.normal(f_noiseless(x), stdev)

# Generate and save a data set
n = 100
data_xs = np.random.normal(0,1,n)
data_ys = f_noisy(data_xs)

experiment_id = 's'
np.save('syndata/x_%s.npy' % (experiment_id,), data_xs)
np.save('syndata/y_%s.npy' % (experiment_id,), data_ys)


## SETTING UP THE MODEL
ripl = make_ripl()
# Hyperparameters
ripl.assume('alpha_sf', "(tag 'hyperhyper 0 (gamma 7 1))")
ripl.assume('beta_sf', "(tag 'hyperhyper 2 (gamma 1 0.5))")
ripl.assume('alpha_l', "(tag 'hyperhyper 1 (gamma 7 1))")
ripl.assume('beta_l', "(tag 'hyperhyper 3 (gamma 1 0.5))")
ripl.assume('alpha_s', "(tag 'hyperhyper 4 (gamma 7 1))")
ripl.assume('beta_s', "(tag 'hyperhyper 5 (gamma 1 0.5))")
# Parameters of the covariance function
ripl.assume('sf',"(tag 'hyper 0 (log (gamma alpha_sf beta_sf )))")
ripl.assume('l',"(tag 'hyper 1 (log (gamma alpha_l beta_l )))")
ripl.assume('sigma',"(tag 'hyper 2 (uniform_continuous 0 2 ))")
# The covariance function
ripl.assume('se', '(apply_function make_se sf l)')
ripl.assume('wn','(apply_function make_noise sigma)')
ripl.assume('composite_covariance', '(apply_function func_plus se wn)')

# Set up the gpmem
def f_restr(x):
    matches = np.argwhere(np.abs(data_xs - x) < 1e-6)
    if matches.size == 0:
        raise Exception('Illegal query')
    else:
        assert matches.size == 1
        i = matches[0,0]
        return data_ys[i]
f_restr_sp = deterministic_typed(f_restr, [t.NumberType()], t.NumberType())
ripl.bind_foreign_sp('f_restr', f_restr_sp)
ripl.assume('compute_and_emu', '(gpmem f_restr composite_covariance)')

# Shortcut for creating the Venture expression to sample f_emu on a list of x values
def get_emu_expr(xs):
    return '((second compute_and_emu) (array %s))' % ' '.join(str(x) for x in xs)


## RUNNING THE EXPERIMENTS

# Plot the prior, before any observations
plot_current('neal_example_figs/neal_se_1%s' % (experiment_id,))

# Observe the data points (i.e., compute some values of f_restr).
for x in data_xs:
    ripl.predict('((first compute_and_emu) %f)' % (x,))
ripl.infer('(incorporate)')

# Plot the posterior after observations, but before inference on the hyperparameters.
plot_current('neal_example_figs/neal_se_2%s' % (experiment_id,))

# Infer the hyperparameters
ripl.infer("(repeat 100 (do (mh 'hyperhyper one 2) (mh 'hyper one 1)))")
# Debug message
print "sf = %s, l = %s, sigma = %s" % (ripl.sample('sf'), ripl.sample('l'), ripl.sample('sigma'))

# Plot the posterior after inference on the hyperparameters
plot_current('neal_example_figs/neal_se_3%s' % (experiment_id,))

