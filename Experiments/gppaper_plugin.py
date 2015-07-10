# Path hacks because loading as a Venture plugin has kinks
import sys
sys.path.append('.')

sys.path.append('../SPs/')
import seaborn as sns
import pylab as pl
#from plotting import load_experiments
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio
from models.covFunctions import *
from models.tools import array

from venture import shortcuts
import venture.lite.types as t
import venture.lite.sp as sp
from venture.lite.function import VentureFunction
from venture.lite.builtin import deterministic_typed
import gp_der
import gpmem2 as gpmem
import pickle
import collections
from numpy.random import random as rand

BayesOptPlotData = collections.namedtuple('BayesOptPlotData', ['sf1', 'l1', 'Xseen', 'Yseen'])
NealPlotData = collections.namedtuple('NealPlotData', ['name', 'sf', 'l', 'sigma', 'Xseen', 'Yseen'])

@np.vectorize
def neal_f_noiseless(x):
    return 0.3 + 0.4*x + 0.5*np.sin(2.7*x) + (1.1/(1+x**2))


def __venture_start__(ripl, *args):

    # External SPs
    argmaxSP = deterministic_typed(np.argmax, [t.HomogeneousArrayType(t.NumberType())], t.NumberType())
    absSP = deterministic_typed(abs, [t.NumberType()], t.NumberType())
    make_se_SP = deterministic_typed(lambda sf, l:
            VentureFunction(
                squared_exponential(sf, l), sp_type=squaredExponentialType,
                derivatives={0:squared_exponential_der_l(sf,l),1:squared_exponential_der_sf(sf,l)},
                name="SE", parameter=[sf,l]),
            [t.NumberType(), t.NumberType()], t.AnyType("VentureFunction"))
    make_whitenoise_SP = deterministic_typed(lambda s:
            VentureFunction(noise(s), sp_type=covfunctionType,derivatives={0:noise_der(s)},name="WN",parameter=[s]),
        [t.NumberType()], t.AnyType("VentureFunction"))

    add_funcs_SP = deterministic_typed(makeLiftedAdd(lambda x1, x2: x1 + x2),
        [t.AnyType("VentureFunction"), t.AnyType("VentureFunction")],
        t.AnyType("VentureFunction"))

    ripl.bind_foreign_sp('make_gp_part_der', gp_der.makeGPSP)
    ripl.bind_foreign_sp('gpmem', gpmem.gpmemSP)
    ripl.bind_foreign_inference_sp('argmax_of_array', argmaxSP)
    ripl.bind_foreign_sp('abs', absSP)
    ripl.bind_foreign_sp('make_squaredexp', make_se_SP)
    ripl.bind_foreign_sp('add_funcs', add_funcs_SP)
    ripl.bind_foreign_sp('make_whitenoise', make_whitenoise_SP)


    if args[0] == 'bayesopt':
        # The blackbox function
        def f(x):
            f.count += 1 # A tracker for how many times I am called
            print "[COUNT] Number of calls to secret function: %d" % (f.count,)
            return (0.2 + np.exp(-0.1*abs(x-2))) * np.cos(0.4*x)
        f.count = 0
        blackbox_f_sp = deterministic_typed(f, [t.NumberType()], t.NumberType())
        ripl.bind_foreign_sp('get_bayesopt_blackbox', deterministic_typed(
            lambda: sp.VentureSPRecord(blackbox_f_sp), [], sp.SPType([t.NumberType()], t.NumberType())))


        # Accumulator for plot datas
        BAYESOPT_PLOT_DATAS = []
        class CollectPlotDataCallback(object):
            def __call__(self, inferrer, sf_, l_, stats_):
                sf = sf_[0]['value']
                l = l_[0]['value']
                def unpack(p):
                    return [n['value'] for n in p['value']]
                all_pairs = [unpack(p) for p in stats_[0]['value'][1]['value']]
                (Xseen, Yseen) = zip(*all_pairs)
                plot_data = BayesOptPlotData(sf, l, Xseen, Yseen)
                if len(BAYESOPT_PLOT_DATAS) > 0 and len(BAYESOPT_PLOT_DATAS[-1]) == 1:
                    BAYESOPT_PLOT_DATAS[-1].append(plot_data)
                else:
                    BAYESOPT_PLOT_DATAS.append([plot_data])

        class DumpPlotDataCallback(object):
            def __call__(self, inferrer):
                log_fname = 'bayesopt_output/plot_data.pkl'
                print "Logging to %s" % (log_fname)
                with open(log_fname, 'wb') as f:
                    pickle.dump(BAYESOPT_PLOT_DATAS, f)
                print "Done."

        ripl.bind_callback("collect_bayesopt_plot_data", CollectPlotDataCallback())
        ripl.bind_callback("dump_bayesopt_plot_data", DumpPlotDataCallback())

    if args[0] == 'neal':
        # Noise model
        @np.vectorize
        def f_noisy(x):
            p_outlier = 0.1
            stdev = (1.0 if rand() < p_outlier else 0.1)
            return np.random.normal(neal_f_noiseless(x), stdev)

        # Generate and save a data set
        print "Generating Neal example data set"
        n = 100
        neal_data_xs = np.random.normal(0,1,n)
        neal_data_ys = f_noisy(neal_data_xs)

        np.save('neal_output/data_xs.npy', neal_data_xs)
        np.save('neal_output/data_ys.npy', neal_data_ys)

        ## The probe function
        def f_restr(x):
            matches = np.argwhere(np.abs(neal_data_xs - x) < 1e-6)
            if matches.size == 0:
                raise Exception('Illegal query')
            else:
                assert matches.size == 1
                i = matches[0,0]
                return neal_data_ys[i]
        f_restr_sp = deterministic_typed(f_restr, [t.NumberType()], t.NumberType())
        ripl.bind_foreign_sp('get_neal_prober', deterministic_typed(
            lambda: sp.VentureSPRecord(f_restr_sp), [], sp.SPType([t.NumberType()], t.NumberType())))

        get_neal_data_xs_SP = deterministic_typed(lambda: neal_data_xs, [], t.HomogeneousArrayType(t.NumberType()))
        ripl.bind_foreign_sp('get_neal_data_xs', get_neal_data_xs_SP)
        
        # Accumulator for plot datas
        NEAL_PLOT_DATAS = []
        class CollectPlotDataCallback(object):
            def __call__(self, inferrer, name_, sf_, l_, sigma_, stats_):
                name = name_[0]['value']
                sf = sf_[0]['value']
                l = l_[0]['value']
                sigma = sigma_[0]['value']
                def unpack(p):
                    return [n['value'] for n in p['value']]
                all_pairs = [unpack(p) for p in stats_[0]['value'][1]['value']]
                if len(all_pairs) > 0:
                    (Xseen, Yseen) = zip(*all_pairs)
                else:
                    (Xseen, Yseen) = ([], [])
                plot_data = NealPlotData(name, sf, l, sigma, Xseen, Yseen)
                NEAL_PLOT_DATAS.append(plot_data)

        class DumpPlotDataCallback(object):
            def __call__(self, inferrer):
                log_fname = 'neal_output/plot_datas.pkl'
                print "Logging to %s" % (log_fname)
                with open(log_fname, 'wb') as f:
                    pickle.dump(NEAL_PLOT_DATAS, f)
                print "Done."

        ripl.bind_callback("collect_neal_plot_data", CollectPlotDataCallback())
        ripl.bind_callback("dump_neal_plot_data", DumpPlotDataCallback())
