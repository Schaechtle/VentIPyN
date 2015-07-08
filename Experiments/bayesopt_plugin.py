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
from venture.lite.function import VentureFunction
from venture.lite.builtin import deterministic_typed
import gp_der
import gpmem2 as gpmem
import pickle
import collections

PlotData = collections.namedtuple('PlotData', ['sf1', 'l1', 'Xseen', 'Yseen'])

def __venture_start__(ripl, *args):

    print "Args:", args

    argmaxSP = deterministic_typed(np.argmax, [t.HomogeneousArrayType(t.NumberType())], t.NumberType())
    absSP = deterministic_typed(abs, [t.NumberType()], t.NumberType())
    make_se_SP = deterministic_typed(lambda sf, l:
            VentureFunction(
                squared_exponential(sf, l), sp_type=squaredExponentialType,derivatives={0:squared_exponential_der_l(sf,l),1:squared_exponential_der_sf(sf,l)},name="SE",parameter=[sf,l]),
            [t.NumberType(), t.NumberType()], t.AnyType("VentureFunction"))
    ripl.bind_foreign_sp('make_gp_part_der', gp_der.makeGPSP)
    ripl.bind_foreign_sp('gpmem', gpmem.gpmemSP)
    ripl.bind_foreign_inference_sp('argmax_of_array', argmaxSP)
    ripl.bind_foreign_sp('abs', absSP)
    ripl.bind_foreign_sp('make_se', make_se_SP)

    def f(x):
        f.count += 1 # A tracker for how many times I am called
        print "[COUNT] Number of calls to secret function: %d" % (f.count,)
        return (0.2 + np.exp(-0.1*abs(x-2))) * np.cos(0.4*x)
    f.count = 0
    f_sp = deterministic_typed(f, [t.NumberType()], t.NumberType())
    ripl.bind_foreign_sp('blackbox_f', f_sp)

    PLOT_DATAS = []
    class CollectPlotDataCallback(object):
        def __call__(self, inferrer, sf_, l_, stats_):
            sf = sf_[0]['value']
            l = l_[0]['value']
            def unpack(p):
                return [n['value'] for n in p['value']]
            all_pairs = [unpack(p) for p in stats_[0]['value'][1]['value']]
            (Xseen, Yseen) = zip(*all_pairs)
            plot_data = PlotData(sf, l, Xseen, Yseen)
            if len(PLOT_DATAS) > 0 and len(PLOT_DATAS[-1]) == 1:
                PLOT_DATAS[-1].append(plot_data)
            else:
                PLOT_DATAS.append([plot_data])

    collect_plot_data_callback = CollectPlotDataCallback()
    ripl.bind_callback("collect_plot_data", collect_plot_data_callback)

    class DumpPlotDataCallback(object):
        def __call__(self, inferrer):
            log_fname = 'log_with_gpmem/plot_data.pkl'
            print "Logging to %s" % (log_fname)
            with open(log_fname, 'wb') as f:
                pickle.dump(PLOT_DATAS, f)
            print "Done."

    dump_plot_data_callback = DumpPlotDataCallback()
    ripl.bind_callback("dump_plot_data", dump_plot_data_callback)
