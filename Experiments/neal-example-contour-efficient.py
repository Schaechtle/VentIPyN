import matplotlib

matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
# from plotting import load_experiments
import numpy as np
import pandas as pd
import scipy.io as scio
from models.covFunctions import *

from venture import shortcuts
import time
import sys

sys.path.append('../SPs/')
import venture.lite.types as t
from venture.lite.function import VentureFunction
import gp_der
import scipy.stats
from models.tools import array

figlength = 30
figheigth = 15
import random
import matplotlib.pyplot as plt
import os

figlength = 10
figheigth = 10

from scipy import stats

n_it = 100

particles = '100'
no = "cont_0_"
outlier_sigma = 1
dir_name= "temp"
syn_data_name="temp"
for i in range(1, len(sys.argv)):
    if str(sys.argv[i]) == "--no":  # structure posterior
        no = str(sys.argv[i + 1])
    if str(sys.argv[i]) == "-s":  # structure posterior
        outlier_sigma = float(sys.argv[i + 1])
    if str(sys.argv[i]) == "-i":  # structure posterior
        n_it = int(sys.argv[i + 1])
    if str(sys.argv[i]) == "-p":  # structure posterior
        particles = str(sys.argv[i + 1])
    if str(sys.argv[i]) == "--dir":  # structure posterior
        dir_name= str(sys.argv[i + 1])


for i in range(1, len(sys.argv)):
    if str(sys.argv[i]) == "-sp":  # structure posterior
        structure_posterior = True

def array(xs):
    return t.VentureArrayUnboxed(np.array(xs), t.NumberType())


def makeObservations(x, y, ripl):
    xString = genSamples(x)
    ripl.observe(xString, array(y))


def genSamples(x):
    sampleString = '(gp (array '
    for i in range(len(x)):
        sampleString += str(x[i]) + ' '
    sampleString += '))'
    # print(sampleString)
    return sampleString


def f(x):
    return 0.3 + 0.4 * x + 0.5 * np.sin(2.7 * x) + (1.1 / (1 + x ** 2))


def plot_hyper(n_iteration,dir_name,syn_data_name):
    assert(os.path.exists("syndata/"+syn_data_name))
    if not os.path.exists("syndata/"+dir_name):
        os.makedirs("syndata/"+dir_name)
    if not os.path.exists("/home/ulli/Dropbox/gpmemplots/syndata/"+dir_name):
        os.makedirs("/home/ulli/Dropbox/gpmemplots/syndata/"+dir_name)
    if not os.path.isfile("syndata/"+dir_name+"/after_parameters_"+str(n_iteration)):
        x = np.load("/home/ulli/Dropbox/gpmemplots/syndata/"+dir_name+"/data/x_s.npy")
        y = np.load("/home/ulli/Dropbox/gpmemplots/syndata/"+dir_name+"/data/y_s.npy")

        ripl = shortcuts.make_lite_church_prime_ripl()
        ripl.bind_foreign_sp("make_gp_part_der", gp_der.makeGPSP)
        ripl.assume('make_const_func', VentureFunction(makeConstFunc, [t.NumberType()], constantType))
        ripl.assume('zero', "(apply_function make_const_func 0)")



        # ripl.infer('(resample '+ particles +' )')

        ripl.assume("func_plus", makeLiftedAdd(lambda x1, x2: x1 + x2))
        ripl.assume('make_se',
                    VentureFunction(makeSquaredExponential, [t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
        ripl.assume('make_noise', VentureFunction(makeNoise, [t.NumberType()], t.AnyType("VentureFunction")))

        ripl.assume('alpha_sf','(tag (quote hyperhyper) 0 (gamma 7 1))')
        ripl.assume('beta_sf','(tag (quote hyperhyper) 2 (gamma 1 0.5))')
        ripl.assume('alpha_l','(tag (quote hyperhyper) 1 (gamma 7 1))')
        ripl.assume('beta_l','(tag (quote hyperhyper) 3 (gamma 1 0.5))')

        ripl.assume('sf', '(tag (quote hyper) 0  (gamma alpha_sf beta_sf ))')
        ripl.assume('l', '(tag (quote hyper) 1 (gamma alpha_l beta_l ))')

        ripl.assume('sf_l', '(log sf)')
        ripl.assume('l_l', '(log l)')

        ripl.assume('sigma', '(tag (quote hyper) 2 (uniform_continuous 0 2 ))')
        ripl.assume('l_sigma', '(log sigma)')

        ripl.assume('se', "(apply_function make_se sf_l l_l )")
        ripl.assume('wn', '(apply_function make_noise l_sigma  )')




        ripl.assume('gp', """(tag (quote model) 0
                                (make_gp_part_der zero (apply_function func_plus se wn  )
                                        )
                                     )""")

        makeObservations(x, y, ripl)

        rd = ripl.infer("(collect sf l sigma alpha_l beta_l alpha_sf beta_sf)")
        df_parameter_before = rd.asPandas()
        df_parameter_before.to_pickle("syndata/"+dir_name+"/before_parameters_"+str(n_iteration))

        ripl.infer("(repeat 200 (do (mh (quote hyperhyper) one 2) (mh (quote hyper) one 1)))")
        # ripl.infer("(mh (quote hyper) one 100)")






        rd = ripl.infer("(collect sf l sigma alpha_l beta_l alpha_sf beta_sf)")
        df_parameter_after = rd.asPandas()
        df_parameter_after.to_pickle("syndata/"+dir_name+"/after_parameters_"+str(n_iteration))


def plot_contours(df, name, n_iteration):
    df = df.loc[df['l'] < 10]
    df = df.loc[df['sf'] < 10]
    df = df.loc[df['l'] > -10]
    df = df.loc[df['sf'] > -10]
    joint_grid_plot("l", "sigma", df, name, n_iteration)
    joint_grid_plot("l", "sf", df, name, n_iteration)
    joint_grid_plot("sf", "sigma", df, name, n_iteration)
    joint_grid_plot("l", "sigma", df, name, n_iteration, False)
    joint_grid_plot("l", "sf", df, name, n_iteration, False)
    joint_grid_plot("sf", "sigma", df, name, n_iteration, False)


def joint_grid_plot(var1, var2, df, name, n_iteration, marginal=True):
    if marginal:
        g = sns.JointGrid(var1, var2, df, space=0)
        g.plot_marginals(sns.kdeplot, shade=True)
        name = "_marginal_" + name
        g.plot_joint(sns.kdeplot, shade=True, cmap="PuBu", n_levels=40);
        ax = g.ax_joint
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        sns.kdeplot(df[[var1, var2]].values, shade=True, cmap="PuBu", n_levels=40);
        sns.plt.xlabel("")
        sns.plt.ylabel("")
    sns.set(font_scale=2)
    plt.savefig('/home/ulli/Dropbox/gpmemplots/neal_contour_' + var1 + '_vs_' + var2 + '_' + no + '_' + str(
        n_iteration) + '_' + name + '.png', dpi=200, bbox_inches='tight')
    plt.clf()



for i in range(n_it):
    print(i)
    plot_hyper(i,dir_name,syn_data_name)
