
import seaborn as sns
import pylab as pl
#from plotting import load_experiments
import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio
from models.covFunctions import *

from venture import shortcuts
import sys
sys.path.append('../SPs/')
import venture.lite.types as t
from venture.lite.function import VentureFunction
import gp_der
import pickle
from without_gpmem import PlotData, f_true

from models.tools import array

def scalarify(mean_and_cov):
    mean, cov = mean_and_cov
    assert mean.size == 1
    assert cov.size == 1
    return (mean[0], cov[0,0])

def sample_curve_from_gp(plot_data, curve_xs):
    covariance = squared_exponential(plot_data.sf1, plot_data.l1)
    def getNormal(xs, Xseen, Yseen):
        def cov_matrix(x1s, x2s=None):
            if x2s is None:
                return covariance(np.asmatrix(x1s).T)
            return covariance(np.asmatrix(x1s).T, np.asmatrix(x2s).T)
    
        if len(Xseen) == 0:
            mu = np.zeros(xs.shape)
            sigma = cov_matrix(xs, xs)
        else:
            x2s = np.array(Xseen)
            o2s = np.array(Yseen)
            
            mu1 = np.zeros(xs.shape)
            mu1 = mu1.reshape((mu1.size,))
            mu2 = np.zeros(x2s.shape)
            a2 = np.matrix(o2s.reshape((len(o2s),1)))
        
            sigma11 = cov_matrix(xs, xs)
            sigma12 = cov_matrix(xs, x2s)
            sigma21 = cov_matrix(x2s, xs)
            sigma22 = cov_matrix(x2s,x2s)
            inv22 = la.pinv(sigma22)
    
            plusterm = np.asarray(np.dot(sigma12, np.dot(inv22, (a2 - mu2.reshape(a2.shape))))).squeeze()
            # print plusterm.shape
            mu = mu1 + plusterm
            sigma = sigma11 - np.dot(sigma12,np.dot(inv22,sigma21))
    
        return mu, sigma

    mu, sigma = getNormal(curve_xs, plot_data.Xseen, plot_data.Yseen)
    ys = np.random.multivariate_normal(mu, sigma)
    return ys

if __name__ == '__main__':

    ## Load in the data from log
    logfname = 'log_without_gpmem/plot_datas.pkl'
    with open(logfname) as f:
        plot_datas = pickle.load(f)
    # TODO choose the set of plot datas you want to use
    plot_datas = plot_datas[::3]
    num_plots = len(plot_datas)

    ## Set up the main figure
    sns.set(font_scale=3)
    plotwidth = 30
    plotheight = 10

    fig, axs = plt.subplots(num_plots, 1)
    fig.set_dpi(200)
    fig.set_figheight(num_plots * plotheight)
    fig.set_figwidth(plotwidth)

    ## Make one subplot for each plot_data

    xpost = np.linspace(-20, 20, 100)

    def draw_plot(plot_data, ax):
        for i in range(500):
            #ripl = make_ripl_with_g(plot_data)
            #sampleString = '(g (array %s))' % (' '.join(str(x) for x in xpost),)
            #ypost = ripl.sample(sampleString)
            #yp = [y for (x,y) in sorted(zip(xpost,ypost))]
            yp = sample_curve_from_gp(plot_data, xpost)
            ax.plot(xpost, yp, c="red", alpha=0.1, linewidth=2)

        ax.plot(xpost, [f_true(x) for x in xpost], 'b-', label='true')  
        ax.scatter(plot_data.Xseen, plot_data.Yseen,
                color='black', marker='x', s=400, edgecolor='black', linewidth='3')

        ax.set_xlim(-20, 20)
        ax.set_ylim(-1.5, 1.5)
        ax.legend()

    for (plot_data, ax) in zip(plot_datas, axs):
        print "Plotting new panel:"
        print "sf1=%f, l1=%f" % (plot_data.sf1, plot_data.l1)
        print "Xseen = ", plot_data.Xseen
        print "Yseen = ", plot_data.Yseen
        draw_plot(plot_data, ax)

    fig.savefig('BayesOpt_sequence.svg', dpi=fig.dpi,bbox_inches='tight')
    fig.savefig('BayesOpt_sequence.png', dpi=fig.dpi,bbox_inches='tight')


## The below is for generating the vertical bars
#
#def plott(points, ax, *args):
#    return ax.plot(points[:,0], points[:,1], *args)
#
#def draw_interval(ax, length, center, bdepth, blength):
#    top_endpt = center - np.array([0, 0.5*length])
#    bottom_endpt = center + np.array([0, 0.5*length])
#    tbr = np.array([
#        [0.5*blength, bdepth],
#        [0.5*blength, 0],
#        [-0.5*blength, 0],
#        [-0.5*blength, bdepth],
#        ])
#    bbr = -tbr
#
#    line = np.array([top_endpt, bottom_endpt])
#
#    plott(top_endpt + tbr, ax, 'g-')
#    plott(bottom_endpt + bbr, ax, 'g-')
#    plott(line, ax, 'g-')
#
#    for i in range(len(all_pairs)):
#        (x,y) = all_pairs[i]
#        pairs_omit_one = [all_pairs[j] for j in range(len(all_pairs)) if j != i]
#    
#        Xseen_omit_one, Yseen_omit_one = zip(*pairs_omit_one)
#        mu, sigma = getNormal(np.array([x]), Xseen_omit_one, Yseen_omit_one)
#        draw_interval(plt, 2*sigma[0,0], np.array([x, mu[0,0]]), 0.02, 0.5)
    
    
