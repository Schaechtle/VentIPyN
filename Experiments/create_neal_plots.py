
import seaborn as sns
import pylab as pl
#from plotting import load_experiments
import numpy as np
import numpy.linalg as la
import pandas as pd
from matplotlib import patches, pyplot as plt
import scipy.io as scio
from models.covFunctions import *

from venture import shortcuts
import sys
sys.path.append('../SPs/')
import venture.lite.types as t
from venture.lite.function import VentureFunction
import gp_der
import pickle
import argparse

from models.tools import array

from bayesopt_plugin import neal_f_noiseless

def sample_curve_from_gp(plot_data, curve_xs):
    noiseless_cov = squared_exponential(plot_data.sf, plot_data.l)
    def covariance(x1s, x2s=None):
        return noiseless_cov(x1s, x2s) + noise(plot_data.sigma)(x1s, x2s)

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

    datafname='neal_output/plot_datas.pkl'
    with open(datafname) as f:
        plot_datas = pickle.load(f)

    ## Global figure setup
    sns.set(font_scale=2)

    def draw_plot(plot_data):
        figwidth = 10
        figheight = 10
        fig = plt.figure(figsize=(figwidth,figheight), dpi=200)
        #xs = np.sort(np.random.uniform(-3,3,200))
        xs = np.linspace(-3,3,200)
        for i in range(100):
            yp = sample_curve_from_gp(plot_data, xs)
            plt.plot(xs, yp, c="red", alpha=0.02, linewidth=2)
        plt.axis((-2,2,-1,3))

        x2plot = np.linspace(-3,3,1000)
        f2plot = neal_f_noiseless(x2plot)
        plt.plot(x2plot,f2plot,color='blue')
        plt.scatter(plot_data.Xseen, plot_data.Yseen,
                color='black', marker='x', s=50, edgecolor='black', linewidth='1.5')

        fname_prefix = 'neal_output/%s' % (plot_data.name,)
        print "Saving plots %s.{svg,png}" % (fname_prefix,)
        fig.savefig(fname_prefix + '.svg', dpi=fig.dpi)
        fig.savefig(fname_prefix + '.png', dpi=fig.dpi)
        return fig

    for plot_data in plot_datas:
        draw_plot(plot_data)

