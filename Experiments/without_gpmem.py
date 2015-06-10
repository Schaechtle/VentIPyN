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

from models.tools import array
figwidth = 40
figheight = 10

no = "a"


def array(xs):
    return t.VentureArrayUnboxed(np.array(xs),  t.NumberType())

def makeObservations(x,y):
    xString = genSamples(x)
    ripl.observe(xString, array(y))

def genSamples(x):
    sampleString='(gp (array '
    for i in range(len(x)):
        sampleString+= str(x[i]) + ' '
    sampleString+='))'
    #print(sampleString)
    return sampleString
ripl = shortcuts.make_lite_church_prime_ripl()
ripl.bind_foreign_sp("make_gp_part_der",gp_der.makeGPSP)
ripl.assume('make_const_func', VentureFunction(makeConstFunc, [t.NumberType()], constantType))
ripl.assume('zero', "(apply_function make_const_func 0)")
ripl.assume('make_se',VentureFunction(makeSquaredExponential,[t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))

ripl.assume('sf1','(tag (quote hyper) 0 (log (uniform_continuous 0 10)))')
ripl.assume('l1','(tag (quote hyper) 1 (log (uniform_continuous 0 10)))')

ripl.assume('se', "(apply_function make_se sf1 l1 )")

ripl.assume('gp',"""(tag (quote model) 0
                        (make_gp_part_der zero
                          se
                             ))""")

def f_true(x):
    f_true.count += 1
    return (0.2 + np.exp(-0.1*abs(x-2))) * np.cos(0.4*x)
f_true.count = 0

global Xseen
global Yseen
Xseen =[]
Yseen =[]

def not_yet_happy():
    TOTAL_STEPS = 15
    not_yet_happy.count += 1
    if not_yet_happy.count % 50 == 0:
        print "not_yet_happy.count = %d" % (not_yet_happy.count,)
    return not_yet_happy.count <= TOTAL_STEPS


def search_for_argmax(ripl):
    # Grid search
    MIN_X = -20
    MAX_X = 20
    NUM_BINS = 100
    xs = np.random.uniform(MIN_X, MAX_X, NUM_BINS)
    xs_string  =genSamples(xs)
    f_xs = ripl.sample(xs_string)
    i = np.argmax(f_xs)
    Xseen.append(xs[i])
    return xs[i]


not_yet_happy.count = 0
while not_yet_happy():
    x_newt = search_for_argmax(ripl)
    input_string = genSamples([x_newt])
    evaluated = f_true(x_newt)
    Yseen.append(evaluated)
    ripl.observe(input_string, array([evaluated]))
    ripl.infer('(mh (quote hyper) one 2)')


#print "Inferred sigma = %.2f, l = %.2f" % gpt[-1].get_params()
i = np.argmax(Yseen)
x = Xseen[i]
y = Yseen[i]
print "Best (x,y) pair: (%.2f, %.2f)" % (x, y)
print "f_true.count = %d" % (f_true.count,)

sns.set(font_scale=3)
figwidth = 30
figheight = 10
# Visualize the point samples of some of the GP snapshots
xpost = np.linspace(-20, 20, 100)
def scalarify(mean_and_cov):
    mean, cov = mean_and_cov
    assert mean.size == 1
    assert cov.size == 1
    return (mean[0], cov[0,0])
fig, ax = plt.subplots(1)
fig.set_dpi(200)
fig.set_figheight(figheight)
fig.set_figwidth(figwidth)

for i in range(500):
    sampleString=genSamples(xpost)
    ypost = ripl.sample(sampleString)
    yp = [y_temp for (x,y_temp) in sorted(zip(xpost,ypost))]
    ax.plot(sorted(xpost),yp,c="red",alpha=0.1,linewidth=2)

ax.plot(xpost, [f_true(x) for x in xpost], 'b-', label='true')  
ax.scatter(Xseen,Yseen,color='black',marker='x',s=400,edgecolor='black',linewidth='3')

ax.legend()

#sf1 = ripl.sample('sf1')
#l1 = ripl.sample('l1')
#all_pairs = zip(Xseen, Yseen)
#logdata = ((sf1, l1), all_pairs)
#log_fname = 'log_without_gpmem/log.pkl'
#print "Logging to %s" % (log_fname)
#with open(log_fname, 'wb') as f:
#    pickle.dump(logdata, f)




#covariance = squared_exponential(sf1, l1)
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
        mu2 = np.zeros(x2s.shape)
        a2 = np.matrix(o2s.reshape((len(o2s),1)))
    
        sigma11 = cov_matrix(xs, xs)
        sigma12 = cov_matrix(xs, x2s)
        sigma21 = cov_matrix(x2s, xs)
        sigma22 = cov_matrix(x2s,x2s)
        inv22 = la.pinv(sigma22)

        mu = mu1 +np.dot(sigma12,(np.dot(inv22, (a2 - mu2))))
        sigma = sigma11 - np.dot(sigma12,np.dot(inv22,sigma21))

    return mu, sigma

def plott(points, ax, *args):
    return ax.plot(points[:,0], points[:,1], *args)

def draw_interval(ax, length, center, bdepth, blength):
    top_endpt = center - np.array([0, 0.5*length])
    bottom_endpt = center + np.array([0, 0.5*length])
    tbr = np.array([
        [0.5*blength, bdepth],
        [0.5*blength, 0],
        [-0.5*blength, 0],
        [-0.5*blength, bdepth],
        ])
    bbr = -tbr

    line = np.array([top_endpt, bottom_endpt])

    plott(top_endpt + tbr, ax, 'g-')
    plott(bottom_endpt + bbr, ax, 'g-')
    plott(line, ax, 'g-')

#for i in range(len(all_pairs)):
#    (x,y) = all_pairs[i]
#    pairs_omit_one = [all_pairs[j] for j in range(len(all_pairs)) if j != i]
#
#    Xseen_omit_one, Yseen_omit_one = zip(*pairs_omit_one)
#    mu, sigma = getNormal(np.array([x]), Xseen_omit_one, Yseen_omit_one)
#    draw_interval(plt, 2*sigma[0,0], np.array([x, mu[0,0]]), 0.02, 0.5)

plt.xlim(-20, 20)
plt.ylim(-1.5, 1.5)

fig.savefig('BayesOpt'+no+'.svg', dpi=fig.dpi,bbox_inches='tight')
fig.savefig('BayesOpt'+no+'.png', dpi=fig.dpi,bbox_inches='tight')

plt.show()
