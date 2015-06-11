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

from models.tools import array

argmaxSP = deterministic_typed(np.argmax, [t.HomogeneousArrayType(t.NumberType())], t.NumberType())
absSP = deterministic_typed(abs, [t.NumberType()], t.NumberType())

ripl = shortcuts.make_lite_church_prime_ripl()
ripl.bind_foreign_sp('make_gp_part_der', gp_der.makeGPSP)
ripl.bind_foreign_sp('gpmem', gpmem.gpmemSP)
ripl.bind_foreign_sp('argmax_of_array', argmaxSP)
ripl.bind_foreign_sp('abs', absSP)

ripl.assume('sf1','(tag (quote hyper) 0 (log (uniform_continuous 0 10)))')
ripl.assume('l1','(tag (quote hyper) 1 (log (uniform_continuous 0 10)))')
ripl.assume('make_se',VentureFunction(makeSquaredExponential,[t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
ripl.assume('se', '(apply_function make_se sf1 l1)')
ripl.assume('f', '''
    (lambda (x)
      (*
        (+ 0.2
           (exp (*
                  -0.1
                  (abs (- x 2)))))
        (cos (* 0.4 x))))
    ''')
ripl.assume('package', '(gpmem f se)')
ripl.assume('f_compute', '(first package)')
ripl.assume('f_emu', '(second package)')

# TODO easy to use other search strategies such as gaussian drift
ripl.assume('get_uniform_candidate', '''
    (lambda (prev_xs) (uniform_continuous -20 20))
    ''')
ripl.assume('mc_argmax', '''
    (lambda (emulator prev_xs)
      ((lambda (candidate_xs)
         (lookup  candidate_xs
                  (argmax_of_array (mapv (lambda (i) (get_uniform_candidate prev_xs))
                        (linspace 0 0 15)))))
       (mapv (lambda (i) (get_uniform_candidate prev_xs))
            (linspace 0 0 15))))
    ''')

for i in range(15):
    xs = [ripl.sample('(uniform_continuous -20 20)') for dummy in range(15)]
    ys = [ripl.sample('(lookup (f_emu (array %f)) 0)' % x) for x in xs]
    ripl.predict('(f_compute %f)' % xs[np.argmax(ys)])
    # Once the GP copying stuff works, we will be able to replace the above with:
    # ripl.predict('(f_compute (mc_argmax (lambda (x) (f_emu (array x))) (quote LOLNOTHING)))')
    ripl.infer('(mh (quote hyper) one 50)')

print "Inferred sigma = %.2f, l = %.2f" % (ripl.sample('sf1'), ripl.sample('l1'))
stats = ripl.infer('(extract_stats f_emu)')
(xbest, ybest) = stats[0]
all_pairs = stats[1]
(Xseen, Yseen) = zip(*all_pairs)
print "Best (x,y) pair: (%.2f, %.2f)" % (xbest, ybest)

sns.set(font_scale=3)
figlength = 30
figheigth = 10
fig = plt.figure(figsize=(figlength,figheigth), dpi=200)

xpost = np.linspace(-20, 20, 100)
for i in range(100):
    sample_expr = '(f_emu (array %s))' % (' '.join(str(x) for x in xpost),)
    ypost = ripl.sample(sample_expr)
    plt.plot(xpost, ypost, c='red', alpha=0.1, linewidth=2)

plt.plot(xpost, [ripl.sample('(f %f)' % (x,)) for x in xpost], 'b-', label='true')  
plt.xlim(-20,20)
plt.ylim(-2,2)
plt.scatter(Xseen,Yseen,color='black',marker='x',s=400,edgecolor='black',linewidth='3')

plt.legend()
plt.show()

fig.savefig('BayesOpt_with_gpmem.svg', dpi=fig.dpi,bbox_inches='tight')
fig.savefig('BayesOpt_with_gpmem.png', dpi=fig.dpi,bbox_inches='tight')

# Now trying to figure out what's up...
stats = ripl.infer('(extract_stats f_emu)')
(xbest, ybest) = stats[0]
all_pairs = stats[1]
(Xseen, Yseen) = zip(*all_pairs)
print "After plotting, len(all_pairs) = %s and all_pairs = %s" % (len(all_pairs), all_pairs)
sf1 = ripl.sample('sf1')
l1 = ripl.sample('l1')
logdata = ((sf1, l1), all_pairs)
log_fname = 'log_with_gpmem/log.pkl'
print "Logging to %s" % (log_fname)
with open(log_fname, 'wb') as f:
    pickle.dump(logdata, f)

