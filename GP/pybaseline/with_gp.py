import utils as u
import gphelper
import numpy as np
from numpy.random import random as rand
import matplotlib.pyplot as plt

# Hyperparameters for the distribution of sigma
sigmahyper_k = 1.0
sigmahyper_theta = 1.0

# Hyperparameters for the distribution of l
lhyper_k = 1.0
lhyper_theta = 1.0

def draw_sigma():
    return np.random.gamma(shape=sigmahyper_k, scale=sigmahyper_theta)
@u.attach_to(draw_sigma)
def logassess(s):
    return u.logassess_gamma(s, sigmahyper_k, sigmahyper_theta)

def draw_l():
    return np.random.gamma(shape=lhyper_k, scale=lhyper_theta)
@u.attach_to(draw_l)
def logassess(s):
    return u.logassess_gamma(s, lhyper_k, lhyper_theta)

def draw_gp_params():
    return (draw_sigma(), draw_l())
@u.attach_to(draw_gp_params)
def logassess(params):
    (sigma, l) = params
    return draw_sigma.logassess(sigma) + draw_l.logassess(l)

# Just an example of a function that could be used
def f_true(x):
    f_true.count += 1
    return (0.2 + np.exp(-0.1*abs(x-2))) * np.cos(0.4*x)
f_true.count = 0

def sample_conditional_hparams(gp):
    # MH sampler
    params_current = gp.get_params()
    def do_step():
        params_prop = (draw_sigma(), draw_l())
        prop_score = (draw_gp_params.logassess(params_current)
                - draw_gp_params.logassess(params_prop)
                + gp.params_local_ll(params_prop)
                - gp.params_local_ll(params_current))
        accprob = np.exp(min(0.0, prop_score))
        if rand() < accprob:
            return params_prop
        else:
            return params_current

    NUM_STEPS = 5
    for dummy in range(NUM_STEPS):
        params_current = do_step()
    return params_current

def search_for_argmax(f):
    # Grid search
    MIN_X = -20
    MAX_X = 20
    NUM_BINS = 100
    xs = np.linspace(MIN_X, MAX_X, NUM_BINS)
    fvec = np.vectorize(f)
    i = np.argmax(fvec(xs))
    return xs[i]


gpt = [gphelper.GPSnapshot(draw_gp_params())]

def not_yet_happy():
    TOTAL_STEPS = 30
    not_yet_happy.count += 1
    if not_yet_happy.count % 50 == 0:
        print "not_yet_happy.count = %d" % (not_yet_happy.count,)
    return not_yet_happy.count <= TOTAL_STEPS
not_yet_happy.count = 0

while not_yet_happy():
    curf = lambda x: gpt[-1].sample_at_point(x)
    x_newt = search_for_argmax(curf)
    gpnew = gpt[-1].copy()
    gpnew.observe(x_newt, f_true(x_newt))
    new_params = sample_conditional_hparams(gpnew)
    gpnew.set_params(new_params)
    gpt.append(gpnew)

print "Inferred sigma = %.2f, l = %.2f" % gpt[-1].get_params()
i = np.argmax(gpt[-1].Yseen)
x = gpt[-1].Xseen[i]
y = gpt[-1].Yseen[i]
print "Best (x,y) pair: (%.2f, %.2f)" % (x, y)
print "f_true.count = %d" % (f_true.count,)
print "Xseen = %s" % (gpt[-1].Xseen,)

# Visualize the point samples of some of the GP snapshots
xs = np.linspace(-20, 20, 40)
def scalarify(mean_and_cov):
    mean, cov = mean_and_cov
    assert mean.size == 1
    assert cov.size == 1
    return (mean[0], cov[0,0])
snapshots = (gpt[-25], gpt[-15], gpt[-1])
linestyles = ('g-', 'y-', 'r-')
labels = ('25th-most recent', '15th-most recent', 'most recent')
plt.plot(xs, [f_true(x) for x in xs], 'b-', label='true')
for (s, style, label) in zip(snapshots, linestyles, labels):
    outputs = [scalarify(s.mean_and_cov_at(np.array([x]))) for x in xs]
    means = np.array([mean for (mean, var) in outputs])
    variances = np.array([var for (mean, var) in outputs])
    stdevs = np.sqrt(variances)

    plt.plot(xs, means, style, label=label)
    plt.plot(xs, means + stdevs, style + '-')
    plt.plot(xs, means - stdevs, style + '-')

plt.legend()
plt.show()
