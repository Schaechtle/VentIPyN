import utils as u
import numpy as np
from numpy.random import random as rand

# Hyperparameters for the distribution of mu
mu_hyp = 0.0
sigma_hyp = 1.0

# Hyperparameters for the distribution of sigma
k_hyp = 1.0
theta_hyp = 1.0

def draw_mu():
    return np.random.normal(loc=mu_hyp, scale=sigma_hyp)
@u.attach_to(draw_mu)
def logassess(m):
    return u.logassess_normal(m, mu_hyp, sigma_hyp)

def draw_sigma():
    return np.random.gamma(shape=k_hyp, scale=theta_hyp)
@u.attach_to(draw_sigma)
def logassess(s):
    return u.logassess_gamma(s, k_hyp, theta_hyp)

# Parametric form of the learned function
def f_parametric(x, mu, sigma):
    return u.logassess_normal(x, mu, sigma)

# Parameters of the true function
mu_true = draw_mu()
sigma_true = draw_sigma()
print "mu_true = %.2f, sigma_true = %.2f" % (mu_true, sigma_true)
def f_true(x):
    f_true.count += 1
    return f_parametric(x, mu_true, sigma_true)
f_true.count = 0

def logcost(x, approx_fx, params):
    COST_STDEV = 1.0
    return u.logassess_normal(f_parametric(x, *params) -
            approx_fx, 0.0, COST_STDEV)

def sample_conditional_hparams(Xseen, Yseen, params_initial):
    # MH sampler
    def logassess_prior(mu, sigma):
        return draw_mu.logassess(mu) + draw_sigma.logassess(sigma)
    def local_ll(mu, sigma):
        return sum(logcost(x,y, (mu,sigma)) for (x,y) in zip(Xseen, Yseen))

    params_current = params_initial
    def do_step():
        params_prop = (draw_mu(), draw_sigma())
        prop_score = (logassess_prior(*params_current)
                - logassess_prior(*params_prop)
                + local_ll(*params_prop)
                - local_ll(*params_current))
        accprob = np.exp(min(0.0, prop_score))
        if rand() < accprob:
            return params_prop
        else:
            return params_current

    NUM_STEPS = 50
    for dummy in range(NUM_STEPS):
        params_current = do_step()
    return params_current

def search_for_argmax(f):
    # Grid search
    MIN_X = -20
    MAX_X = 20
    NUM_BINS = 1000
    xs = np.linspace(MIN_X, MAX_X, NUM_BINS)
    fvec = np.vectorize(f)
    i = np.argmax(fvec(xs))
    return xs[i]


mut = [draw_mu()]
sigmat = [draw_sigma()]
Xseen = []
Yseen = []

def not_yet_happy():
    TOTAL_STEPS = 300
    not_yet_happy.count += 1
    if not_yet_happy.count % 50 == 0:
        print "not_yet_happy.count = %d" % (not_yet_happy.count,)
    return not_yet_happy.count <= TOTAL_STEPS
not_yet_happy.count = 0

while not_yet_happy():
    curf = lambda x: f_parametric(x, mut[-1], sigmat[-1])
    x_newt = search_for_argmax(curf)
    Xseen.append(x_newt)
    Yseen.append(f_true(x_newt))
    (mu_new, sigma_new) = sample_conditional_hparams(Xseen, Yseen, (mut[-1], sigmat[-1]))
    mut.append(mu_new)
    sigmat.append(sigma_new)

print "Inferred mu = %.2f, sigma = %.2f" % (mut[-1], sigmat[-1])
print "f_true.count = %d" % (f_true.count,)
# print "(x,y) pairs seen:", np.vstack([Xseen, Yseen])
