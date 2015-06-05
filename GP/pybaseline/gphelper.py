import numpy as np
import numpy.linalg as la

class GPSnapshot:
    def __init__(self, params):
        self.set_params(params)
        self.Xseen = []
        self.Yseen = []

    def get_params(self):
        return (self.sigma, self.l)

    def set_params(self, params):
        self.sigma, self.l = params

    def prior_cov(self, x1, x2):
        return self.sigma**2 * np.exp(-(x1-x2)**2 / (2 * self.l**2))

    def prior_covmatrix(self, x1s, x2s):
        return np.array([[self.prior_cov(x1, x2) for x2 in x2s] for x1 in x1s])

    def params_local_ll(self, params):
        covmat = self.prior_covmatrix(self.Xseen, self.Xseen)
        invcovmat = la.pinv(covmat)
        y = self.Yseen
        return (-0.5 * np.log(la.det(covmat))
                - 0.5 * np.dot(y, np.dot(invcovmat, y)))

    def copy(self):
        c = GPSnapshot((self.sigma, self.l))
        c.Xseen = list(self.Xseen)
        c.Yseen = list(self.Yseen)
        return c

    def observe(self, x, y):
        if x in self.Xseen:
            raise Exception("Observed two data points with the same x-value")
        self.Xseen.append(x)
        self.Yseen.append(y)
        # TODO should some other data structures exist that would need to be
        # updated?

    def sample_at(self, xs):
        xs = np.array(xs)
        # Stole and adapted this from backend/lite/gp.py
        if len(self.Xseen) == 0:
            mu = np.zeros(xs.shape)
            sigma = self.prior_covmatrix(xs, xs)
        else:
            x2s = np.array(self.Xseen)
            y2s = np.array(self.Yseen)

            mu1 = np.zeros(xs.shape)
            mu2 = np.zeros(x2s.shape)

            sigma11 = self.prior_covmatrix(xs, xs)
            sigma12 = self.prior_covmatrix(xs, x2s)
            # sigma21 = self.prior_covmatrix(x2s, xs)
            sigma21 = sigma12.T
            sigma22 = self.prior_covmatrix(x2s, x2s)
            inv22 = la.pinv(sigma22)

            mu = mu1 + sigma12 * (inv22 * (y2s - mu2))
            sigma = sigma11 - sigma12 * inv22 * sigma21
        
        # XXX I'm currently debugging this
        print "len(xs), mu.shape, sigma.shape = %s, %s, %s" % (len(xs), mu.shape, sigma.shape)
        return np.random.multivariate_normal(mu.flatten(), sigma)

    def sample_at_point(self, x):
        xs = [x]
        ys = self.sample_at(xs)
        assert len(ys) == 1
        return ys[0]
