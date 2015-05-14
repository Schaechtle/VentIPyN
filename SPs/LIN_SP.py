import numpy as np

from venture.lite.psp import NullRequestPSP

class LinMaker(NullRequestPSP):
    def childrenCanAAA(self): return True
    def simulate(self,args):
        sf= np.exp(2.*args.operandValues[0])
        def f(x1,  x2=None):
            if x2 is None: # self-covariances for test cases
                nn,D = x1.shape
                A = np.dot(x1,x1.T)
            else:
                A = np.dot(x1,x2.T) + 1e-10    # required for numerical accuracy
            return sf * A
        return f

    def gradientOfSimulate(self, args, _value, _direction):
        sf= np.exp(2.*args.operandValues[0])
        def f(x1, x2):
            A = np.dot(x1,x2.T)
            return 2 * sf * A
        return f





