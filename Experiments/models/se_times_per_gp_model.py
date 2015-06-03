__author__ = 'ulli'

from venture_gp_model import Venture_GP_model
from venture.lite.function import VentureFunction
import venture.lite.types as t
from covFunctions import makeSquaredExponential,makePeriodic,makeLiftedMult
import numpy as np
class SE_T_PER_Venture_GP_Model(Venture_GP_model):
    #def __init__(self):
    #super(SE_GP_Model, self).__init__()
    def make_gp(self, ripl):
        ripl.assume('make_periodic', VentureFunction(makePeriodic, [t.NumberType(), t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
        ripl.assume('make_se',VentureFunction(makeSquaredExponential,[t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))




        ripl.assume('sf1','(tag (quote parameter) 0 (log (uniform_continuous 0 10)))')
        ripl.assume('l1','(tag (quote parameter) 1 (log (uniform_continuous 0 10)))')



        ripl.assume('ell','(tag (quote parameter) 2 (log (uniform_continuous 0 10)))')
        ripl.assume('p','(tag (quote parameter) 3 (log (uniform_continuous 0 10)))')
        ripl.assume('s','(tag (quote parameter) 4 (log (uniform_continuous 0 10)))')
        ripl.assume("func_times", makeLiftedMult(lambda x1, x2: np.multiply(x1,x2)))

        ripl.assume('se', "(apply_function make_se sf1 l1 )")

        ripl.assume('per', "(apply_function make_periodic ell p s)")

        ripl.assume('gp',"""(tag (quote model) 0
                        (make_gp_part_der zero
                            (apply_function func_times se per
                                )

                             ))""")