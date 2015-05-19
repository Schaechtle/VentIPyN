__author__ = 'ulli'

from venture_gp_model import Venture_GP_model
from venture.lite.function import VentureFunction
import venture.lite.types as t
from covFunctions import makeLinear,makePeriodic,makeLiftedAdd,makeLiftedMult

class LIN_Venture_GP_Model(Venture_GP_model):
    #def __init__(self):
    #super(SE_GP_Model, self).__init__()
    def make_gp(self, ripl):
        ripl.assume('make_lin',VentureFunction(makeLinear,[t.NumberType()], t.AnyType("VentureFunction")))
        ripl.assume('gp',"""(tag (quote parameter) 0
                            (make_gp_part_der zero
                            (apply_function make_lin (uniform_continuous 0 8 ) )))""")
        ripl.assume("func_plus", makeLiftedAdd(lambda x1, x2: x1+x2))

        ripl.assume("func_times", makeLiftedMult(lambda x1, x2: np.multiply(x1,x2)))
