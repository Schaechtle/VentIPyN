__author__ = 'ulli'

from venture_gp_model import Venture_GP_model
from venture.lite.function import VentureFunction
import venture.lite.types as t
from covFunctions import makeLinear,makePeriodic,makeLiftedAdd

class LIN_P_PER_Venture_GP_Model(Venture_GP_model):
    #def __init__(self):
    #super(SE_GP_Model, self).__init__()
    def make_gp(self, ripl):
        ripl.assume('make_lin',VentureFunction(makeLinear,[t.NumberType()], t.AnyType("VentureFunction")))
        ripl.assume('make_per',VentureFunction(makePeriodic,[t.NumberType(), t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
        ripl.assume('a',"(tag (quote parameter) 0 (log  (uniform_continuous 0 8 ) ))")
        ripl.assume('sf',"(tag (quote parameter) 1 (log (uniform_continuous 0 8 ) ))")
        ripl.assume('p',"(tag (quote parameter) 2 (log (uniform_continuous 0.1 8) ))")
        ripl.assume('l',"(tag (quote parameter) 3 (log (uniform_continuous 0 8) ))")

        ripl.assume("func_plus", makeLiftedAdd(lambda x1, x2: x1+x2))

        ripl.assume('gp',"""(tag (quote model) 0
                            (make_gp_part_der zero
                                (apply_function func_plus
                                (apply_function make_per l p sf  )
                                (apply_function make_lin a ))))""")