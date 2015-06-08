__author__ = 'ulli'

from venture_gp_model import Venture_GP_model
from venture.lite.function import VentureFunction
import venture.lite.types as t
from covFunctions import makeLinear

class LIN_Venture_GP_Model(Venture_GP_model):
    def __init__(self):
        Venture_GP_model.__init__(self)
    def make_gp(self, ripl):
        ripl.assume('make_lin',VentureFunction(makeLinear,[t.NumberType()], t.AnyType("VentureFunction")))
        ripl.assume('gp',"""(tag (quote parameter) 0
                            (make_gp_part_der zero
                            (apply_function make_lin (log (uniform_continuous 0 8 )) )))""")
