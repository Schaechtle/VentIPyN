__author__ = 'ulli'

from gp_model import GP_model
from venture.lite.function import VentureFunction
import venture.lite.types as t
from covFunctions import makeSquaredExponential

class SE_GP_Model(GP_model):
    #def __init__(self):
    #super(SE_GP_Model, self).__init__()
    def make_gp(self, ripl):
        ripl.assume('make_se',VentureFunction(makeSquaredExponential,[t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
        ripl.assume('gp',"""(tag (quote parameter) 0
                            (make_gp_part_der zero
                            (apply_function make_se (uniform_continuous 0 8 ) (uniform_continuous 0 8 ) )))""")
