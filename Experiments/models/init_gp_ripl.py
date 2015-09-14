__author__ = 'ulli'

from venture import shortcuts
import sys
sys.path.append('../SPs/')
from covFunctions_labeled import makeConstFunc,constantType
import venture.lite.types as t
from venture.lite.function import VentureFunction
import gp_with_der

def init_gp_ripl():
    ripl = shortcuts.make_lite_church_prime_ripl()
    ripl.bind_foreign_sp("make_gp_part_der",gp_with_der.makeGPSP)
    ripl.assume('make_const_func', VentureFunction(makeConstFunc, [t.NumberType()], constantType))
    ripl.assume('zero', "(apply_function make_const_func 0)")
    return ripl

