__author__ = 'ulli'

from venture_gp_model import Venture_GP_model
from venture.lite.function import VentureFunction
from venture.lite.types import VentureSimplex
import venture.lite.types as t
from covFunctions import makePeriodic,constantType,makeConst,makeLinear,makeSquaredExponential,covfunctionType,makeNoise,makeRQ
from venture.lite.builtin import typed_nr
import itertools
import sys
sys.path.append('../SPs/')
from grammar5 import Grammar
from kernel_interpreter import GrammarInterpreter

class Grammar_Venture_GP_model_unif_mauna(Venture_GP_model):
    def __init__(self):
        Venture_GP_model.__init__(self)
        self.record_interpretation = True
    def make_gp(self, ripl):
        ripl.assume('make_linear', VentureFunction(makeLinear, [t.NumberType()], t.AnyType("VentureFunction")))
        ripl.assume('make_periodic', VentureFunction(makePeriodic, [t.NumberType(), t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
        ripl.assume('make_se',VentureFunction(makeSquaredExponential,[t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
        ripl.assume('make_rq',VentureFunction(makeRQ, [t.NumberType(), t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))

        ripl.assume('a','(tag (quote parameter) 0 (log  (uniform_continuous  0 5)))')
        ripl.assume('sf1','(tag (quote parameter) 1 (log (uniform_continuous  0 5 )))')
        ripl.assume('sf2',' (tag (quote parameter) 2 (log (uniform_continuous  0 5 )))')
        ripl.assume('p',' (tag (quote parameter) 3 (log (uniform_continuous  0.01 5)))')
        ripl.assume('l',' (tag (quote parameter) 4 (log (uniform_continuous  0 5)))')

        ripl.assume('l1',' (tag (quote parameter) 5 (log (uniform_continuous  0 5)))')
        ripl.assume('l2',' (tag (quote parameter) 6 (log (uniform_continuous  0 5)))')
        ripl.assume('sf_rq','(tag (quote hypers) 7 (log (uniform_continuous 0 5)))')
        ripl.assume('l_rq','(tag (quote hypers) 8 (log (uniform_continuous 0 5)))')
        ripl.assume('alpha','(tag (quote hypers)9 (log (uniform_continuous 0 5)))')
        ripl.assume('sf',' (tag (quote parameter) 10 (log (uniform_continuous  0 5)))')

        ripl.assume('lin1', "(apply_function make_linear a   )")
        ripl.assume('per1', "(apply_function make_periodic l  p  sf  )")
        ripl.assume('se1', "(apply_function make_se sf1 l1)")
        ripl.assume('se2', "(apply_function make_se sf2 l2)")
        ripl.assume('rq', "(apply_function make_rq l_rq sf_rq alpha)")

         #### GP Structure Prior

        ###### for simplicity, I start with the max amount of kernel per type given

        ripl.assume("max_lin","(array  lin1 )")
        ripl.assume("max_per","(array  per1 )")
        ripl.assume("max_se","(array  se1 se2)")
        ripl.assume("max_rq","(array  rq)")



        number = 5
        simplex = "( simplex  "
        for i in range(number):
            simplex+=str(1./number) + " "

        simplex+=" )"

        ripl.bind_foreign_sp("gp_grammar", typed_nr(Grammar(), [t.HomogeneousArrayType(t.HomogeneousArrayType(t.AnyType())),t.AnyType()], covfunctionType, min_req_args=0))

        ripl.assume("cov_structure","(tag (quote grammar) 0 (gp_grammar (array max_lin max_rq max_per max_se) "+simplex+" ))")
        #ripl.bind_foreign_sp("covfunc_interpreter",typed_nr(GrammarInterpreter(), [t.AnyType()], t.AnyType()))
        #ripl.assume("interp","(covfunc_interpreter grammar)")

        ripl.assume('gp','(tag (quote model) 0 (make_gp_part_der zero cov_structure))')

        ripl.bind_foreign_sp("covfunc_interpreter",typed_nr(GrammarInterpreter(), [t.AnyType()], t.AnyType()))
        ripl.assume("interp","(covfunc_interpreter cov_structure)")



