__author__ = 'ulli'

from venture_gp_model import Venture_GP_model
from venture.lite.function import VentureFunction
from venture.lite.types import VentureSimplex
import venture.lite.types as t
from covFunctions import makePeriodic,constantType,makeConst,makeLinear,makeSquaredExponential,covfunctionType,makeNoise
from venture.lite.builtin import typed_nr
import itertools
import sys
sys.path.append('../SPs/')
from grammar5 import Grammar
from kernel_interpreter import GrammarInterpreter

class Grammar_Venture_GP_model_unif_airline(Venture_GP_model):
    def __init__(self):
        Venture_GP_model.__init__(self)
        self.record_interpretation = True
    def make_gp(self, ripl):
        ripl.assume('make_linear', VentureFunction(makeLinear, [t.NumberType()], t.AnyType("VentureFunction")))
        ripl.assume('make_periodic', VentureFunction(makePeriodic, [t.NumberType(), t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
        ripl.assume('make_se',VentureFunction(makeSquaredExponential,[t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
        ripl.assume('make_noise', VentureFunction(makeNoise, [t.NumberType()], t.AnyType("VentureFunction")))

        ripl.assume('a','(mem (lambda (i) (tag (quote parameter) i (log  (uniform_continuous 1 8)))))')
        ripl.assume('sn','(tag (quote parameter) 10 (log (uniform_continuous 1 3)))')
        #ripl.assume('sn','0.7')
        ripl.assume('sf','(mem (lambda (i) (tag (quote parameter) i (log (uniform_continuous 0.1 8 )))))')
        ripl.assume('sf2','(mem (lambda (i) (tag (quote parameter) i (log (uniform_continuous 0.1 8 )))))')
        ripl.assume('p','(mem (lambda (i) (tag (quote parameter) i (log (uniform_continuous 0.1 8)))))')
        ripl.assume('l','(mem (lambda (i) (tag (quote parameter) i (log (uniform_continuous 0.1 8)))))')
        ripl.assume('l2','(mem (lambda (i) (tag (quote parameter) i (log (uniform_continuous 0.1 8)))))')
        
        ripl.assume('lin1', "(apply_function make_linear ( a 0)  )")
        ripl.assume('lin2', "(apply_function make_linear ( a 1)  )")
        ripl.assume('lin3', "(apply_function make_linear ( a 2)  )")
        ripl.assume('per1', "(apply_function make_periodic ( l 3) ( p 4) ( sf 5) )")
        ripl.assume('se1', "(apply_function make_se ( sf2 6) ( l2 7))")
        ripl.assume('se2', "(apply_function make_se ( sf2 8) ( l2 9))")


        ripl.assume('wn', "(apply_function make_noise sn)")

         #### GP Structure Prior

        ###### for simplicity, I start with the max amount of kernel per type given

        ripl.assume("max_lin","(array  lin1 lin2 lin3 )")
        ripl.assume("max_per","(array  per1 )")
        ripl.assume("max_se","(array  se1 se2)")
        ripl.assume("max_wn","(array wn)")


        number = 7
        simplex = "( simplex  "
        for i in range(number):
            simplex+=str(1./number) + " "

        simplex+=" )"

        ripl.bind_foreign_sp("gp_grammar", typed_nr(Grammar(), [t.HomogeneousArrayType(t.HomogeneousArrayType(t.AnyType())),t.AnyType()], covfunctionType, min_req_args=0))

        ripl.assume("cov_structure","(tag (quote grammar) 0 (gp_grammar (array max_lin max_wn max_per max_se) "+simplex+" ))")
        #ripl.bind_foreign_sp("covfunc_interpreter",typed_nr(GrammarInterpreter(), [t.AnyType()], t.AnyType()))
        #ripl.assume("interp","(covfunc_interpreter grammar)")

        ripl.assume('gp','(tag (quote model) 0 (make_gp_part_der zero cov_structure))')

        ripl.bind_foreign_sp("covfunc_interpreter",typed_nr(GrammarInterpreter(), [t.AnyType()], t.AnyType()))
        ripl.assume("interp","(covfunc_interpreter cov_structure)")

    def collect_parameters(self,ripl):
        return [ripl.sample("(l2 7)"),ripl.sample("(l2 9)"),ripl.sample("(sf2 6)"),ripl.sample("(sf2 8)")]



