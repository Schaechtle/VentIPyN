__author__ = 'ulli'

from venture_gp_model import Venture_GP_model
from venture.lite.function import VentureFunction
from venture.lite.types import VentureSimplex
import venture.lite.types as t
from covFunctions import makePeriodic,constantType,makeConst,makeLinear,makeSquaredExponential,covfunctionType,makeNoise,makeRQ,makeLiftedMult,makeLiftedAdd
from venture.lite.builtin import typed_nr
import itertools
import sys

sys.path.append('../SPs/')
from grammar5 import Grammar
from kernel_interpreter import GrammarInterpreter
from subset import Subset

class Grammar_Venture_GP_model_airline(Venture_GP_model):
    def __init__(self):
        Venture_GP_model.__init__(self)
        self.record_interpretation = True
    def make_gp(self, ripl):
        ripl.assume('make_periodic', VentureFunction(makePeriodic, [t.NumberType(), t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
        ripl.assume('make_se',VentureFunction(makeSquaredExponential,[t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
        ripl.assume('make_rq',VentureFunction(makeRQ, [t.NumberType(), t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
    
        ripl.assume('a1','(tag (quote parameter) 0 (log  (uniform_continuous  0 10)))')
        ripl.assume('sf1','(tag (quote parameter) 1 (log (uniform_continuous  0 10 )))')
        ripl.assume('sf2',' (tag (quote parameter) 2 (log (uniform_continuous  0 10 )))')
        ripl.assume('p',' (tag (quote parameter) 3 (log (uniform_continuous  0.01 10)))')
        ripl.assume('l',' (tag (quote parameter) 4 (log (uniform_continuous  0 10)))')
    
        ripl.assume('l1',' (tag (quote parameter) 5 (log (uniform_continuous  0 10)))')
        ripl.assume('l2',' (tag (quote parameter) 6 (log (uniform_continuous  0 10)))')
        ripl.assume('sf_rq','(tag (quote hypers) 7 (log (uniform_continuous 0 10)))')
        ripl.assume('l_rq','(tag (quote hypers) 8 (log (uniform_continuous 0 10)))')
        ripl.assume('alpha','(tag (quote hypers)9 (log (uniform_continuous 0 10)))')
        ripl.assume('sf',' (tag (quote parameter) 10 (log (uniform_continuous  0 10)))')


        ripl.assume('a2','(tag (quote parameter) 11 (log  (uniform_continuous  0 10)))')

        ripl.assume('lin1', "(apply_function make_linear a1   )")
        ripl.assume('lin2', "(apply_function make_linear a2   )")
        ripl.assume('per1', "(apply_function make_periodic l  p  sf ) ")
        ripl.assume('se1', "(apply_function make_se sf1 l1)")
        ripl.assume('se2', "(apply_function make_se sf2 l2)")
        ripl.assume('rq', "(apply_function make_rq l_rq sf_rq alpha)")
    
         #### GP Structure Prior
    
        ###### for simplicity, I start with the max amount of kernel per type given
    
        ripl.assume("func_times", makeLiftedMult(lambda x1, x2: np.multiply(x1,x2)))
        ripl.assume("func_plus", makeLiftedAdd(lambda x1, x2: x1 + x2))
    
    
        ripl.assume('cov_list','(list lin1 per1 se1 se2 rq)')
        ripl.bind_foreign_sp("subset",typed_nr(Subset(), [t.ListType(),t.SimplexType()], t.ListType()))
    
        number = 10
    
        total_perms =0
        perms = []
        for i in range(number):
            perms.append((len(list(itertools.permutations([j for j in range(i+1)])))))
            total_perms+=perms[i]
    
    
        simplex = "( simplex  "
        for i in range(number):
            simplex+=str(float(perms[i])/total_perms) + " "
    
        simplex+=" )"
        ripl.assume('s',' (tag (quote grammar) 0 (subset cov_list '+simplex + ' ))')
        ripl.assume('cov_compo',"""
             (lambda (l )
                (if (lte ( size l) 1)
                     (first l)
                         (if (flip)
                             (apply_function func_plus (first l) (cov_compo (rest l)))
                             (apply_function func_times (first l) (cov_compo (rest l)))
                    )
        ))
        """)
    
        #ripl.bind_foreign_sp("covfunc_interpreter",typed_nr(GrammarInterpreter(), [t.AnyType()], t.AnyType()))
        #ripl.assume("interp","(covfunc_interpreter grammar)")
        ripl.assume('cov_structure','(tag (quote grammar) 1 (cov_compo s))')
        ripl.assume('gp','(tag (quote model) 0 (make_gp_part_der zero cov_structure))')
    
        ripl.bind_foreign_sp("covfunc_interpreter",typed_nr(GrammarInterpreter(), [t.AnyType()], t.AnyType()))
        ripl.assume("interp","(covfunc_interpreter cov_structure)")




    def collect_parameters(self,ripl):
        return [ripl.sample("(l2 7)"),ripl.sample("(l2 9)"),ripl.sample("(sf2 6)"),ripl.sample("(sf2 10)")]




