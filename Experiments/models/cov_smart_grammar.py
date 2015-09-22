__author__ = 'ulli'

from venture_gp_model import Venture_GP_model
from venture.lite.function import VentureFunction
from venture.lite.types import VentureSimplex
import venture.lite.types as t
from venture.lite.value import VentureSymbol,VentureArray
from covFunctions_labeled import makePeriodic,constantType,makeConst,makeLinear,makeSquaredExponential,covfunctionType,makeNoise,makeRQ,makeLiftedAdd,makeLiftedMult
from venture.lite.sp_help import typed_nr,deterministic_typed
import itertools
import sys
sys.path.append('../SPs/')
from subset import Subset
import numpy as np

class Grammar_Venture_GP_model_Smart(Venture_GP_model):
    def __init__(self):
        Venture_GP_model.__init__(self)
        self.record_interpretation = True
    def make_gp(self, ripl):
        ripl.assume('make_linear', VentureFunction(makeLinear, [t.NumberType(),t.IntegerType()], t.AnyType("VentureFunction")))
        ripl.assume('make_periodic', VentureFunction(makePeriodic, [t.NumberType(), t.NumberType(), t.NumberType(),t.IntegerType()], t.AnyType("VentureFunction")))
        ripl.assume('make_se',VentureFunction(makeSquaredExponential,[t.NumberType(), t.NumberType(),t.IntegerType()], t.AnyType("VentureFunction")))
        ripl.assume('make_noise', VentureFunction(makeNoise, [t.NumberType(),t.IntegerType()], t.AnyType("VentureFunction")))
        #ripl.assume('make_rq', VentureFunction(makeRQ, [t.NumberType(), t.NumberType(), t.NumberType(),t.IntegerType()], t.AnyType("VentureFunction")))
        #ripl.assume('make_const_cov', VentureFunction(makeConst, [t.NumberType(),t.IntegerType()], t.AnyType("VentureFunction")))


        ripl.assume('a',' (tag (quote 0 ) 0 (uniform_continuous  0.01 10))')

        ripl.assume('l',' (tag (quote 1) 0 (uniform_continuous  0.01 10))')
        ripl.assume('q',' (tag (quote 1 ) 1 (uniform_continuous  0.01 10))')
        ripl.assume('s','(tag (quote 1 ) 2 (uniform_continuous  0.01 10))')

        ripl.assume('ell',' (tag (quote 2) 0 (uniform_continuous  0.01 10))')
        ripl.assume('sf',' (tag (quote 2 ) 1 (uniform_continuous  0.01 10))')

        ripl.assume('n','(tag (quote 3 ) 0 (uniform_continuous  0.01 10))')


        ripl.assume('lin', "(apply_function make_linear a  0 )")
        ripl.assume('per', "(apply_function make_periodic l q s 1 ) ")
        ripl.assume('se', "(apply_function make_se ell sf 2 )")
        #ripl.assume('rq', "(apply_function make_rq (hyper_parameter 3 0) (hyper_parameter 3 1)  (hyper_parameter 3 2) 3)")
        ripl.assume('wn', "(apply_function make_noise (n  3 )")
        #ripl.assume('c',"(apply_function make_const_cov (hyper_parameter 4 0)  4 )")
        #ripl.assume('se2', "(apply_function make_se(hyper_parameter 3 0) (hyper_parameter 3 1) 3 )")
        #ripl.assume('rq', "(apply_function make_rq (hyper_parameter 4 0) (hyper_parameter 4 1)  (hyper_parameter 4 2) 4)")


         #### GP Structure Prior

        ###### for simplicity, I start with the max amount of kernel per type given

        ripl.assume("func_times", makeLiftedMult(lambda x1, x2: np.multiply(x1,x2)))
        ripl.assume("func_plus", makeLiftedAdd(lambda x1, x2: x1 + x2))


        ripl.assume('cov_list','(list lin per se wn )')
        ripl.bind_foreign_sp("subset",typed_nr(Subset(), [t.ListType(),t.SimplexType()], t.ListType()))

        number = 4

        total_perms =0
        perms = []
        for i in range(number):
            perms.append((len(list(itertools.permutations([j for j in range(i+1)])))))
            total_perms+=perms[i]


        simplex = "( simplex  "
        for i in range(number):
            simplex+=str(float(perms[i])/total_perms) + " "

        simplex+=" )"
        ripl.assume('s','(tag (quote grammar) 1 (subset cov_list '+simplex + ' ))')
        ripl.assume('cov_compo',"""
         (tag (quote grammar) 0
             (lambda (l )
                (if (lte ( size l) 1)
                     (first l)
                         (if (flip)
                             (apply_function func_plus (first l) (cov_compo (rest l)))
                             (apply_function func_times (first l) (cov_compo (rest l)))
                    )
        )))
        """)


        ripl.assume('cov_structure','(cov_compo s)')
        ripl.assume('gp','(tag (quote model) 0 (make_gp_part_der zero cov_structure))')

        ripl.bind_foreign_sp("covariance_string",
                  deterministic_typed(lambda x:VentureSymbol(x.stuff['name']), [t.AnyType()], t.AnyType(),
                                      descr="returns the covariance type"))

        ripl.bind_foreign_sp("covariance_label",
                  deterministic_typed(lambda x:x.stuff['label_list'], [t.AnyType()], t.ArrayType(),
                                      descr="returns the covariance label"))



    def collect_parameters(self,ripl):
        return []

