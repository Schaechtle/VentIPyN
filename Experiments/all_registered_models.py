# register models to be testesd in the experiment here
registered_models={}

from models.se_gp_model import SE_Venture_GP_Model
se_gp=SE_Venture_GP_Model()
registered_models['venture-gp-se']=se_gp

from models.per_gp_model import PER_Venture_GP_Model
per_gp = PER_Venture_GP_Model()
registered_models['venture-gp-per']=per_gp

from models.lin_gp_model import LIN_Venture_GP_Model
lin_gp = LIN_Venture_GP_Model()
registered_models['venture-gp-lin']=lin_gp


from models.lin_plus_per_gp_model import LIN_P_PER_Venture_GP_Model
lin_p_per_gp = LIN_P_PER_Venture_GP_Model()
registered_models['venture-gp-lin-p-per']=lin_p_per_gp

from models.lin_plus_wn_gp_model import LIN_P_WN_Venture_GP_Model
lin_p_wn_gp = LIN_P_WN_Venture_GP_Model()
registered_models['venture-gp-lin-p-wn']=lin_p_wn_gp


from models.lin_times_per_gp_model import LIN_T_PER_Venture_GP_Model
lin_t_per_gp = LIN_T_PER_Venture_GP_Model()
registered_models['venture-gp-lin-t-per']=lin_t_per_gp

from models.se_times_per_gp_model import SE_T_PER_Venture_GP_Model
se_t_per_gp = SE_T_PER_Venture_GP_Model()
registered_models['venture-gp-se-t-per']=se_t_per_gp

'''
from models.cov_by_grammar_gp import Grammar_Venture_GP_model
venture_cov_learning = Grammar_Venture_GP_model()
registered_models['venture-cov-learning']=venture_cov_learning

from models.cov_by_grammar_gp_unif_number import Grammar_Venture_GP_model_unif
venture_cov_learning_unif = Grammar_Venture_GP_model_unif()
registered_models['venture-cov-learning-unif']=venture_cov_learning_unif


from models.cov_by_grammar_gp_simple import Grammar_Venture_GP_model_simple
venture_cov_simple = Grammar_Venture_GP_model_simple()
registered_models['venture_cov_simple']=venture_cov_simple

from models.cov_by_grammar_gp_simpleSELINPER import Grammar_Venture_GP_model_simpleSELINPER
venture_cov_simpleSELINPER = Grammar_Venture_GP_model_simpleSELINPER()
registered_models['venture_cov_simple_selinper']=venture_cov_simpleSELINPER

from models.cov_by_grammar_gp_simpleLINPERWN import Grammar_Venture_GP_model_simpleLINPERWN
venture_cov_simpleLINPERWN = Grammar_Venture_GP_model_simpleLINPERWN()
registered_models['venture gp cov learning 4']=venture_cov_simpleLINPERWN

from models.cov_by_grammar_gp_airline import Grammar_Venture_GP_model_airline
venture_cov_learning_airline = Grammar_Venture_GP_model_airline()
registered_models['venture-cov-learning-airline']=venture_cov_learning_airline

from models.cov_by_grammar_gp_unif_number_airline import Grammar_Venture_GP_model_unif_airline
venture_cov_learning_unif_airline = Grammar_Venture_GP_model_unif_airline()
registered_models['venture-cov-learning-unif-airline']=venture_cov_learning_unif_airline

from models.cov_by_grammar_gp_unif_number_gamma import Grammar_Venture_GP_model_unif_gamma
venture_cov_learning_unif_gamma = Grammar_Venture_GP_model_unif_gamma()
registered_models['venture-cov-learning-unif-gamma']=venture_cov_learning_unif_gamma

from models.cov_by_grammar_gp_co2 import Grammar_Venture_GP_model_mauna
venture_cov_learning_mauna = Grammar_Venture_GP_model_mauna()
registered_models['venture-cov-learning-co2']=venture_cov_learning_mauna

from models.cov_by_grammar_gp_unif_number_co2 import Grammar_Venture_GP_model_unif_mauna
venture_cov_learning_unif_mauna = Grammar_Venture_GP_model_unif_mauna()
registered_models['venture-cov-learning-unif-co2']=venture_cov_learning_unif_mauna

from models.cov_by_grammar_gp_airold import Grammar_Venture_GP_model_airold
venture_cov_learning_airold = Grammar_Venture_GP_model_airold()
registered_models['venture-cov-learning-airold']=venture_cov_learning_airold
'''

# first dynamically scoped inference
from models.cov_smart_grammar import Grammar_Venture_GP_model_Smart
venture_cov_learning_smart = Grammar_Venture_GP_model_Smart()
registered_models['venture-cov-learning-smart']=venture_cov_learning_smart

# dynamically scoped inference and change points
from models.cov_cp_grammar import Grammar_Venture_GP_model_cp
venture_cov_learning_cp= Grammar_Venture_GP_model_cp()
registered_models['venture-cov-learning-cp']=venture_cov_learning_cp

from models.cov_smart_simple import Grammar_Venture_GP_smart_simple
venture_cov_learning_smart_simple = Grammar_Venture_GP_smart_simple()
registered_models['venture-cov-learning-smart-simple']=venture_cov_learning_smart_simple

from models.cov_smart_rq import Grammar_Venture_GP_smart_rq
venture_cov_learning_smart_rq = Grammar_Venture_GP_smart_rq()
registered_models['venture-cov-learning-smart-rq']=venture_cov_learning_smart_rq