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
