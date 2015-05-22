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


from models.lin_times_per_gp_model import LIN_T_PER_Venture_GP_Model
lin_t_per_gp = LIN_T_PER_Venture_GP_Model()
registered_models['venture-gp-lin-t-per']=lin_t_per_gp

from models.cov_by_grammar_gp import Grammar_Venture_GP_model
venture_cov_learning = Grammar_Venture_GP_model()
registered_models['venture-cov-learning']=venture_cov_learning

from models.cov_by_grammar_gp_testing import Grammar_Venture_GP_model_test
venture_cov_learning = Grammar_Venture_GP_model_test()
registered_models['venture-cov-learning-testing']=venture_cov_learning

from models.cov_by_grammar_gp_simple import Grammar_Venture_GP_model_simple
venture_cov_simple = Grammar_Venture_GP_model_simple()
registered_models['venture_cov_simple']=venture_cov_simple