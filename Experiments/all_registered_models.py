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
