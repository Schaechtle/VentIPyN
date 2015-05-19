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




