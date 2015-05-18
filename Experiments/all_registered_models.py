from test_model import test_model
from models.se_gp_model import SE_GP_Model
from models.per_gp_model import PER_GP_Model
from models.lin_gp_model import LIN_GP_Model
#registered_models={'test_model':test_model}
registered_models={}
se_gp=SE_GP_Model()
registered_models['venture-gp-se']=se_gp
per_gp = PER_GP_Model()
registered_models['venture-gp-per']=per_gp
lin_gp = LIN_GP_Model()
registered_models['venture-gp-lin']=lin_gp
