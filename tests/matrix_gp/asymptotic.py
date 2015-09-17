
# You should have received a copy of the GNU General Public License
# along with Venture.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from scipy import stats
import venture.lite.types as t
from venture import shortcuts
from venture.lite.function import VentureFunction
import imp
import collections
gp_w_der = imp.load_source('gp_with_der', '/home/ulli/VentIPyN/SPs/gp_with_der.py')
covs = imp.load_source('rationalQuadratic', '/home/ulli/VentIPyN/Experiments/covFunctions_noLog.py')
import matplotlib
from ShiftedColorMap import shiftedColorMap
import seaborn as sns

###############################################
#############Test GP Utils     ################
###############################################

def array(xs):
  return t.VentureArrayUnboxed(np.array(xs),  t.NumberType())

def makeObservations(x,y,ripl):
    xString = genSamples(x)
    ripl.observe(xString, array(y))

def genSamples(x):
    sampleString='(gp (array '
    for i in range(len(x)):
        sampleString+= str(x[i]) + ' '
    sampleString+='))'
    #print(sampleString)
    return sampleString


###############################################################
############# synthetic data generating functions  ############
###############################################################

def f(x):
    return 0.3 + 0.4*x + 0.5*np.sin(2.7*x) + (1.1/(1+x**2))

def f_periodic(x):
    return 5*np.sin(x)

def f_sqrt(x):
    return np.sqrt(x)
###############################################################
############# KL matrices; asymptotic covnergence ############
############################################### ###############


# add the moment, all I do with the matrices is plotting them

# find this on stats.stackexchange
def KL_normal(mu_1,sigma_1,mu_2,sigma_2):
  return np.log(sigma_2/sigma_1) + (sigma_1**2 +(mu_1-mu_2)**2 )/( 2 * sigma_2**2) - 0.5


# find this on stat.stackexchange and wikipedia
def KL_mvn(mu_0,Sigma_0,mu_1,Sigma_1):
  k = Sigma_0.shape[0]
  Sigma_1_inv = np.linalg.inv(Sigma_1)
  y = np.asmatrix(mu_1 - mu_0)
  return 0.5 * (np.trace(Sigma_1_inv*Sigma_0)  + y.T * Sigma_1_inv * y - k + np.log(np.linalg.det(Sigma_1)/np.linalg.det(Sigma_0)))



###############################################################
############# Actual Tests                         ############
###############################################################

def test_gp_inference():

  observations_n = range(10,100,2)
  number_steps =80
  ripl = shortcuts.make_lite_church_prime_ripl()
  every_n_step=1
  kl_matrix=np.zeros((len(observations_n),number_steps/every_n_step))

  for n_i in range(len(observations_n)):
    n = observations_n[n_i]
    x = np.random.uniform(-3,3,n)
    y = f(x) + np.random.normal(0,0.1,n)
    ripl.clear()
    ripl.bind_foreign_sp("make_gp_part_der",gp_w_der.makeGPSP)
    ripl.assume('make_const_func', VentureFunction(covs.makeConstFunc, [t.NumberType()], covs.constantType))
    ripl.assume('zero', "(apply_function make_const_func 0)")

    ripl.assume("func_plus", covs.makeLiftedAdd(lambda x1, x2: x1 + x2))

    ripl.assume('make_se',VentureFunction(covs.makeSquaredExponential,[t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
    ripl.assume('make_noise',VentureFunction(covs.makeNoise,[t.NumberType()], t.AnyType("VentureFunction")))
    ripl.assume('alpha_sf','(tag (quote hyperhyper) 0 (gamma 7 1))')
    ripl.assume('beta_sf','(tag (quote hyperhyper) 2 (gamma 2 0.5))')
    ripl.assume('alpha_l','(tag (quote hyperhyper) 1 (gamma 7 1))')
    ripl.assume('beta_l','(tag (quote hyperhyper) 3 (gamma 1 0.5))')


    ripl.assume('sf','(tag (quote hyper) 0 (gamma alpha_sf beta_sf ))')
    ripl.assume('l','(tag (quote hyper) 1 (gamma alpha_l beta_l ))')

    ripl.assume('sigma','0.1')
    ripl.assume('l_sigma','sigma')

    ripl.assume('se', "(apply_function make_se sf l )")
    ripl.assume('wn','(apply_function make_noise sigma  )')
    ripl.assume('gp',"""(tag (quote model) 0
                        (make_gp_part_der zero (apply_function func_plus se wn  )
                                )

                             )""")
    makeObservations(x,y,ripl)
    for steps in range(number_steps):
      if (steps % every_n_step )==0:
        xpost = 0.5
        ypost = []
        for i in range(100):
            y = ripl.sample("(gp (array " + str(xpost) + " ))")
            ypost.append(y)
        kl_matrix[n_i][steps/every_n_step]= KL_normal(np.mean(ypost),np.std(ypost),f(xpost),0.1)
      ripl.infer("(do (mh (quote hyperhyper) one 2) (mh (quote hyper) one 1))")
  orig_cmap = matplotlib.cm.coolwarm
  shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0.3, name='shifted')  
  sns.heatmap(kl_matrix,cmap=shifted_cmap,yticklabels=observations_n)
  sns.plt.show()
  max_kl = kl_matrix.max()
  shift = 1./max_kl
  heavily_shifted_cmap = shiftedColorMap(orig_cmap, midpoint=shift, name='shifted')  
  sns.heatmap(kl_matrix,cmap=  heavily_shifted_cmap ,yticklabels=observations_n)
  sns.plt.show()

def test_gp_inference_lin():

  observations_n = range(10,50,1)
  number_steps =50
  ripl = shortcuts.make_lite_church_prime_ripl()
  every_n_step=1
  kl_matrix=np.zeros((len(observations_n),number_steps/every_n_step))

  for n_i in range(len(observations_n)):
    n = observations_n[n_i]
    x = np.random.uniform(-30,30,n)
    slope = 2
    intercept = 5 
    y = slope * x + intercept + np.random.normal(0,0.1,n)
    ripl.clear()
    ripl.bind_foreign_sp("make_gp_part_der",gp_w_der.makeGPSP)
    ripl.assume('make_const_func', VentureFunction(covs.makeConstFunc, [t.NumberType()], covs.constantType))
    ripl.assume('zero', "(apply_function make_const_func 0)")

    ripl.assume("func_plus", covs.makeLiftedAdd(lambda x1, x2: x1 + x2))

    ripl.assume('make_lin',VentureFunction(covs.makeLinear,[t.NumberType()], t.AnyType("VentureFunction")))
    ripl.assume('make_constant',VentureFunction(covs.makeConst,[t.NumberType()], t.AnyType("VentureFunction")))
    ripl.assume('make_noise',VentureFunction(covs.makeNoise,[t.NumberType()], t.AnyType("VentureFunction")))
    
    

    ripl.assume('slope','(tag (quote hyper) 0 (uniform_continuous 0 100 ))')
    ripl.assume('intercept','(tag (quote hyper) 1 (uniform_continuous 0 100 ))')

    ripl.assume('sigma','0.1')

    ripl.assume('lin', "(apply_function make_lin slope )")
    ripl.assume('wn','(apply_function make_noise sigma  )')
    ripl.assume('c','(apply_function make_constant intercept  )')
    ripl.assume('gp',"""(tag (quote model) 0
                        (make_gp_part_der zero (apply_function func_plus (apply_function func_plus lin c) wn)
                        )
                             )""")
    makeObservations(x,y,ripl)
    for steps in range(number_steps):
      if (steps % every_n_step )==0:
        xpost = 31
        ypost = []
        for i in range(100):
            y = ripl.sample("(gp (array " + str(xpost) + " ))")
            ypost.append(y)
        kl_matrix[n_i][steps/every_n_step]= KL_normal(np.mean(ypost),np.std(ypost),slope*xpost+intercept,0.1)
      ripl.infer("(mh (quote hyper) one 1)")
  orig_cmap = matplotlib.cm.coolwarm
  shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0.3, name='shifted')  
  sns.heatmap(kl_matrix,cmap=shifted_cmap,yticklabels=observations_n)
  sns.plt.show()
  max_kl = kl_matrix.max()
  shift = 1./max_kl
  heavily_shifted_cmap = shiftedColorMap(orig_cmap, midpoint=shift, name='shifted')  
  sns.heatmap(kl_matrix,cmap=  heavily_shifted_cmap ,yticklabels=observations_n)
  sns.plt.show()



def test_gp_inference_per():
  # few to many observations. Less than 4 here normally crashes to due to bad covaricance matrices
  observations_n = range(20,50,2)
  number_steps =100
  ripl = shortcuts.make_lite_church_prime_ripl()
  every_n_step=1
  kl_matrix=np.zeros((len(observations_n),number_steps/every_n_step))

  for n_i in range(len(observations_n)):
    n = observations_n[n_i]
    x = np.random.uniform(0,30,n)
    y = f_periodic(x)# + np.random.normal(0,0.1,n)
    ripl.clear()
    ripl.bind_foreign_sp("make_gp_part_der",gp_w_der.makeGPSP)
    ripl.assume('make_const_func', VentureFunction(covs.makeConstFunc, [t.NumberType()], covs.constantType))
    ripl.assume('zero', "(apply_function make_const_func 0)")



    ripl.assume('make_per',VentureFunction(covs.makePeriodic,[t.NumberType(), t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))

    ripl.assume('make_noise',VentureFunction(covs.makeNoise,[t.NumberType()], t.AnyType("VentureFunction")))

    ripl.assume("func_plus", covs.makeLiftedAdd(lambda x1, x2: x1 + x2))

    ripl.assume('sf','(tag (quote hyper) 0 (uniform_continuous 0 100 ))')
    ripl.assume('l','(tag (quote hyper) 1 (uniform_continuous 0 100 ))')
    ripl.assume('p','(tag (quote hyper) 2 (uniform_continuous 0.01 100 ))')


    ripl.assume('sigma','0.1')

    ripl.assume('per', "(apply_function make_per sf p l )")
    ripl.assume('wn', "(apply_function make_noise sigma )")
    ripl.assume('gp',"""(tag (quote model) 0
                        (make_gp_part_der zero (apply_function func_plus per wn  )
                                )

                             )""")
   

    makeObservations(x,y,ripl)
    for steps in range(number_steps):
      if (steps % every_n_step )==0:
        xpost = np.random.uniform(33,36,1)[0]
        ypost = []
        for i in range(100):
            y = ripl.sample("(gp (array " + str(xpost) + " ))")
            ypost.append(y)
        kl_matrix[n_i][steps/every_n_step]= KL_normal(np.mean(ypost),np.std(ypost),f(xpost),0.1)
      ripl.infer("(mh (quote hyper) one 1)")
  orig_cmap = matplotlib.cm.coolwarm
  shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0.3, name='shifted')  
  sns.heatmap(kl_matrix,cmap=shifted_cmap,yticklabels=observations_n)
  sns.plt.show()
  max_kl = kl_matrix.max()
  shift = 1./max_kl
  heavily_shifted_cmap = shiftedColorMap(orig_cmap, midpoint=shift, name='shifted')  
  sns.heatmap(kl_matrix,cmap=  heavily_shifted_cmap ,yticklabels=observations_n)
  sns.plt.show()

def test_gp_inference_smooth():
  observations_n = range(10,100,10)
  number_steps =100
  ripl = shortcuts.make_lite_church_prime_ripl()
  every_n_step=1
  kl_matrix=np.zeros((len(observations_n),number_steps/every_n_step))

  for n_i in range(len(observations_n)):
    n = observations_n[n_i]
    x = np.linspace(0,50,n)
    y = f_sqrt(x) + np.random.normal(0,1,n)
    ripl.clear()
    ripl.bind_foreign_sp("make_gp_part_der",gp_w_der.makeGPSP)
    ripl.assume('make_const_func', VentureFunction(covs.makeConstFunc, [t.NumberType()], covs.constantType))
    ripl.assume('zero', "(apply_function make_const_func 0)")

    ripl.assume("func_plus", covs.makeLiftedAdd(lambda x1, x2: x1 + x2))

    ripl.assume('make_se',VentureFunction(covs.makeSquaredExponential,[t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
    ripl.assume('make_noise',VentureFunction(covs.makeNoise,[t.NumberType()], t.AnyType("VentureFunction")))
    ripl.assume('alpha_sf','(tag (quote hyperhyper) 0 (gamma 7 1))')
    ripl.assume('beta_sf','(tag (quote hyperhyper) 2 (gamma 1 0.5))')
    ripl.assume('alpha_l','(tag (quote hyperhyper) 1 (gamma 7 1))')
    ripl.assume('beta_l','(tag (quote hyperhyper) 3 (gamma 1 0.5))')


    ripl.assume('sf','(tag (quote hyper) 0 (gamma alpha_sf beta_sf ))')
    ripl.assume('l','(tag (quote hyper) 1 (gamma alpha_l beta_l ))')

    ripl.assume('sigma','0.1')
    ripl.assume('l_sigma','sigma')

    ripl.assume('se', "(apply_function make_se sf l )")
    ripl.assume('wn','(apply_function make_noise sigma  )')
    ripl.assume('gp',"""(tag (quote model) 0
                        (make_gp_part_der zero (apply_function func_plus se wn  )
                                )

                             )""")
    makeObservations(x,y,ripl)
    for steps in range(number_steps):
      ripl.infer("(do (mh (quote hyperhyper) one 2) (mh (quote hyper) one 1))")
      if (steps % every_n_step )==0:
        xpost = 51
        ypost = []
        for i in range(100):
            y = ripl.sample("(gp (array " + str(xpost) + " ))")
            ypost.append(y)
        kl_matrix[n_i][steps/every_n_step]= KL_normal(np.mean(ypost),np.std(ypost),f(xpost),0.1)
  sns.heatmap(kl_matrix,cmap="coolwarm",yticklabels=observations_n)
  
  sns.plt.show()
  cleaned_matrix = []
  print(kl_matrix.max())
  print(kl_matrix.min())
  for n_i in range(len(observations_n)):
    if  kl_matrix[n_i,:].max()<5000:
      cleaned_matrix.append( kl_matrix[n_i,:])
  print(cleaned_matrix)
  sns.heatmap(cleaned_matrix,cmap="coolwarm")
  sns.plt.show()


def test_gp_inference_uniform():
  observations_n = range(10,40)
  number_steps =6666660
  ripl = shortcuts.make_lite_church_prime_ripl()
  every_n_step=1
  kl_matrix=np.zeros((len(observations_n),number_steps/every_n_step))

  for n_i in range(len(observations_n)):
    n = observations_n[n_i]
    x = np.random.uniform(0,1,n)
    y = f(x) + np.random.normal(0,0.1,n)
    ripl.clear()
    ripl.bind_foreign_sp("make_gp_part_der",gp_w_der.makeGPSP)
    ripl.assume('make_const_func', VentureFunction(covs.makeConstFunc, [t.NumberType()], covs.constantType))
    ripl.assume('zero', "(apply_function make_const_func 0)")

    ripl.assume("func_plus", covs.makeLiftedAdd(lambda x1, x2: x1 + x2))

    ripl.assume('make_se',VentureFunction(covs.makeSquaredExponential,[t.NumberType(), t.NumberType()], t.AnyType("VentureFunction")))
    ripl.assume('make_noise',VentureFunction(covs.makeNoise,[t.NumberType()], t.AnyType("VentureFunction")))


    ripl.assume('sf','(tag (quote hyper) 0 (uniform_continuous 0 100 ))')
    ripl.assume('l','(tag (quote hyper) 1 (uniform_continuous 0  100 ))')

    ripl.assume('sigma','0.1')
    ripl.assume('l_sigma','sigma')

    ripl.assume('se', "(apply_function make_se sf l )")
    ripl.assume('wn','(apply_function make_noise sigma  )')
    ripl.assume('gp',"""(tag (quote model) 0
                        (make_gp_part_der zero (apply_function func_plus se wn  )
                                )

                             )""")
    makeObservations(x,y,ripl)
    for steps in range(number_steps):
      ripl.infer("(mh (quote hyper) one 1)")
      if (steps % every_n_step )==0:
        xpost = 0.5
        ypost = []
        for i in range(100):
            y = ripl.sample("(gp (array " + str(xpost) + " ))")
            ypost.append(y)
        kl_matrix[n_i][steps/every_n_step]= KL_normal(np.mean(ypost),np.std(ypost),f(xpost),0.1)
  sns.heatmap(kl_matrix,cmap="coolwarm",yticklabels=observations_n)
  sns.plt.show()
 
