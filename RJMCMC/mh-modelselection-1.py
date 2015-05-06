
# In[1]:

from venture import shortcuts
import numpy as np
import pylab as pl
import seaborn
import time


# In[2]:




# Out[2]:

#     Populating the interactive namespace from numpy and matplotlib
# 

# In[3]:

ripl = shortcuts.make_lite_church_prime_ripl()


# In[4]:

two_components=True


# In[5]:

if two_components:
    data_generating_program = """
    [ASSUME gauss1 (lambda (  ) (normal 3 2))]
    [ASSUME gauss2 (lambda (  ) (normal -3 1))]
    [ASSUME data (lambda ( ) ((categorical (simplex 0.4 0.6) (array gauss1 gauss2))))]
    """
else:
    data_generating_program = """
    [ASSUME gauss1 (lambda (  ) (normal 5 1))]
    [ASSUME gauss2 (lambda (  ) (normal 0 3))]
    [ASSUME gauss2 (lambda (  ) (normal -6 1))]
    [ASSUME data (lambda ( ) ((categorical (simplex 0.3 0.3 0.4) (array gauss1 gauss2 gauss3))))]
    """
ripl.execute_program(data_generating_program)


# Out[5]:

#     [{'directive_id': 1, 'value': {'type': 'sp', 'value': 'unknown spAux'}},
#      {'directive_id': 2, 'value': {'type': 'sp', 'value': 'unknown spAux'}},
#      {'directive_id': 3, 'value': {'type': 'sp', 'value': 'unknown spAux'}}]

# In[6]:

data = []
n = 100
for i in range(n):
    data.append(ripl.sample("(data)"))


# In[7]:

pl.hist(data)


# Out[7]:

#     (array([ 10.,  22.,  26.,   8.,   2.,   9.,   8.,   6.,   6.,   3.]),
#      array([-5.60954572, -4.31492325, -3.02030078, -1.72567831, -0.43105584,
#             0.86356663,  2.1581891 ,  3.45281157,  4.74743404,  6.04205651,
#             7.33667898]),
#      <a list of 10 Patch objects>)

# image file:

# In[8]:

#ripl.infer("(resample 10)")


# In[9]:

program = """
[ASSUME d  (scope_include (quote hypers) 0 (make_uc_sym_dir_mult 0.5 2 (array 2 3)))]
[ASSUME mixing  (scope_include (quote hypers) 1 (make_uc_sym_dir_mult 0.5  (d)))]
[ASSUME g 1]
[ASSUME h 1]
[ASSUME alpha 1]



[ASSUME get_mean (mem (lambda (component)
  (scope_include (quote parameters) component (normal 0 10))))]

[ASSUME get_beta (mem (lambda (component)
  (scope_include (quote parameters) component (gamma g h))))]

[ASSUME get_variance (mem (lambda (component)
  (scope_include (quote parameters) component (gamma alpha (get_beta component)))))]


[ASSUME get_datapoint (lambda (i)    (normal (get_mean (mixing)) (get_variance (mixing)))) ]

"""
ripl.execute_program(program)


# Out[9]:

#     [{'directive_id': 104,
#       'value': {'type': 'sp',
#        'value': {'alpha': 0.5,
#         'counts': [0, 0],
#         'n': 2,
#         'type': 'uc_sym_dir_mult'}}},
#      {'directive_id': 105,
#       'value': {'type': 'sp',
#        'value': {'alpha': 0.5,
#         'counts': [0, 0, 0],
#         'n': 3,
#         'type': 'uc_sym_dir_mult'}}},
#      {'directive_id': 106, 'value': {'type': 'number', 'value': 1.0}},
#      {'directive_id': 107, 'value': {'type': 'number', 'value': 1.0}},
#      {'directive_id': 108, 'value': {'type': 'number', 'value': 1.0}},
#      {'directive_id': 109, 'value': {'type': 'sp', 'value': 'unknown spAux'}},
#      {'directive_id': 110, 'value': {'type': 'sp', 'value': 'unknown spAux'}},
#      {'directive_id': 111, 'value': {'type': 'sp', 'value': 'unknown spAux'}},
#      {'directive_id': 112, 'value': {'type': 'sp', 'value': 'unknown spAux'}}]

# In[10]:


ripl.infer("""(let ((ds (empty)))
   (do (repeat 1
        (do (sample  (d))
            (bind (collect (d) ) (curry into ds))))
       (plotf (quote (h0)) ds)))""")


# Out[10]:

#     stat_bin: binwidth defaulted to range/30.
#         Use 'binwidth = x' to adjust this.
# 

# image file:

#     []

# In[11]:

for i in range(n):
    ripl.observe("(get_datapoint %d )"% i,data[i]) 


# In[12]:

ripl.infer("(repeat 10 (do (mh hypers all 2) (mh parameters one 3) ))")


# Out[12]:

#     []

# In[13]:

ripl.sample("(d)")


# Out[13]:

#     2.0

# In[14]:

#dataset=ripl.infer("(collect (d) )")
#df =dataset.asPandas()


# In[15]:


ripl.infer("""(let ((ds (empty)))
   (do (repeat 1
        (do (sample  (d))
            (bind (collect (d) ) (curry into ds))))
       (plotf (quote (h0)) ds)))""")


