
# In[1]:

import seaborn as sns
import pylab as pl
from plotting import load_experiments
import numpy as np


# In[2]:

sns.set(font_scale=2) 


# In[3]:

#get_ipython().magic(u'pylab inline')


# Out[3]:

#     Populating the interactive namespace from numpy and matplotlib
# 

# In[4]:

experiment_ini_file ="experiment2plot.ini"


# In[5]:

date_experiment = "2015-05-23"
#date_experiment = "2015-05-20"


# In[6]:

experimental_df = load_experiments(experiment_ini_file,date_experiment)


# Out[6]:



### True Function vs Amount of Data

# In[ ]:

order=[str(item) for item in experimental_df['n'].unique()]


# In[ ]:

sns.set(font_scale=1) 
g = sns.FacetGrid(experimental_df.loc[experimental_df['n'] == '50'], row="test_problem", col="noise", margin_titles=True,size=3, aspect=3,hue="model")
g.map(pl.plot,"logscore");
g.add_legend();
g.set_axis_labels("MCMC Steps", "Log-likelihood");
#g.set(ylim=(-500, 0));
g.savefig("test.png", dpi=300)


# In[ ]:

pl.show()

'''
# In[ ]:


g = sns.FacetGrid(experimental_df.loc[experimental_df['test_problem'] == 'function-per'], row="n", col="noise", margin_titles=True,size=8, aspect=1,hue="model",row_order=order)
g.map(pl.plot,"logscore");
g.add_legend();
g.set_axis_labels("MCMC Steps", "Log-likelihood");
g.set(ylim=(-200, 0));


# In[ ]:

g = sns.FacetGrid(experimental_df.loc[experimental_df['test_problem'] == 'function-linear'], row="n", col="noise", margin_titles=True,size=7, aspect=1,hue="model",row_order=order)
g.map(pl.plot,"logscore");
g.add_legend();
g.set_axis_labels("MCMC Steps", "Log-likelihood");
g.set(ylim=(-1000, 0));


# In[ ]:

g = sns.FacetGrid(experimental_df.loc[experimental_df['n'] == '30'], row="test_problem", col="noise", margin_titles=True,size=8, aspect=1,hue="model")
g.map(pl.errorbar,"index","residuals");
g.add_legend();
g.set_axis_labels("MCMC Steps", "Residuals");
g.set(xlim=(200,500),ylim=(0,5));


# In[ ]:




# In[ ]:

df_temp=experimental_df.loc[experimental_df['noise'] == '0.7']

g = sns.FacetGrid(experimental_df.loc[experimental_df['noise'] == '0.7'], row="n", col="test_problem", margin_titles=True,size=4, aspect=3.5,hue="model",row_order=order)
g.map(pl.axhline, y=np.mean(df_temp['base-line']), c="black");
g.map(pl.plot,"residuals");
g.set(ylim=(0,2));
g.add_legend();



# In[ ]:

df_temp=experimental_df.loc[experimental_df['model'] == 'venture-cov-learning']
g = sns.FacetGrid(experimental_df.loc[experimental_df['model'] == 'venture-cov-learning'],  col="cycle", row="test_problem", margin_titles=True,size=4, aspect=3.5)
g.map(pl.plot,"logscore");


# In[ ]:

df_temp=experimental_df.loc[experimental_df['model'] == 'venture-cov-learning']
df=df_temp.loc[df_temp['noise'] == '0.1']
g = sns.FacetGrid(df ,  col="n", row="test_problem", margin_titles=True,size=4, aspect=3.5)
g.map(pl.plot,"logscore");


# In[ ]:

df_temp=experimental_df.loc[experimental_df['model'] == 'venture-cov-learning']
df=df_temp.loc[df_temp['noise'] == '0.7']
g = sns.FacetGrid(df ,  col="n", row="test_problem", margin_titles=True,size=4, aspect=3.5)
g.map(pl.plot,"logscore");
g.set(xlim=(0,400),ylim=(-200,200));


# In[ ]:

df1=df.loc[df['n'] == '50']


# In[ ]:

df2=df1.loc[df1['test_problem'] == 'function-linxper']


# In[ ]:

df2[1:120]


# In[ ]:



'''