import pylab as pl
import seaborn as sns
from plotting import get_last_n_residuals
import pandas as pd
df,b=get_last_n_residuals("experiment_test_residualgrep.ini","2015-05-20",10)


print(df)

#sns.boxplot(df['residuals'],df['model'],ylim=(05))


'''
g = sns.FacetGrid(pd.melt(df, id_vars='test_problem'), col='test_problem')
g.map(sns.boxplot, 'residuals', 'model')

pl.show()
'''