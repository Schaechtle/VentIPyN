import seaborn as sns
import pylab as pl

def remove_massive_outliers(df,replace_by=50.)
  count=0
  for i in range(len(df.index)):
      current =df['residuals'].iloc[i]
      if (current > 50):
          df['residuals'].iloc[i]=replace_by
          count+=1
      if (current < -50):
          df['residuals'].iloc[i]=-replace_by
          count+=1
  print("ratio: "+ str(float(count)/i))
  return df

df = pd.read_pickle("results/n_res_2015-05-23")
 
sns.set_context(rc={"figure.figsize": (8, 8)})
sns.set(font_scale=2) 

order = ['venture-gp-lin','venture-gp-se','venture-gp-per','venture-gp-lin-p-per','venture-gp-lin-t-per','venture_cov_simple','venture_cov_simple_selinper','venture-cov-learning']

residual_list=[]
model_names = []
for i in range(len(order)):
    df_temp=df.loc[df['model'] == order[i]]
    residual_list.append(df_temp['residuals'].tolist())
    model_names.append(order[i])
fig = plt.figure(figsize=(8,8), dpi=200)
g = sns.boxplot(residual_list,names=model_names)
g.set_xticklabels(model_names,rotation=30)
g.set_ylabel("Residuals")
plt.savefig(format='png')
