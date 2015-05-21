from plotting import load_experiments,get_last_n_residuals

experimental_df = load_experiments("experiment_test_residualgrep.ini","2015-05-20")

#print(experimental_df)

d,b=get_last_n_residuals("experiment_test_residualgrep.ini","2015-05-20",100)
print(d)

print(b)