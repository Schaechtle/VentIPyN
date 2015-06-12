import itertools
import numpy as np
base_kernels =['LIN','SE','PER','RQ']
local_interaction = list(itertools.combinations(base_kernels, 2))
local_interaction_pp = [combi[0]+' x '+combi[1] for combi in local_interaction]

print(local_interaction)
# count all the base kernels,

# count all the local interactions
# build all local interactions via all subsets
# split at +, look for occurences of entries in the powerset_n
# for each number of global components, return a count of base kernels
# for each number of global components return a local interaction



def count_base(structure_str):
    counts = []
    for base in base_kernels:
        if base in structure_str:
            counts.append(1)
        else:
            counts.append(0)
    return np.array(counts)

def count_local_interactions(structure_str):
    counts = []
    products = structure_str.split('+')
    for interaction in local_interaction:
        for product in products:
            if (interaction[0] in structure_str) and  (interaction[0] in structure_str):
                counts.append(1)
                break
            else:
                counts.append(0)
    return np.array(counts)



# something that takes a df in and returns twice two lists  (counts labels(base))   (counts labels(interaction))
# should normalise between 0 and 1
def structure_marginal(df):
    base_counts = np.empty(len(base_kernels))
    interaction_counts = np.empty(len(local_interaction))
    index_i =0
    while next(df.iterrows()) is not None:
        print(index_i)
        index_i+=1
        row = next(df.iterrows())
        structure_str = row[1]['Covariance Structure']
        base_counts+=count_base(structure_str)
        interaction_counts+=count_local_interactions(structure_str)
    return (base_counts,base_kernels),(interaction_counts,local_interaction_pp)










