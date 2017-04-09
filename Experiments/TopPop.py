from __future__ import division
import numpy as np
import pandas as pd
import graphlab as gl
import nltk
from scipy import sparse

user_profiles = pd.read_csv("user_profile.csv", delimiter='\t')
target_users = pd.read_csv("target_users.csv", delimiter='\t')
item_profiles = pd.read_csv("item_profile.csv", delimiter='\t')
interactions = pd.read_csv('training_data.csv', delimiter="\t")

users = user_profiles['user_id'].values
ids = item_profiles['id'].values
index_to_ids = dict(zip(range(ids.size),ids))
ids_to_index = {v: k for k, v in index_to_ids.items()}
index_to_uds = dict(zip(range(users.size),users))
uds_to_index = {v: k for k, v in index_to_uds.items()}

RA = sparse.lil_matrix((40000,167956))
remaining = 40000

interactions = interactions[interactions['interaction_type'] == 1]
interactions = interactions.drop(['user_id','created_at'],1)
interactions = interactions.groupby('item_id').aggregate(np.sum).reset_index().sort_values('interaction_type',ascending=False)

def weight(i):
    return 1 - np.true_divide(1,i**(1/2))

interactions['pop_score'] = map(weight,interactions['interaction_type'].values)
interactions = interactions.sort_values('pop_score',ascending=False)
interactions = interactions.head(1000)
indices = [ids_to_index.get(k) for k in interactions['item_id'].values]
scores = interactions['pop_score'].values
# scores = np.true_divide(scores,np.max(scores))

for u in range(40000):
    RA[u,indices] = scores
    remaining -= 1
    print(str(remaining))

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

save_sparse_csr('./R_TopPopLight',RA.tocsr())