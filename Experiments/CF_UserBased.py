from __future__ import division
from timedecay import similarity
import numpy as np
import pandas as pd
import graphlab as gl
from scipy import sparse

items0 = pd.read_csv("item_profile.csv", delimiter='\t')
target_users = pd.read_csv("target_users.csv", delimiter='\t')
interactions = pd.read_csv('interactions.csv', delimiter='\t')
user_profile = pd.read_csv('user_profile.csv', delimiter='\t')
ids = items0['id'].values
users = target_users['user_id'].values
index_to_ids = dict(zip(range(ids.size),ids))
ids_to_index = {v: k for k, v in index_to_ids.items()}
index_to_uds = dict(zip(range(users.size),users))
uds_to_index = {v: k for k, v in index_to_uds.items()}

uds2 = user_profile['user_id'].values
index_to_uds2 = dict(zip(range(uds2.size),uds2))
uds_to_index2 = {v: k for k, v in index_to_uds2.items()}

def interagiti(user_id):
    return np.unique(interactions[interactions['user_id'] == user_id]['item_id'].values)

R_out = sparse.lil_matrix((10000,167956))
n = 10000

for u in users:
    seen = interagiti(u)

    if seen.size > 0:
        simi_rows = similarity[[ids_to_index.get(i) for i in seen]]
        rec = simi_rows.sum(axis=0)
        aux = simi_rows.power(0)
        aux = aux.sum(axis=0)
        N = np.multiply(rec,aux)
        D = aux + 80
        rec = np.true_divide(N,D)
        rec = np.true_divide(rec,np.max(rec))
        R_out[uds_to_index.get(u)] = rec
    n -= 1
    print(str(n))

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

save_sparse_csr('Collaborative_ALL', R_out.tocsr())