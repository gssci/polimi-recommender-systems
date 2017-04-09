from __future__ import division
import numpy as np
import pandas as pd
import graphlab as gl
from scipy import sparse
from timedecay import pezdict

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

user_profiles = pd.read_csv("user_profile.csv", delimiter='\t')
target_users = pd.read_csv("target_users.csv", delimiter='\t')
all_users = user_profiles['user_id'].values
target = target_users['user_id'].values

item_profiles = pd.read_csv("item_profile.csv", delimiter='\t')
ids = item_profiles['id'].values
items_inattivi = item_profiles[item_profiles['active_during_test'] == 0]['id'].values

index_to_ids = dict(zip(range(ids.size),ids))
ids_to_index = {v: k for k, v in index_to_ids.items()}
index_to_uds = dict(zip(range(all_users.size),all_users))
uds_to_index = {v: k for k, v in index_to_uds.items()}

interactions = pd.read_csv('training_data.csv', delimiter="\t")

def pez_decay(ud,id):
    return pezdict.get((ud,id),0)

interactions['time'] = list(map(pez_decay, interactions['user_id'].values, interactions['item_id'].values))
observations = gl.SFrame(interactions)

model_UserBased = gl.recommender.item_similarity_recommender.create(observations, user_id='item_id', item_id='user_id', target='time',
                                                               user_data=None, item_data=None, similarity_type='cosine', only_top_k=1700, verbose=True)
R_userbased = sparse.lil_matrix((40000,167956))

remaining = ids.size

def interagitori(item_id):
    return np.unique(interactions[interactions['item_id'] == item_id]['user_id'].values)

for itemid in ids:
    vs = interagitori(itemid)

    if vs.size > 0:
        suggestions = model_UserBased.get_similar_items(list(vs),k=1000).to_dataframe()
        suggestions = suggestions.groupby(['similar','user_id']).aggregate(np.sum).reset_index()

        R_userbased[[uds_to_index.get(k) for k in suggestions['similar'].values], ids_to_index.get(itemid)] += np.matrix(suggestions['score'].values).T
    remaining -= 1
    print(str(remaining))

R_userbased = R_userbased.tocsr()
R_userbased = R_userbased[[uds_to_index.get(t) for t in target],:]
save_sparse_csr('CF_UserBased',R_userbased)
