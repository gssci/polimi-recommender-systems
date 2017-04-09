from __future__ import division
import numpy as np
import pandas as pd
import graphlab as gl
from scipy import sparse

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
index_to_uds = dict(zip(range(target.size),target))
uds_to_index = {v: k for k, v in index_to_uds.items()}

interactions = pd.read_csv('training_data.csv ', delimiter="\t")
interactions = interactions.drop('Unnamed: 0', 1)
#interactions = interactions[interactions['interaction_type'] == 1]
observations = gl.SFrame(interactions)

model_SVD = gl.ranking_factorization_recommender.create(observations, user_id='user_id', item_id='item_id', target=None, num_factors=70,max_iterations=50)

recs = model_SVD.recommend(target,k=1000)
rec = recs.to_dataframe()

RMF = sparse.lil_matrix((10000,167956))

RMF[[uds_to_index.get(u) for u in rec['user_id'].values],[ids_to_index.get(i) for i in rec['item_id'].values]] = rec['score'].values

save_sparse_csr('RMF',RMF.tocsr())
