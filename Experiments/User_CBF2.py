from __future__ import division
import numpy as np
import pandas as pd
import graphlab as gl
import nltk
from scipy import sparse
from sklearn import preprocessing
scaler = preprocessing.MaxAbsScaler()

items0 = pd.read_csv("item_profile.csv", delimiter='\t')
target_users = pd.read_csv("target_users.csv", delimiter='\t')
interactions = pd.read_csv('interactions.csv', delimiter='\t')
user_profile = pd.read_csv('user_profile.csv', delimiter='\t')
di = {'de': 1, "at": 2, "ch":3, "non_dach":0}
user_profile = user_profile.replace({'country':di})
user_profile = user_profile.fillna(0)
users = target_users['user_id'].values
ids = items0['id'].values
index_to_ids = dict(zip(range(ids.size),ids))
ids_to_index = {v: k for k, v in index_to_ids.items()}
index_to_uds = dict(zip(range(users.size),users))
uds_to_index = {v: k for k, v in index_to_uds.items()}

uds2 = user_profile['user_id'].values
index_to_uds2 = dict(zip(range(uds2.size),uds2))
uds_to_index2 = {v: k for k, v in index_to_uds2.items()}

def interagiti(user_id):
    return np.unique(interactions[interactions['user_id'] == user_id]['item_id'].values)

anonimi = [u for u in users if interagiti(u).size<=0]
anonimi_indices = [uds_to_index.get(a) for a in anonimi]

to_compute = ['career_level']


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)
    return

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                             shape=loader['shape'])

ALS = load_sparse_csr('ALS4k_ALL.npz')
ALS = scaler.fit_transform(ALS.T).T

for attr in to_compute:
    us = user_profile[['user_id',attr]]
    us = us.fillna(0)
    us[attr] = us[attr].astype(int)
    data = us[us[attr] > 0]
    data = gl.SFrame(data)

    model = gl.recommender.item_similarity_recommender.create(data,user_id=attr,item_id='user_id',similarity_type='jaccard')

    R_sorgente = ALS
    R_out = sparse.lil_matrix((10000,167956))
    remaining = np.size(users)

    sims = model.get_similar_items(list(users),k=100)
    similar = sims.to_dataframe()

    for a in users:
        simili = similar[similar['user_id'] == a]
        rec =  np.dot(R_sorgente[[uds_to_index2.get(v) for v in simili['similar'].values]].toarray().T,simili['score'].values)
        cols = np.argsort(rec)[::-1][:1000]
        R_out[uds_to_index.get(a), cols] = rec[cols]
        remaining -= 1
        print(str(remaining))

    save_sparse_csr('R_ano_' + attr + '_ALL',R_out.tocsr())



