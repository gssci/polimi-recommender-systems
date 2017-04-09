from __future__ import division
import numpy as np
import pandas as pd
import scipy.sparse as sps
import implicit as impl
from sklearn.preprocessing import MinMaxScaler

target_users = np.genfromtxt("target_users.csv", delimiter="\t", dtype=np.dtype(int), skip_header=1)
users = pd.read_csv("user_profile.csv", delimiter="\t")
items = pd.read_csv("item_profile.csv", delimiter="\t")
interactions = pd.read_csv("training_data.csv", delimiter="\t")

# PRE-PROCESSING

# drop useless columns
inter = interactions.copy()
inter = inter.drop(['interaction_type', 'created_at'], 1)

# count duplicates
inter = inter.groupby(['user_id', 'item_id']).size().reset_index()
inter = inter.rename(columns={0: 'count'})

# users and items dict : associate user/item id with index
users_dict = dict(zip(users['user_id'], users.index))
items_dict = dict(zip(items['id'], items.index))
inv_users_dict = {v: k for k, v in users_dict.items()}  # inverse of users_dict
inv_items_dict = {v: k for k, v in items_dict.items()}  # inverse of items_dict

# add users/items indexes columns
inter['user_index'] = [users_dict.get(i) for i in inter['user_id']]
inter['item_index'] = [items_dict.get(i) for i in inter['item_id']]

# non-active items ids/indexes
items_nact_ids = items[items['active_during_test'] == 0]['id'].drop_duplicates().values
items_nact = [items_dict.get(i) for i in items_nact_ids]

# URM : user-rating matrix
urm = sps.csr_matrix((inter['count'], (inter['user_index'], inter['item_index'])))

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )
    return

# ALS: Alternative Least Squares

alpha = 40
factors = 300
regularization = 0.01
iterations = 20

matr = sps.lil_matrix((len(users), urm.shape[1]))
min_max = MinMaxScaler()
user_vecs, item_vecs = impl.alternating_least_squares((urm*alpha).astype('double'), factors, regularization, iterations)

l = len(users)
for u in range(l):
    # dot product of user vector with all item vectors
    rec_vector = user_vecs[u, :].dot(item_vecs.T)
    rec_vector[items_nact] = 0
    # scale recommendation vector rec_vector between 0 and 1
    rec = min_max.fit_transform(rec_vector.reshape(-1, 1))[:, 0]
    cols = np.argsort(rec)[::-1][:1000]
    matr[u, cols] = rec[cols]
    print(u)

save_sparse_csr('ALS4k', matr.tocsr())
print("done")





