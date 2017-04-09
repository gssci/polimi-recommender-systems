import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.preprocessing import MinMaxScaler
import map5

from scipy.sparse.linalg import svds
import math as mt

from timestamp_processing import urm_time

target_users = np.genfromtxt("target_users.csv", delimiter="\t", dtype=np.dtype(int), skip_header=1)
users = pd.read_csv("user_profile.csv", delimiter="\t")
items = pd.read_csv("item_profile.csv", delimiter="\t")
# interactions = pd.read_csv("interactions.csv", delimiter="\t")
interactions = pd.read_csv("interactions.csv", delimiter="\t")

def interagiti(user_id):
    return np.unique(interactions[interactions['user_id'] == user_id]['item_id'].values)

rusers = users['user_id'].values
attivi = [u for u in rusers if interagiti(u).size>0]

# users and items dict : associate user/item id with index
users_dict = dict(zip(users['user_id'], users.index))
items_dict = dict(zip(items['id'], items.index))
inv_users_dict = {v: k for k, v in users_dict.items()}  # inverse of users_dict
inv_items_dict = {v: k for k, v in items_dict.items()}  # inverse of items_dict

# non-active items ids/indexes
items_nact_ids = items[items['active_during_test'] == 0]['id'].drop_duplicates().values
items_nact = [items_dict.get(i) for i in items_nact_ids]

# SVD : Singolar Value Decomposition

# submission matrix
subm = []
matr = sps.lil_matrix((40000, 167956))

def svd():
    # u, s, vt = sparsesvd(urm, 2000)
    u, s, vt = svds(urm_time, 1500)

    U = sps.csc_matrix(u)
    S = sps.csc_matrix(np.diag(s))
    Vt = sps.csc_matrix(vt)

    SVt = S * Vt

    l = len(attivi)
    for u_id in attivi:
        # index of user id u_id
        u = users_dict.get(u_id)
        # dot product of user vector with all item vectors
        rec_vector = (U[u, :] * SVt).toarray()
        # scale recommendation vector rec_vector between 0 and 1
        min_max = MinMaxScaler()
        rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1, 1))[:, 0]

        # already interacted and non-active items indexes multiplied by 0
        rec = rec_vector_scaled

        # takes 5 best items indexes
        ritem_indexes = np.argsort(rec)[::-1][:1000]

        matr[users_dict.get(u_id), ritem_indexes] = rec[ritem_indexes]

        l -= 1
        print(l)
    save_sparse_csr('SVD4k_ALL', matr.tocsr())


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

