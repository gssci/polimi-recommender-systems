from __future__ import division
import numpy as np
import pandas as pd
import graphlab as gl
import nltk
from scipy import sparse as sps
from sklearn import preprocessing

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

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )
    return

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sps.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

model = gl.load_model('edu_fieldofstudies')

target_users = np.genfromtxt("target_users.csv", delimiter="\t", dtype=np.dtype(int), skip_header=1)
usersals = pd.read_csv("user_profile.csv", delimiter="\t")
itemsals = pd.read_csv("item_profile.csv", delimiter="\t")
interactions = pd.read_csv("interactions.csv", delimiter="\t")
uds_all = usersals['user_id'].values
# PRE-PROCESSING

# drop useless columns
inter = interactions.copy()
inter = inter.drop(['interaction_type', 'created_at'], 1)
inter = inter.drop_duplicates()
inter['r'] = 1

# users and items dict : associate user/item id with index
users_dict = dict(zip(usersals['user_id'], usersals.index))
items_dict = dict(zip(itemsals['id'], itemsals.index))
inv_users_dict = {v: k for k, v in users_dict.items()}  # inverse of users_dict
inv_items_dict = {v: k for k, v in items_dict.items()}  # inverse of items_dict

# add users/items indexes columns
inter['user_index'] = [users_dict.get(i) for i in inter['user_id']]
inter['item_index'] = [items_dict.get(i) for i in inter['item_id']]

# non-active items ids/indexes
items_nact_ids = itemsals[itemsals['active_during_test'] == 0]['id'].drop_duplicates().values
items_nact = [items_dict.get(i) for i in items_nact_ids]


# URM : user-rating matrix
urm = sps.csr_matrix((inter['r'], (inter['user_index'], inter['item_index'])))
urm = urm.astype(float)

sims = model.get_similar_items(list(users),k=500)
similar = sims.to_dataframe()

R_out = sps.lil_matrix((10000,167956))
remaining = len(users)

for a in users:
    simili = similar[similar['user_id'] == a]

    if len(simili) > 0:
        s = simili['similar'].values
        v = simili['score'].values

        rec = urm[uds_to_index2.get(s[0]),:].toarray().ravel() * v[0]

        for j in range(len(s)-1):
            rec += urm[uds_to_index2.get(s[j+1]),:].toarray().ravel() * v[j+1]

        #rec = np.dot(simili['score'].values.T,urm[[uds_to_index2.get(v) for v in simili['similar'].values], :].toarray())
        cols = np.argsort(rec)[::-1][:10000]
        rec[items_nact] = 0
        if len(rec) > 0:
            rec = np.true_divide(rec,np.max(rec))
        R_out[uds_to_index.get(a),cols] = rec[cols]
        remaining -= 1
        print(str(remaining))

save_sparse_csr('RFS_ALL',R_out.tocsr())
print("done")